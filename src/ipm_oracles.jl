# Barrier oracles for PSDCone constraints with three evaluation modes:
# - `:dense`: Cholesky-based, exact,
# - `:matrixfree`: Lanczos-based (no factorization, deterministic, exact up to
#   the Lanczos tolerance; keeps O(n²) Lanczos-basis storage),
# - `:trace`: randomized — stochastic Lanczos quadrature for log det,
#   Hutchinson for the gradient contractions (with CG for `X⁻¹v`), and a
#   biased-Hutchinson Gauss-Newton approximation of the barrier Hessian.
#
# These are consumed by NonconvexInteriorPoint.jl through its package
# extension (the semidefinite code lives here, in NonconvexSemidefinite.jl).

using Random: Random
import LinearAlgebra: Symmetric, SymTridiagonal, cholesky, logdet, eigvals, eigen
using IterativeSolvers: IterativeSolvers
using Zygote: Zygote

"""
    PSDBarrierOracle(mode, c, dim, nprobes, lanczos_deg, cg_tol, rng)

Barrier oracle for a PSD constraint `c(x) ⪰ 0` (`c` is callable, returning a
`dim×dim` matrix or a length-`dim²` vector; `LinearFunction`-backed
constraints are dispatched structurally). `mode` is `:dense`, `:matrixfree`
or `:trace`. The `logdet_est` field is the log-determinant estimator used by
the `:trace` mode — a callable `(f!, n, rng) -> estimate` (defaults to
stochastic Lanczos quadrature); the trace mode can use any of the
matrix-free methods of [`logdet_estimate`](@ref).
"""
mutable struct PSDBarrierOracle{C,L,R<:Random.AbstractRNG}
    mode::Symbol
    c::C
    dim::Int
    nprobes::Int
    lanczos_deg::Int
    cg_tol::Float64
    rng::R
    logdet_est::L
    # caches (per-iteration)
    Q::Union{Nothing,Matrix{Float64}} # Lanczos basis (matrixfree)
    T::Union{Nothing,SymTridiagonal{Float64,Vector{Float64}}}
    Bs::Union{Nothing,Vector{Matrix{Float64}}} # Q' Mᵢ Q (matrixfree)
    Xinv::Union{Nothing,Matrix{Float64}} # dense inverse (dense mode)
    probes::Union{Nothing,Vector{Tuple{Vector{Float64},Vector{Float64}}}}
end

function PSDBarrierOracle(
    mode::Symbol,
    c::C,
    dim::Int,
    nprobes::Int,
    lanczos_deg::Int,
    cg_tol::Float64,
    rng::R,
    logdet_est::L=nothing,
) where {C,L,R<:Random.AbstractRNG}
    return PSDBarrierOracle{C,L,R}(
        mode,
        c,
        dim,
        nprobes,
        lanczos_deg,
        cg_tol,
        rng,
        logdet_est,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
    )
end

# The constraint value as a symmetric matrix.
function _oracle_mat(o::PSDBarrierOracle, x)
    v = o.c(x)
    return Symmetric(v isa AbstractMatrix ? Matrix(v) : reshape(vec(v), o.dim, o.dim))
end

# Derivative of X(x) along the i-th coordinate: ∂X/∂xᵢ. (Jacobian-vector
# products: NonconvexCore exposes no JVP, and value_jacobian would
# materialize the dense Jacobian — Zygote's forward mode is kept here.)
function _deriv_block(o::PSDBarrierOracle, x, i::Int)
    if o.c isa LinearFunction
        return reshape(view(o.c.A, :, i), o.dim, o.dim)
    end
    J = Matrix(reshape(Zygote.forward_jacobian(o.c, x)[2], :, length(x)))
    return reshape(J[:, i], o.dim, o.dim)
end

# Derivative of X(x) along the direction w: ∂X/∂x · w.
function _deriv_dir(o::PSDBarrierOracle, x, w::AbstractVector)
    if o.c isa LinearFunction
        return reshape(o.c.A * w, o.dim, o.dim)
    end
    J = Matrix(reshape(Zygote.forward_jacobian(o.c, x)[2], :, length(x)))
    return reshape(J * w, o.dim, o.dim)
end

# ---------------------------------------------------------------------------
# Value
# ---------------------------------------------------------------------------

"""
    psd_barrier_value(o::PSDBarrierOracle, x)

Barrier value `-log det X(x)` in the oracle's mode. Returns `-Inf` when the
matrix is not positive definite in the deterministic modes.
"""
function psd_barrier_value(o::PSDBarrierOracle, x)
    X = _oracle_mat(o, x)
    if o.mode === :dense
        F = cholesky(X; check=false)
        return issuccess(F) ? -logdet(F) : -Inf
    elseif o.mode === :matrixfree
        T, Q = _lanczos(X; tol=1e-10, maxiter=max(2 * o.dim, 20))
        o.Q = Q
        o.T = T
        vals = eigvals(Symmetric(Matrix(T)))
        all(vals .> 0) || return -Inf
        return -sum(log, vals)
    else # :trace
        f! = (y, v) -> mul!(y, X, v)
        if o.logdet_est === nothing
            return -slq_logdet(
                f!, o.dim; lanczos_deg=o.lanczos_deg, nprobes=o.nprobes, rng=o.rng
            )
        end
        return -o.logdet_est(f!, o.dim, o.rng)
    end
end

# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------

"""
    psd_barrier_grad!(o::PSDBarrierOracle, x, g)

Gradient of the barrier `-∇ₓ log det X(x)` written into `g`. Computed from
the adjoint (pullback) of the constraint map with the matrix-space gradient
`-vec(X⁻¹)` — the Jacobian of `c` is never materialized, so the gradient
path stays first-order-compatible.
"""
function psd_barrier_grad!(o::PSDBarrierOracle, x, g::AbstractVector)
    if o.mode === :dense
        X = _oracle_mat(o, x)
        F = cholesky(X)
        Xinv = F \ Matrix{Float64}(I, o.dim, o.dim)
        o.Xinv = Xinv
        dφ = -vec(Matrix(Xinv)) # matrix-space gradient of -log det
        if o.c isa LinearFunction
            mul!(g, o.c.A', dφ)
        else
            _, gr = NonconvexCore.value_gradient(z -> dot(dφ, vec(o.c(z))), x)
            gr === nothing ? fill!(g, 0.0) : copyto!(g, gr)
        end
    elseif o.mode === :matrixfree
        T, Q = _lanczos(_oracle_mat(o, x); tol=1e-10, maxiter=max(2 * o.dim, 20))
        o.T = T
        o.Q = Q
        k = size(Q, 2)
        # Lanczos approximation of the inverse: X⁻¹ ≈ Q T⁻¹ Q'
        Xinv_approx = Q * (Matrix(T) \ Matrix{Float64}(I, k, k)) * Q'
        dφ = -vec(Xinv_approx)
        if o.c isa LinearFunction
            mul!(g, o.c.A', dφ)
        else
            _, gr = NonconvexCore.value_gradient(z -> dot(dφ, vec(o.c(z))), x)
            gr === nothing ? fill!(g, 0.0) : copyto!(g, gr)
        end
    else # :trace
        X = _oracle_mat(o, x)
        z = zeros(o.dim)
        u = similar(z)
        probes = Tuple{Vector{Float64},Vector{Float64}}[]
        fill!(g, 0.0)
        for _ in 1:o.nprobes
            _rademacher!(z, o.rng)
            u, hist = IterativeSolvers.cg(Matrix(X), z; reltol=o.cg_tol, maxiter=10 * o.dim)
            for i in 1:length(x)
                M = _deriv_block(o, x, i)
                g[i] -= dot(u, M * z) / o.nprobes
            end
            push!(probes, (copy(z), copy(u)))
        end
        o.probes = probes
    end
    return g
end

# ---------------------------------------------------------------------------
# Type-aware barrier oracles (BarrierSelection's `sd_type`): :logdet is the
# default; :inverse (tr(X⁻¹)) and :inverse_square (tr(X⁻²)/2) are provided
# for the dense evaluation mode.
# ---------------------------------------------------------------------------

"""
    psd_barrier_value(o::PSDBarrierOracle, x, type::Symbol)

Barrier value of the PSD constraint with the barrier type `type`
(`:logdet`, `:inverse` or `:inverse_square`). The non-log-determinant types
require the dense evaluation mode.
"""
function psd_barrier_value(o::PSDBarrierOracle, x, type::Symbol)
    type === :logdet && return psd_barrier_value(o, x)
    o.mode === :dense || throw(
        ArgumentError(
            "sd barrier type :$(type) requires the dense evaluation mode (mode :$(o.mode))",
        ),
    )
    Xinv = _dense_inv(o, x)
    return if type === :inverse
        tr(Xinv)
    elseif type === :inverse_square
        tr(Xinv * Xinv) / 2
    else
        throw(ArgumentError("unknown sd barrier type :$(type)"))
    end
end

function _dense_inv(o::PSDBarrierOracle, x)
    X = _oracle_mat(o, x)
    T = eltype(X)
    F = cholesky(X)
    Xinv = F \ Matrix{T}(I, o.dim, o.dim)
    Xinv isa Matrix{Float64} && (o.Xinv = Xinv)
    return Xinv
end

# Exponent q of the non-log-determinant barrier tr(X^{-q})/q.
_sd_power(type::Symbol) =
    if type === :inverse
        1
    elseif type === :inverse_square
        2
    else
        throw(ArgumentError("unknown sd barrier type :$(type)"))
    end

"""
    psd_barrier_grad!(o::PSDBarrierOracle, x, g, type::Symbol)

Gradient of the type-`type` PSD barrier. Computed from the adjoint
(pullback) of the constraint map with the matrix-space gradient
(`-X⁻¹`, `-X⁻²` or `-X⁻³`) — the Jacobian is never materialized.
"""
function psd_barrier_grad!(o::PSDBarrierOracle, x, g::AbstractVector, type::Symbol)
    type === :logdet && return psd_barrier_grad!(o, x, g)
    o.mode === :dense || throw(
        ArgumentError(
            "sd barrier type :$(type) requires the dense evaluation mode (mode :$(o.mode))",
        ),
    )
    Xinv = _dense_inv(o, x)
    dφ = if type === :inverse
        -vec(Xinv * Xinv)
    elseif type === :inverse_square
        -vec(Xinv * Xinv * Xinv)
    else
        throw(ArgumentError("unknown sd barrier type :$(type)"))
    end
    if o.c isa LinearFunction
        mul!(g, o.c.A', dφ)
    else
        _, gr = NonconvexCore.value_gradient(z -> dot(dφ, vec(o.c(z))), x)
        gr === nothing ? fill!(g, 0.0) : copyto!(g, gr)
    end
    return g
end

"""
    psd_hess_dense!(o::PSDBarrierOracle, x, H, type::Symbol)

Assembled Hessian of the type-`type` PSD barrier (dense mode). The
non-log-determinant types use the analytic Hessian of `tr(X⁻ᑫ)/q`,

    d∇φ(X)·Y = Σ_{k=0}^{q} X^{-(k+1)} Y X^{-(q+1-k)}

(`q = 1` for `:inverse`, `q = 2` for `:inverse_square`), so no second-order
AD through the factorization is needed.
"""
function psd_hess_dense!(o::PSDBarrierOracle, x, H::AbstractMatrix, type::Symbol)
    type === :logdet && return psd_hess_dense!(o, x, H)
    o.mode === :dense || throw(
        ArgumentError(
            "sd barrier type :$(type) requires the dense evaluation mode (mode :$(o.mode))",
        ),
    )
    n = length(x)
    Xinv = _dense_inv(o, x)
    q = _sd_power(type)
    Ps = [_deriv_block(o, x, i) for i in 1:n]
    Xm = Vector{Matrix{Float64}}(undef, q + 2)
    Xm[1] = Matrix{Float64}(I, o.dim, o.dim)
    for m in 1:(q + 1)
        Xm[m + 1] = Xm[m] * Xinv
    end
    for i in 1:n, j in 1:n
        s = 0.0
        for k in 0:q
            s += tr(Xm[k + 2] * Ps[i] * Xm[q - k + 2] * Ps[j])
        end
        H[i, j] = s
    end
    return H
end

"""
    psd_hess_prod!(o::PSDBarrierOracle, x, w, y, type::Symbol)

Hessian-vector product of the type-`type` PSD barrier.
"""
function psd_hess_prod!(
    o::PSDBarrierOracle, x, w::AbstractVector, y::AbstractVector, type::Symbol
)
    type === :logdet && return psd_hess_prod!(o, x, w, y)
    o.mode === :dense || throw(
        ArgumentError(
            "sd barrier type :$(type) requires the dense evaluation mode (mode :$(o.mode))",
        ),
    )
    n = length(x)
    Xinv = _dense_inv(o, x)
    q = _sd_power(type)
    Ps = [_deriv_block(o, x, i) for i in 1:n]
    Xm = Vector{Matrix{Float64}}(undef, q + 2)
    Xm[1] = Matrix{Float64}(I, o.dim, o.dim)
    for m in 1:(q + 1)
        Xm[m + 1] = Xm[m] * Xinv
    end
    Pw = sum(w[i] * Ps[i] for i in 1:n) # Σⱼ ∂Xⱼ wⱼ
    for i in 1:n
        s = 0.0
        for k in 0:q
            s += tr(Xm[k + 2] * Ps[i] * Xm[q - k + 2] * Pw)
        end
        y[i] = s
    end
    return y
end

"""
    psd_hess_dense!(o::PSDBarrierOracle, x, H)

Assembled Hessian `∇²ₓₓ(-log det X(x))` written into `H`.
"""
function psd_hess_dense!(o::PSDBarrierOracle, x, H::AbstractMatrix)
    nvars = length(x)
    X = _oracle_mat(o, x)
    F = cholesky(X)
    Xinv = F \ Matrix{Float64}(I, o.dim, o.dim)
    o.Xinv = Xinv
    Ps = Matrix{Float64}[]
    for i in 1:nvars
        M = _deriv_block(o, x, i)
        push!(Ps, Xinv * M)
    end
    for i in 1:nvars, j in 1:nvars
        H[i, j] = sum(Ps[i] .* Ps[j]')
    end
    return H
end

"""
    psd_hess_prod!(o::PSDBarrierOracle, x, w, y)

Hessian-vector product `∇²ₓₓ(-log det X(x)) w` written into `y`.
"""
function psd_hess_prod!(o::PSDBarrierOracle, x, w::AbstractVector, y::AbstractVector)
    nvars = length(x)
    if o.mode === :dense
        X = _oracle_mat(o, x)
        F = cholesky(X)
        Xinv = F \ Matrix{Float64}(I, o.dim, o.dim)
        Mw = _deriv_dir(o, x, w)
        Pw = Xinv * Mw
        for i in 1:nvars
            M = _deriv_block(o, x, i)
            y[i] = sum(Pw .* (Xinv * M)')
        end
    elseif o.mode === :matrixfree
        T = o.T
        Q = o.Q
        if T === nothing || Q === nothing
            T, Q = _lanczos(_oracle_mat(o, x); tol=1e-10, maxiter=max(2 * o.dim, 20))
            o.T = T
            o.Q = Q
        end
        Tinv = inv(Matrix(T))
        Bw = Q' * _deriv_dir(o, x, w) * Q
        C = Tinv * Bw * Tinv
        if o.Bs === nothing
            Bs = Matrix{Float64}[]
            for i in 1:nvars
                M = _deriv_block(o, x, i)
                push!(Bs, Q' * M * Q)
            end
            o.Bs = Bs
        end
        for i in 1:nvars
            y[i] = sum(C .* o.Bs[i]')
        end
    else # :trace — biased-Hutchinson Gauss-Newton approximation
        X = _oracle_mat(o, x)
        z = zeros(o.dim)
        u = similar(z)
        fill!(y, 0.0)
        for _ in 1:o.nprobes
            _rademacher!(z, o.rng)
            u, hist = IterativeSolvers.cg(Matrix(X), z; reltol=o.cg_tol, maxiter=10 * o.dim)
            Mwz = _deriv_dir(o, x, w) * z
            s = dot(u, Mwz)
            for i in 1:nvars
                M = _deriv_block(o, x, i)
                y[i] += s * dot(u, M * z) / o.nprobes
            end
        end
    end
    return y
end

# ---------------------------------------------------------------------------
# Linear algebra building blocks (self-contained: no dependence on
# NonconvexInteriorPoint)
# ---------------------------------------------------------------------------

# Lanczos tridiagonalization of a symmetric matrix with full
# reorthogonalization. Returns (T, Q) with Q' A Q = T.
function _lanczos(A::Symmetric; tol::Float64=1e-10, maxiter::Int=200)
    n = size(A, 1)
    q = randn(n)
    q ./= norm(q)
    qprev = zeros(n)
    β = 0.0
    αs = Float64[]
    βs = Float64[]
    Q = zeros(n, max(1, min(maxiter, n)))
    v = Vector{Float64}(undef, n)
    k = 0
    while k < min(maxiter, n)
        mul!(v, A, q)
        α = dot(q, v)
        v .-= α .* q
        v .-= β .* qprev
        # full reorthogonalization
        for j in 1:k
            v .-= dot(v, view(Q, :, j)) .* view(Q, :, j)
        end
        β = norm(v)
        push!(αs, α)
        k += 1
        Q[:, k] .= q
        if β < tol || k == min(maxiter, n)
            break
        end
        push!(βs, β)
        qprev .= q
        q .= v ./ β
    end
    T = SymTridiagonal(αs, length(βs) == length(αs) - 1 ? βs : βs[1:(k - 1)])
    return T, Q[:, 1:k]
end
