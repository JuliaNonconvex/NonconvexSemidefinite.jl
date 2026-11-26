# Matrix-free log-determinant estimators.
#
# All estimators operate through the matrix-vector product `f!(y, x)`
# computing `y = A * x` for a symmetric positive definite `A`; some also
# need solves with shifted systems (through Krylov methods). They power the
# `:trace` evaluation mode of the semidefinite barrier oracles in this
# package (consumed by NonconvexInteriorPoint.jl through its extension).

using Random: Random
import LinearAlgebra:
    Symmetric, SymTridiagonal, eigen, eigvals, logdet, dot, norm, normalize!
using ChainRulesCore: ChainRulesCore
import ChainRulesCore: NoTangent, ProjectTo, rrule
using IterativeSolvers: IterativeSolvers

function _rademacher!(z::AbstractVector{Float64}, rng::Random.AbstractRNG)
    @inbounds for i in eachindex(z)
        z[i] = rand(rng, Bool) ? 1.0 : -1.0
    end
    return z
end

"""
    slq_logdet(f!, n; lanczos_deg = 30, nprobes = 10, rng)

Estimate `log det A` (for symmetric positive definite `A`, `f!(y, x)`
computes `y = A * x`) with stochastic Lanczos quadrature (Golub–Meurant,
Ubaru–Chen–Saad).
"""
function slq_logdet(
    f!,
    n::Int;
    lanczos_deg::Int=30,
    nprobes::Int=10,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    est = 0.0
    for _ in 1:nprobes
        z = randn(rng, n)
        normalize!(z)
        est += _lanczos_quad_logdet(f!, z, lanczos_deg)
    end
    return (n / nprobes) * est
end

function _lanczos_quad_logdet(f!, z::AbstractVector, m::Int)
    n = length(z)
    q = copy(z)
    normalize!(q)
    qprev = zeros(n)
    β = 0.0
    αs = Float64[]
    βs = Float64[]
    v = Vector{Float64}(undef, n)
    out = 0.0
    for j in 1:m
        f!(v, q)
        α = dot(q, v)
        v .-= α .* q
        v .-= β .* qprev
        β = norm(v)
        push!(αs, α)
        if β < 1e-13 || j == m
            break
        end
        push!(βs, β)
        qprev .= q
        q .= v ./ β
    end
    T = SymTridiagonal(αs, βs)
    E = eigen(T)
    for i in eachindex(E.values)
        θ = E.values[i]
        θ > 0 || continue
        w = E.vectors[1, i]^2
        out += w * log(θ)
    end
    return out
end

"""
    lanczos_bounds(f!, n; k = 10, rng)

Lower/upper bounds for the spectrum of the symmetric matrix given by the
matrix-vector product `f!`, from a `k`-step Lanczos run: `(θmin - βk,
θmax + βk)` with a positive floor for the lower bound.
"""
function lanczos_bounds(f!, n::Int; k::Int=10, rng::Random.AbstractRNG=Random.default_rng())
    z = randn(rng, n)
    normalize!(z)
    q = copy(z)
    qprev = zeros(n)
    β = 0.0
    αs = Float64[]
    βs = Float64[]
    v = Vector{Float64}(undef, n)
    for j in 1:min(k, n)
        f!(v, q)
        α = dot(q, v)
        v .-= α .* q
        v .-= β .* qprev
        β = norm(v)
        push!(αs, α)
        β < 1e-13 && break
        push!(βs, β)
        qprev .= q
        q .= v ./ β
    end
    T = SymTridiagonal(αs, βs)
    θ = eigvals(T)
    lmax = θ[end] + (isempty(βs) ? 0.0 : βs[end])
    lmin = θ[1] - (isempty(βs) ? 0.0 : βs[end])
    return max(lmin, 1e-10 * max(lmax, 1.0)), lmax
end

"""
    chebyshev_logdet(f!, n; deg = 60, nprobes = 10, lmin = nothing, lmax = nothing, rng)

Estimate `log det A` through a Chebyshev polynomial expansion of the
logarithm on `[lmin, lmax]` with Hutchinson trace estimation of the
Chebyshev moments `tr(T_j(A))` (matrix-vector products only; the spectral
bounds are estimated with a short Lanczos run when not supplied).
"""
function chebyshev_logdet(
    f!,
    n::Int;
    deg::Int=60,
    nprobes::Int=10,
    lmin=nothing,
    lmax=nothing,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    a, b = if lmin === nothing || lmax === nothing
        lanczos_bounds(f!, n; k=10, rng=rng)
    else
        (lmin, lmax)
    end
    αm = (b - a) / 2
    βm = (a + b) / 2
    # Chebyshev coefficients of g(t) = log(αm t + βm) on [-1, 1] via
    # Gauss-Chebyshev quadrature with deg+1 nodes
    m = deg + 1
    cs = zeros(deg + 1)
    for k in 1:m
        θ = (k - 0.5) * π / m
        t = cos(θ)
        g = log(αm * t + βm)
        for j in 0:deg
            cs[j + 1] += g * cos(j * θ)
        end
    end
    cs .*= 2 / m
    cs[1] /= 2
    # matvec with the shifted/scaled operator Â = (A - βm I)/αm
    aop! = (y, x) -> begin
        f!(y, x)
        y .-= βm .* x
        y ./= αm
    end
    # tr(T_j(Â)) via Hutchinson with the three-term recurrence
    trs = zeros(deg + 1)
    z = zeros(n)
    q0 = similar(z)
    q1 = similar(z)
    q2 = similar(z)
    for _ in 1:nprobes
        _rademacher!(z, rng)
        copyto!(q0, z)
        aop!(q1, z)
        trs[1] += dot(z, q0)
        deg >= 1 && (trs[2] += dot(z, q1))
        for j in 2:deg
            aop!(q2, q1)
            q2 .*= 2
            q2 .-= q0
            trs[j + 1] += dot(z, q2)
            q0, q1, q2 = q1, q2, q0
        end
    end
    trs ./= nprobes
    return dot(cs, trs)
end

# Stochastic Lanczos quadrature restricted to the range of a deflated
# operator R = (I - QQ')A(I - QQ'): the probes are projected onto
# range(I - QQ'), so the null space (zero eigenvalues of R) does not corrupt
# the Gauss quadrature, and the estimate scales with the range dimension.
function _slq_logdet_range(
    f!,
    n::Int,
    Q::AbstractMatrix;
    lanczos_deg::Int=30,
    nprobes::Int=10,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    k = size(Q, 2)
    est = 0.0
    for _ in 1:nprobes
        z = randn(rng, n)
        z .-= Q * (Q' * z)
        norm(z) > 1e-12 || continue
        normalize!(z)
        est += _lanczos_quad_logdet(f!, z, lanczos_deg)
    end
    return ((n - k) / nprobes) * est
end

"""
    hutchpp_logdet(f!, n; k = 20, lanczos_deg = 30, nprobes = 10, rng)

Estimate `log det A` with a Hutch++-style estimator for the matrix
logarithm: a Nyström part `log det(Q'AQ)` plus stochastic Lanczos
quadrature on the deflated operator `(I - QQ')A(I - QQ')` (restricted to
its range). No factorization is required anywhere: the small Nyström matrix
`C = Q'AQ` is handled through its matrix-vector products (Lanczos for the
log-determinant, conjugate gradients for the adjoint solves).
"""
function hutchpp_logdet(
    f!,
    n::Int;
    k::Int=20,
    lanczos_deg::Int=30,
    nprobes::Int=10,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    k = min(k, n)
    Ω = randn(rng, n, k)
    Q = Matrix(qr(Ω).Q)
    AQ = Matrix{Float64}(undef, n, k)
    for j in 1:k
        f!(view(AQ, :, j), view(Q, :, j))
    end
    C = Q' * AQ
    nystrom = _small_logdet(C)
    # deflated operator R = (I - QQ')A(I - QQ')
    rop! = (y, x) -> begin
        w = x .- Q * (Q' * x)
        f!(y, w)
        y .-= Q * (Q' * y)
    end
    resid = _slq_logdet_range(rop!, n, Q; lanczos_deg=lanczos_deg, nprobes=nprobes, rng=rng)
    return nystrom + resid
end

# log det of a small SPD matrix via its matrix-vector products (full Lanczos
# — deterministic, no factorization).
function _small_logdet(C::AbstractMatrix)
    k = size(C, 1)
    f! = (y, v) -> mul!(y, C, v)
    z = ones(k)
    Q, αs, βs = _lanczos_full(f!, z, k)
    E = eigen(SymTridiagonal(αs, βs))
    return sum(log(θ) for θ in E.values if θ > 0)
end

# C⁻¹ applied to the columns of X through conjugate gradients (no
# factorization).
function _small_solve(C::AbstractMatrix, X::AbstractMatrix; tol::Float64=1e-10)
    k = size(C, 1)
    f! = (y, v) -> mul!(y, C, v)
    cols = [_shifted_cg(f!, view(X, :, j), 0.0; tol=tol, maxiter=4k) for j in 1:size(X, 2)]
    return reduce(hcat, cols)
end

# Gauss-Laguerre nodes and weights on [0, ∞) (weight e^{-t}). The
# log-identity integrands carry no e^{-t} weight, so the consumers reweight
# the quadrature weights by e^{t_k}.
function _gauss_laguerre(m::Int)
    T = SymTridiagonal([2j + 1 for j in 0:(m - 1)], [j for j in 1:(m - 1)])
    E = eigen(T)
    w = [E.vectors[1, i]^2 for i in 1:m] # ∫e^{-t} dt = 1
    return E.values, w
end

# Gauss-Legendre nodes and weights on [0, 1].
function _gauss_legendre_unit(m::Int)
    T = SymTridiagonal(fill(0.5, m), [j / (2 * sqrt(4j^2 - 1)) for j in 1:(m - 1)])
    E = eigen(T)
    w = [E.vectors[1, i]^2 for i in 1:m] # ∫₀¹ dt = 1
    return E.values, w
end

# Conjugate-gradient solve of (A + σ I) x = b (A SPD, σ ≥ 0), with `f!` the
# matvec of A. Returns the solution. (The positive shifts make a
# multishift-CG formulation applicable directly.)
function _shifted_cg(f!, b::AbstractVector, σ::Real; tol::Float64=1e-8, maxiter::Int=200)
    n = length(b)
    aop! = (y, x) -> begin
        f!(y, x)
        y .+= σ .* x
    end
    x = zeros(n)
    r = copy(b)
    p = copy(r)
    γ = dot(r, r)
    v = Vector{Float64}(undef, n)
    for _ in 1:maxiter
        aop!(v, p)
        α = γ / dot(p, v)
        x .+= α .* p
        r .-= α .* v
        γnew = dot(r, r)
        sqrt(γnew) <= tol * sqrt(dot(b, b)) && break
        p .= r .+ (γnew / γ) .* p
        γ = γnew
    end
    return x
end

"""
    rational_logdet(f!, n; nodes = 12, nprobes = 10, cg_tol = 1e-8, rng)

Estimate `log det A` through a rational approximation of the logarithm
(a Gauss-Laguerre partial-fraction expansion of `log x = ∫ (1/(1+t) -
1/(x+t)) dt`), with the resolvent traces estimated by Hutchinson and the
shifted systems solved by conjugate gradients.

Accuracy note: the expansion's `e^t` reweighting makes this a coarse
estimator on wide spectra (a few percent of relative bias; the error does
not shrink much with `nodes`). For higher accuracy prefer `:slq`,
`:chebyshev`, `:taylor` or `:integration`.
"""
function rational_logdet(
    f!,
    n::Int;
    nodes::Int=12,
    nprobes::Int=10,
    cg_tol::Float64=1e-8,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    t, w = _gauss_laguerre(nodes)
    w = w .* exp.(t) # the log-identity integrand carries no e^{-t} weight
    c0 = sum(w .* (1 ./ (1 .+ t)))
    z = zeros(n)
    est = 0.0
    for _ in 1:nprobes
        _rademacher!(z, rng)
        s = 0.0
        for (tk, wk) in zip(t, w)
            u = _shifted_cg(f!, z, tk; tol=cg_tol)
            s += wk * dot(z, u)
        end
        est += c0 * n - s
    end
    return est / nprobes
end

"""
    taylor_logdet(f!, n; terms = 60, lmin = nothing, lmax = nothing, nprobes = 10, rng)

Estimate `log det A` through the Taylor series `log A = log(c)I + Σ (-1)^{j+1}
(A/c - I)^j / j` centered at the spectrum midpoint `c = (λmin + λmax)/2`
(which minimizes the contraction ratio), with Hutchinson estimation of the
moment traces (matrix-vector products only).
"""
function taylor_logdet(
    f!,
    n::Int;
    terms::Int=60,
    nprobes::Int=10,
    lmin=nothing,
    lmax=nothing,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    a, b = if lmin === nothing || lmax === nothing
        lanczos_bounds(f!, n; k=10, rng=rng)
    else
        (lmin, lmax)
    end
    c = (a + b) / 2
    # E = A/c - I
    eop! = (y, x) -> begin
        f!(y, x)
        y ./= c
        y .-= x
    end
    z = zeros(n)
    v = similar(z)
    wbuf = similar(z)
    trs = zeros(terms)
    for _ in 1:nprobes
        _rademacher!(z, rng)
        copyto!(v, z)
        for j in 1:terms
            eop!(wbuf, v)
            trs[j] += dot(z, wbuf)
            v, wbuf = wbuf, v
        end
    end
    trs ./= nprobes
    est = n * log(c)
    for j in 1:terms
        est += (isodd(j) ? 1.0 : -1.0) * trs[j] / j
    end
    return est
end

"""
    trace_integration_logdet(f!, n; nodes = 12, nprobes = 10, cg_tol = 1e-8, rng)

Estimate `log det A` by stochastic trace integration of the identity
`log x = ∫₀¹ (x-1)/(1 + t(x-1)) dt` (Gauss-Legendre quadrature in `t`),
with the resolvent traces estimated by Hutchinson and the shifted systems
`solve (tA + (1-t)I) u = z` by conjugate gradients.
"""
function trace_integration_logdet(
    f!,
    n::Int;
    nodes::Int=12,
    nprobes::Int=10,
    cg_tol::Float64=1e-8,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    t, w = _gauss_legendre_unit(nodes)
    z = zeros(n)
    est = 0.0
    for _ in 1:nprobes
        _rademacher!(z, rng)
        s = 0.0
        for (tk, wk) in zip(t, w)
            # (I + tk(A - I)) u = z  ⟺  (tk A + (1 - tk) I) u = z
            u = _shifted_cg(f!, z, (1 - tk) / tk; tol=cg_tol)
            Au = similar(z)
            f!(Au, u)
            s += (wk / tk) * dot(z, Au .- u)
        end
        est += s
    end
    return est / nprobes
end

"""
    lowrank_logdet(f!, n; r = 20, rng)

Estimate `log det A` for near-identity matrices `A = I + B` from a partial
Lanczos spectrum of `B`: `Σᵢ log(1 + θᵢ)` over the `r` dominant Ritz values
(matrix-vector products only; deterministic given the starting vector).
"""
function lowrank_logdet(f!, n::Int; r::Int=20, rng::Random.AbstractRNG=Random.default_rng())
    # B = A - I
    bop! = (y, x) -> begin
        f!(y, x)
        y .-= x
    end
    z = randn(rng, n)
    normalize!(z)
    q = copy(z)
    qprev = zeros(n)
    β = 0.0
    αs = Float64[]
    βs = Float64[]
    v = Vector{Float64}(undef, n)
    Q = zeros(n, min(r, n))
    for j in 1:min(r, n)
        bop!(v, q)
        α = dot(q, v)
        v .-= α .* q
        v .-= β .* qprev
        for l in 1:(j - 1) # full reorthogonalization
            v .-= dot(v, view(Q, :, l)) .* view(Q, :, l)
        end
        β = norm(v)
        push!(αs, α)
        Q[:, j] .= q
        β < 1e-13 && break
        push!(βs, β)
        qprev .= q
        q .= v ./ β
    end
    T = SymTridiagonal(αs, βs)
    θ = eigvals(T)
    return sum(log(1 + θi) for θi in θ if θi > -1)
end

"""
    logdet_estimate(f!, n; method = :slq, kwargs...)

Estimate `log det A` (A symmetric positive definite, given through the
matrix-vector product `f!(y, x)` computing `y = A x`) with one of the
matrix-free methods:

- `:slq` — stochastic Lanczos quadrature (default),
- `:chebyshev` — Chebyshev polynomial approximation (Hutchinson moments),
- `:hutchpp` — Hutch++/Nyström with the matrix logarithm,
- `:rational` — rational approximation with shifted CG solves,
- `:taylor` — Taylor series at the spectrum midpoint,
- `:integration` — stochastic trace integration with shifted CG solves,
- `:lowrank` — low-rank eigenvalue approximation (near-identity matrices).

See the individual methods for their keyword arguments.
"""
function logdet_estimate(f!, n::Int; method::Symbol=:slq, kwargs...)
    method === :slq && return slq_logdet(f!, n; kwargs...)
    method === :chebyshev && return chebyshev_logdet(f!, n; kwargs...)
    method === :hutchpp && return hutchpp_logdet(f!, n; kwargs...)
    method === :rational && return rational_logdet(f!, n; kwargs...)
    method === :taylor && return taylor_logdet(f!, n; kwargs...)
    method === :integration && return trace_integration_logdet(f!, n; kwargs...)
    method === :lowrank && return lowrank_logdet(f!, n; kwargs...)
    return throw(ArgumentError("unknown log-determinant estimation method :$(method)"))
end

# ---------------------------------------------------------------------------
# Estimator on a parameterized matrix, with an rrule whose pullback requires
# only linear solves (no factorization, no materialized inverse).
#
# Each estimator records its probes/Lanczos vectors at primal time; the
# pullback applies the resulting adjoint functional `X ↦ ⟨ΔA, X⟩` to the
# constraint derivatives. For the rational/integration methods the adjoint
# is built from the stored shifted-system solutions (linear solves); for the
# Lanczos/Chebyshev/Taylor/low-rank methods it is a low-rank combination of
# the stored Krylov vectors (matvecs only).
# ---------------------------------------------------------------------------

# Matrix of the constraint value.
_mat_of(c, x) = c(x) isa AbstractMatrix ? Matrix(c(x)) : c(x)
_mat_of(c::LinearFunction, x) = reshape(c(x), _mat_dim(c, x), _mat_dim(c, x))
_mat_dim(c, x) = begin
    v = c(x)
    v isa AbstractMatrix ? size(v, 1) : round(Int, sqrt(length(v)))
end
_mat_dim(c::LinearFunction, x) = round(Int, sqrt(size(c.A, 1)))

# Full-ish Lanczos with reorthogonalization: returns the basis Q and the
# tridiagonal entries (αs, βs).
function _lanczos_full(f!, z::AbstractVector, m::Int)
    n = length(z)
    q = copy(z)
    normalize!(q)
    qprev = zeros(n)
    β = 0.0
    αs = Float64[]
    βs = Float64[]
    v = Vector{Float64}(undef, n)
    Q = Matrix{Float64}(undef, n, 0)
    for j in 1:m
        f!(v, q)
        α = dot(q, v)
        v .-= α .* q
        v .-= β .* qprev
        for l in 1:(j - 1) # full reorthogonalization
            v .-= dot(v, view(Q, :, l)) .* view(Q, :, l)
        end
        β = norm(v)
        push!(αs, α)
        Q = hcat(Q, q)
        β < 1e-13 && break
        push!(βs, β)
        qprev .= q
        q .= v ./ β
    end
    return Q, αs, βs
end

# Per-method: estimate + adjoint functional (X ↦ ⟨ΔA, X⟩).
function _adjoint_data(f!, n::Int; method::Symbol=:slq, solve=nothing, kwargs...)
    method === :slq && return _slq_adjoint(f!, n; kwargs...)
    method === :hutchpp && return _hutchpp_adjoint(f!, n; kwargs...)
    method === :rational && return _rational_adjoint(f!, n; solve=solve, kwargs...)
    method === :integration && return _integration_adjoint(f!, n; solve=solve, kwargs...)
    method === :lowrank && return _lowrank_adjoint(f!, n; kwargs...)
    method in (:chebyshev, :taylor) && throw(
        ArgumentError(
            "the :$(method) reverse rule is not implemented yet; use :slq, " *
            ":hutchpp, :rational, :integration or :lowrank for differentiable " *
            "estimates (the :$(method) values remain available)",
        ),
    )
    return throw(ArgumentError("unknown log-determinant estimation method :$(method)"))
end

# SLQ adjoint restricted to the range of a deflated operator (see
# `_slq_logdet_range`): probes projected onto range(I - QQ') and the
# (n - k) scaling for both the value and the adjoint functional. The
# adjoint uses the full divided-difference expansion (see `_slq_adjoint`).
function _slq_adjoint_range(
    f!,
    n::Int,
    Q::AbstractMatrix;
    lanczos_deg::Int=30,
    nprobes::Int=10,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    k = size(Q, 2)
    data = Vector{Any}()
    est = 0.0
    for _ in 1:nprobes
        z = randn(rng, n)
        z .-= Q * (Q' * z)
        norm(z) > 1e-12 || continue
        zn = norm(z)
        Qb, αs, βs = _lanczos_full(f!, z, lanczos_deg)
        E = eigen(SymTridiagonal(αs, βs))
        w = [E.vectors[1, i]^2 for i in eachindex(E.values)]
        for i in eachindex(E.values)
            E.values[i] > 0 && (est += w[i] * log(E.values[i]))
        end
        push!(data, (Qb, E.vectors, E.values, zn))
    end
    est *= (n - k) / nprobes
    apply = function (X)
        s = 0.0
        for (Qb, U, θ, zn) in data
            qs = Qb * U
            zq = [U[1, i] for i in eachindex(θ)]
            for i in eachindex(θ)
                θ[i] > 0 || continue
                for j in eachindex(θ)
                    θ[j] > 0 || continue
                    ld = i == j ? 1 / θ[i] : (log(θ[i]) - log(θ[j])) / (θ[i] - θ[j])
                    s += ld * zq[i] * zq[j] * dot(view(qs, :, i), X * view(qs, :, j))
                end
            end
        end
        return s * (n - k) / nprobes
    end
    return est, apply
end

# Stochastic Lanczos quadrature: the adjoint of each probe term z'log(A)z is
# the divided-difference expansion
#
#   Σ_{i,j} log^{[1]}(θᵢ, θⱼ) (z'q̂ᵢ)(z'q̂ⱼ) q̂ᵢ q̂ⱼ',
#
# with log^{[1]}(θ, θ) = 1/θ and log^{[1]}(θᵢ, θⱼ) = (log θᵢ - log θⱼ)/(θᵢ - θⱼ)
# (exact when the Lanczos degree covers the spectrum; the standard Gauss
# quadrature adjoint otherwise). Non-mutating.
function _slq_adjoint(
    f!,
    n::Int;
    lanczos_deg::Int=30,
    nprobes::Int=10,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    data = Vector{Any}()
    est = 0.0
    for _ in 1:nprobes
        z = randn(rng, n)
        zn = norm(z)
        Q, αs, βs = _lanczos_full(f!, z, lanczos_deg)
        E = eigen(SymTridiagonal(αs, βs))
        w = [E.vectors[1, i]^2 for i in eachindex(E.values)]
        for i in eachindex(E.values)
            E.values[i] > 0 && (est += w[i] * log(E.values[i]))
        end
        push!(data, (Q, E.vectors, E.values, zn))
    end
    est *= n / nprobes
    apply = function (X)
        s = 0.0
        for (Q, U, θ, zn) in data
            qs = Q * U # Ritz vectors q̂ᵢ
            zq = [U[1, i] for i in eachindex(θ)] # z̃'q̂ᵢ = u₁ᵢ (unit probe)
            for i in eachindex(θ)
                θ[i] > 0 || continue
                for j in eachindex(θ)
                    θ[j] > 0 || continue
                    ld = i == j ? 1 / θ[i] : (log(θ[i]) - log(θ[j])) / (θ[i] - θ[j])
                    s += ld * zq[i] * zq[j] * dot(view(qs, :, i), X * view(qs, :, j))
                end
            end
        end
        return s * n / nprobes
    end
    return est, apply
end

# Chebyshev: adjoint through the three-term recurrence. The spectral bounds
# are treated as constants (stopped gradients) in the pullback.
function _chebyshev_adjoint(
    f!,
    n::Int;
    deg::Int=60,
    nprobes::Int=10,
    lmin=nothing,
    lmax=nothing,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    a, b = if lmin === nothing || lmax === nothing
        lanczos_bounds(f!, n; k=10, rng=rng)
    else
        (lmin, lmax)
    end
    αm = (b - a) / 2
    βm = (a + b) / 2
    m = deg + 1
    cs = zeros(deg + 1)
    for k in 1:m
        θ = (k - 0.5) * π / m
        t = cos(θ)
        g = log(αm * t + βm)
        for j in 0:deg
            cs[j + 1] += g * cos(j * θ)
        end
    end
    cs .*= 2 / m
    cs[1] /= 2
    aop! = (y, x) -> begin
        f!(y, x)
        y .-= βm .* x
        y ./= αm
    end
    # store the Chebyshev vectors q_j = T_j(Â) z per probe
    probes = Vector{Tuple{Vector{Float64},Vector{Vector{Float64}}}}()
    trs = zeros(deg + 1)
    z = zeros(n)
    for _ in 1:nprobes
        _rademacher!(z, rng)
        qs = Vector{Vector{Float64}}()
        q0 = copy(z)
        push!(qs, copy(q0))
        q1 = similar(q0)
        aop!(q1, q0)
        push!(qs, copy(q1))
        trs[1] += dot(z, q0)
        deg >= 1 && (trs[2] += dot(z, q1))
        qprev = q0
        qcur = q1
        for j in 2:deg
            qnew = similar(qcur)
            aop!(qnew, qcur)
            qnew .*= 2
            qnew .-= qprev
            trs[j + 1] += dot(z, qnew)
            push!(qs, copy(qnew))
            qprev, qcur = qcur, qnew
        end
        push!(probes, (copy(z), qs))
    end
    trs ./= nprobes
    est = dot(cs, trs)
    # adjoint of the Chebyshev moments via the backward recurrence of
    # q_{j+1} = 2Â q_j - q_{j-1} (with q₀ = z): the cotangent of q_j is
    #   ā_j = c_j z + 2Â ā_{j+1} - ā_{j+2},
    # and ΔÂ = Σ_j 2 ā_{j+1} q_j', with dÂ = dA/αm. The backward pass only
    # needs the fixed operator (it does not depend on the differentiated
    # matrix), so it is precomputed outside the apply closure — the apply
    # itself is a pure sum of dot products (Zygote-safe).
    adj_data = Vector{Tuple{Vector{Vector{Float64}},Vector{Vector{Float64}}}}()
    for (zr, qs) in probes
        anext = zeros(n) # ā_{j+2}
        acur = zeros(n) # ā_{j+1}
        as = Vector{Vector{Float64}}(undef, deg + 1)
        for j in deg:-1:0
            aprev = cs[j + 1] .* zr .+ 2 .* _opapply(aop!, acur) .- anext
            as[j + 1] = copy(acur) # ā_{j+1}
            anext = acur
            acur = aprev
        end
        push!(adj_data, (as, qs))
    end
    apply = function (X)
        s = 0.0
        for (as, qs) in adj_data
            for j in 0:(deg - 1)
                s += 2 * dot(as[j + 2], X * qs[j + 1])
            end
        end
        return s / (nprobes * αm)
    end
    return est, apply
end

# Non-mutating operator application.
_opapply(op!, v) = (y=similar(v); op!(y, v); y)

# Hutch++/Nyström with the matrix logarithm: adjoint = Nyström part (C⁻¹
# applied through CG solves) + SLQ adjoint on the deflated operator. No
# factorization required.
function _hutchpp_adjoint(
    f!,
    n::Int;
    k::Int=20,
    lanczos_deg::Int=30,
    nprobes::Int=10,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    k = min(k, n)
    Ω = randn(rng, n, k)
    Q = Matrix(qr(Ω).Q)
    AQ = Matrix{Float64}(undef, n, k)
    for j in 1:k
        f!(view(AQ, :, j), view(Q, :, j))
    end
    C = Q' * AQ
    nystrom = _small_logdet(C)
    rop! = (y, x) -> begin
        w = x .- Q * (Q' * x)
        f!(y, w)
        y .-= Q * (Q' * y)
    end
    resid, rapply = _slq_adjoint_range(
        rop!, n, Q; lanczos_deg=lanczos_deg, nprobes=nprobes, rng=rng
    )
    est = nystrom + resid
    # C⁻¹ through CG solves, precomputed outside the differentiated apply
    # (the adjoint of log det C = ⟨C⁻¹, Q'XQ⟩)
    W = _small_solve(C, Matrix{Float64}(I, k, k))
    apply = function (X)
        # adjoint of the deflated operator R = (I-QQ')A(I-QQ')
        R_adj = X .- Q * (Q' * X) .- X * (Q * Q') .+ Q * (Q' * X * Q) * Q'
        return dot(W, Q' * X * Q) + rapply(R_adj)
    end
    return est, apply
end

# Rational (Gauss-Laguerre partial fractions): the adjoint is built from the
# stored shifted-system solutions u = (A + t I)⁻¹ z.
function _rational_adjoint(
    f!,
    n::Int;
    nodes::Int=12,
    nprobes::Int=10,
    cg_tol::Float64=1e-8,
    solve=nothing,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    solv = solve === nothing ? (fb, b, σ) -> _shifted_cg(fb, b, σ; tol=cg_tol) : solve
    t, w = _gauss_laguerre(nodes)
    w = w .* exp.(t) # the log-identity integrand carries no e^{-t} weight
    c0 = sum(w .* (1 ./ (1 .+ t)))
    data = Vector{Tuple{Vector{Float64},Vector{Vector{Float64}}}}()
    z = zeros(n)
    est = 0.0
    for _ in 1:nprobes
        _rademacher!(z, rng)
        us = [solv(f!, z, tk) for tk in t]
        s = sum(wk * dot(z, u) for (wk, u) in zip(w, us))
        est += c0 * n - s
        push!(data, (copy(z), us))
    end
    est /= nprobes
    apply = function (X)
        s = 0.0
        for (zr, us) in data
            for (wk, u) in zip(w, us)
                s += wk * dot(u, X * u)
            end
        end
        return s / nprobes
    end
    return est, apply
end

# Taylor series at the spectrum midpoint: adjoint from the stored moment
# vectors v_j = E^j z (the spectral bounds are stopped gradients).
function _taylor_adjoint(
    f!,
    n::Int;
    terms::Int=60,
    nprobes::Int=10,
    lmin=nothing,
    lmax=nothing,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    a, b = if lmin === nothing || lmax === nothing
        lanczos_bounds(f!, n; k=10, rng=rng)
    else
        (lmin, lmax)
    end
    c = (a + b) / 2
    eop! = (y, x) -> begin
        f!(y, x)
        y ./= c
        y .-= x
    end
    z = zeros(n)
    data = Vector{Tuple{Vector{Float64},Vector{Vector{Float64}}}}()
    trs = zeros(terms)
    for _ in 1:nprobes
        _rademacher!(z, rng)
        vs = Vector{Vector{Float64}}()
        v = copy(z)
        wbuf = similar(v)
        for j in 1:terms
            eop!(wbuf, v)
            trs[j] += dot(z, wbuf)
            push!(vs, copy(wbuf))
            v, wbuf = wbuf, v
        end
        push!(data, (copy(z), vs))
    end
    trs ./= nprobes
    est = n * log(c)
    for j in 1:terms
        est += (isodd(j) ? 1.0 : -1.0) * trs[j] / j
    end
    apply = function (X)
        s = 0.0
        for (zr, vs) in data
            for j in 1:terms
                coeff = (isodd(j) ? 1.0 : -1.0) / j
                for l in 0:(j - 1)
                    s += coeff * dot(vs[l + 1], X * vs[j - l])
                end
            end
        end
        return s / (nprobes * c)
    end
    return est, apply
end

# Stochastic trace integration: adjoint from the stored shifted-system
# solutions u = (tA + (1-t)I)⁻¹ z.
function _integration_adjoint(
    f!,
    n::Int;
    nodes::Int=12,
    nprobes::Int=10,
    cg_tol::Float64=1e-8,
    solve=nothing,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    solv = solve === nothing ? (fb, b, σ) -> _shifted_cg(fb, b, σ; tol=cg_tol) : solve
    t, w = _gauss_legendre_unit(nodes)
    data = Vector{
        Tuple{
            Vector{Float64},Vector{Tuple{Float64,Float64,Vector{Float64},Vector{Float64}}}
        },
    }()
    z = zeros(n)
    est = 0.0
    for _ in 1:nprobes
        _rademacher!(z, rng)
        rows = Vector{Tuple{Float64,Float64,Vector{Float64},Vector{Float64}}}()
        s = 0.0
        for (tk, wk) in zip(t, w)
            u = solv(f!, z, (1 - tk) / tk)
            Au = similar(z)
            f!(Au, u)
            s += (wk / tk) * dot(z, Au .- u)
            Az = similar(z)
            f!(Az, z)
            u2 = solv(f!, (Az .- z) ./ tk, (1 - tk) / tk) # S⁻¹(A-I)z
            push!(rows, (tk, wk, u, u2))
        end
        est += s
        push!(data, (copy(z), rows))
    end
    est /= nprobes
    apply = function (X)
        s = 0.0
        for (zr, rows) in data
            for (tk, wk, u, u2) in rows
                # adjoint of z'(A-I)S⁻¹z with S = tk A + (1-tk) I:
                # (wk/tk)[⟨z, X u⟩ - tk ⟨S⁻¹(A-I)z, X u⟩]
                s += (wk / tk) * (dot(zr, X * u) - tk * dot(u2, X * u))
            end
        end
        return s / nprobes
    end
    return est, apply
end

# Low-rank eigenvalue approximation: adjoint from the Ritz vectors.
function _lowrank_adjoint(
    f!, n::Int; r::Int=20, nprobes::Int=10, rng::Random.AbstractRNG=Random.default_rng()
)
    bop! = (y, x) -> begin
        f!(y, x)
        y .-= x
    end
    z = randn(rng, n)
    Q, αs, βs = _lanczos_full(bop!, z, r)
    E = eigen(SymTridiagonal(αs, βs))
    θ = E.values
    est = sum(log(1 + θi) for θi in θ if θi > -1)
    # adjoint of Σ log(1 + θᵢ): dλᵢ = q̂ᵢ'dB q̂ᵢ (first-order perturbation;
    # the eigenvector rotation cancels in the eigenvalue sum)
    U = E.vectors
    qs = Q * U
    apply = function (X)
        s = 0.0
        for i in eachindex(θ)
            θi = θ[i]
            θi > -1 || continue
            q̂ = view(qs, :, i)
            s += dot(q̂, X * q̂) / (1 + θi)
        end
        return s
    end
    return est, apply
end

"""
    logdet_estimate_of(c, x; method = :slq, solve = nothing, kwargs...)

Estimate `log det X(x)` for the symmetric positive definite `X(x) = c(x)`
(`c` callable, or a `LinearFunction` with `dim²` rows), with one of the
matrix-free methods of [`logdet_estimate`](@ref).

The reverse-mode rule (`rrule`) of the estimate computes its gradient with
respect to `x` using only **linear solves** — no factorization and no
materialized inverse: the pullback reuses the probes/Lanczos vectors and the
shifted-system solutions recorded by the estimator (the `:rational` and
`:integration` methods need solves with the shifted matrices, supplied by
`solve(f!, b, σ)` or a conjugate-gradient default; the other methods only
need matrix-vector products). The spectral bounds used by the
`:chebyshev`, `:taylor` and `:hutchpp` methods are treated as constants
(stopped gradients).
"""
function logdet_estimate_of(c, x; method::Symbol=:slq, solve=nothing, kwargs...)
    n = _mat_dim(c, x)
    f! = (y, v) -> mul!(y, _mat_of(c, x), v)
    return logdet_estimate(f!, n; method=method, kwargs...)
end

function ChainRulesCore.rrule(
    ::typeof(logdet_estimate_of), c, x; method::Symbol=:slq, solve=nothing, kwargs...
)
    n = _mat_dim(c, x)
    X = _mat_of(c, x)
    f! = (y, v) -> mul!(y, X, v)
    est, apply = _adjoint_data(f!, n; method=method, solve=solve, kwargs...)
    function logdet_estimate_of_pullback(ȳ)
        g = _grad_of(c, x, apply, ȳ)
        return (NoTangent(), NoTangent(), g)
    end
    return est, logdet_estimate_of_pullback
end

# Gradient of the estimator w.r.t. x: the adjoint functional applied to the
# constraint derivatives (per-block for LinearFunction, one pullback
# otherwise).
function _grad_of(c::LinearFunction, x, apply, ȳ)
    d = _mat_dim(c, x)
    g = similar(x)
    @inbounds for i in eachindex(x)
        g[i] = ȳ * apply(reshape(view(c.A, :, i), d, d))
    end
    return g
end
function _grad_of(c, x, apply, ȳ)
    _, gr = NonconvexCore.value_gradient(z -> apply(_mat_of(c, z)), x)
    return gr === nothing ? zeros(eltype(x), length(x)) : ȳ .* gr
end
