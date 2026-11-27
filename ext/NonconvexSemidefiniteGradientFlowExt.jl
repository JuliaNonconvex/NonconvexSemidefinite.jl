# ------------------------------------------------------------------
# Semidefinite support for NonconvexGradientFlow's NullSpaceAlg: the conic
# null-space projection.
#
# For a semidefinite constraint P(x) ⪰ 0, the cone of feasible directions
# at x is { d : DP(x)[d] ∈ T_{S₊ⁿ}(P(x)) }, where the tangent cone of the
# positive semidefinite cone at A is
#     T_{S₊ⁿ}(A) = { H = Hᵀ : qᵀ H q ≥ 0 for all q ∈ ker A }.
# With Q0 the eigenvectors of A with (near-)zero eigenvalues, H ∈ T(A)
# iff Q0ᵀ H Q0 ⪰ 0. The polar cone of the PSD tangent cone is
# { Q0 S Q0ᵀ : S ⪯ 0 }.
#
# Following the paper's dual QP (3.9), the conic analogue
#     min ‖∇f + Aᵀλ + Cᵀμ + Σ_j J_jᵀ vec(Q0_j S_j Q0_jᵀ)‖²
#     s.t. μ ≥ 0, S_j ⪯ 0
# projects −∇f onto the negative polar cone; its residual IS the
# null-space step ξ_J (the projection of ∇f onto the null space of the
# tangency set), and the conic multipliers Λ_j = Q0_j S_j Q0_jᵀ ⪯ 0
# identify the PSD blocks the trajectory stays tangent to. The QP is
# solved by projected gradient (λ free, μ projects onto the nonnegative
# orthant, S_j onto the negative semidefinite cone via `project_psd`).
# Violated (indefinite) blocks enter the QP like the active ones: the
# tangent cone at Π_{S₊ⁿ}(P) is spanned by the eigenvectors of P with
# eigenvalue ≤ rank_tol, and the dual decides whether the flow stays
# tangent to the block or lets it re-enter the feasible domain.
# ------------------------------------------------------------------
module NonconvexSemidefiniteGradientFlowExt

using NonconvexSemidefinite: NonconvexSemidefinite
using NonconvexGradientFlow
using NonconvexCore: VecModel
import NonconvexGradientFlow:
    _sd_blocks,
    _sd_violation,
    _sd_range,
    _sd_merit_terms,
    _sd_step_cap,
    _null_direction,
    _null_space_projection,
    _stack,
    _jac,
    _gauss_newton
using NonconvexSemidefinite: project_psd, project_psd_neg, psd_infeasibility
using LinearAlgebra

const SDPBlock = @NamedTuple{
    J::Matrix{Float64},
    P::Matrix{Float64},
    Q0::Matrix{Float64},
    dim::Int,
    idx::Int,
    λmin::Float64,
    active::Bool,
    violated::Bool,
}

# Per-block data at x: the violation matrix P(x), its Jacobian J (d² × n)
# and the zero-eigenspace basis Q0 of the tangent cone.
function _sd_blocks(m::VecModel, x, fcalls, options)
    blocks = SDPBlock[]
    for (idx, c) in enumerate(m.sd_constraints.fs)
        Pv, J = _jac(z -> c(z), x, fcalls)
        d = c.dim
        P = reshape(Pv, d, d)
        P = (P + P') / 2
        F = eigen(Symmetric(P))
        λs = F.values
        λmin = minimum(λs)
        Q0 = Matrix(F.vectors[:, λs .<= options.rank_tol])
        violated = λmin < -options.rank_tol
        active = !violated && λmin <= options.tangency_tol
        push!(
            blocks,
            (
                J=Matrix{Float64}(J),
                P=P,
                Q0=Q0,
                dim=d,
                idx=idx,
                λmin=λmin,
                active=active,
                violated=violated,
            ),
        )
    end
    return blocks
end

function _sd_violation(m::VecModel, x, sd::Vector)
    isempty(sd) && return 0.0
    return maximum(psd_infeasibility(b.P) for b in sd)
end

# The range-space step for the violated and boundary-near blocks
# (the paper's Ĩ_ε stabilization: the signed residual drives the block to
# the boundary from both sides). Violated blocks: Gauss–Newton on the full
# negative spectral part; active blocks (smallest eigenvalue within the
# tangency tolerance): Gauss–Newton on λmin · u_min u_minᵀ, pushing λmin
# to zero.
function _sd_range(m::VecModel, x, sd::Vector)
    n = length(m.box_min)
    isempty(sd) && return zeros(n)
    ξC = zeros(n)
    for b in sd
        if b.violated
            r = vec(project_psd_neg(b.P))
        elseif b.active
            umin = b.Q0[:, 1]
            r = b.λmin .* vec(umin * umin')
        else
            continue
        end
        iszero(r) && continue
        ξC .+= _gauss_newton(b.J, r)
    end
    return ξC
end

# The conic dual QP (the paper's (3.9) on the negative polar cone) only
# IDENTIFIES the tangency set — its multipliers are computed by projected
# gradient to modest accuracy. The null-space step ξ_J itself is then the
# exact (machine-precision) projection of ∇f onto the null space of the
# tangency set: the scalar working set W plus one row per symmetric basis
# of the zero eigenspace of every block with a nonzero conic multiplier
# (the paper's Definition 3.24, formula (4.4)).
function _null_direction(∇f, A, C, sd::Vector, options, warm)
    p = collect(float.(∇f))
    A = Matrix{Float64}(A)
    C = Matrix{Float64}(C)
    _, λ, μ, Λ, W = _conic_projection(p, A, C, sd, options, warm)
    M = _stack(A, C[W, :])
    M = _stack(M, _sd_tangency_rows(sd, Λ, options.multiplier_tol))
    ξJ = _null_space_projection(p, M, zeros(0, length(p)))
    return ξJ, λ, μ, Λ, W
end

# The rows enforcing Q0_jᵀ (DP_j d) Q0_j = 0 for every block j with a
# nonzero conic multiplier (‖Λ_j‖ > tol): one row per symmetric basis
# matrix S_l of the zero eigenspace, row = vec(Q0 S_l Q0ᵀ)ᵀ J_j.
function _sd_tangency_rows(sd, Λ, tol)
    rows = Matrix{Float64}(undef, 0, 0)
    for (j, b) in enumerate(sd)
        norm(Λ[j]) > tol || continue
        k = size(b.Q0, 2)
        k == 0 && continue
        n = size(b.J, 2)
        isempty(rows) && (rows = Matrix{Float64}(undef, 0, n))
        for l in 1:(k * (k + 1) ÷ 2)
            S_l = zeros(k, k)
            _fill_sym_basis!(S_l, l)
            row = vec(b.Q0 * S_l * b.Q0')' * b.J
            rows = vcat(rows, row)
        end
    end
    return rows
end

# Fill S (k×k) with the l-th upper-triangular unit basis of the symmetric
# matrices (diagonal first, then off-diagonal pairs).
function _fill_sym_basis!(S, l)
    k = size(S, 1)
    d = 0
    for i in 1:k
        d += 1
        if d == l
            S[i, i] = 1.0
            return S
        end
    end
    for i in 1:k, j in (i + 1):k
        d += 1
        if d == l
            S[i, j] = S[j, i] = 0.5
            return S
        end
    end
    return S
end

# min ½‖p + Aᵀλ + Cᵀμ + Σ_j J_jᵀ vec(Q0_j S_j Q0_jᵀ)‖² over λ free,
# μ ≥ 0 and S_j ⪯ 0, by projected gradient. Returns
# (ξJ, λ, μ, Λ, W): the residual, the multipliers, the conic multipliers
# Λ_j = Q0_j S_j Q0_jᵀ (negative semidefinite) and the scalar working set.
function _conic_projection(p, A, C, sd, options, warm)
    m_eq = size(A, 1)
    m_in = size(C, 1)
    ks = [size(b.Q0, 2) for b in sd]
    λ = zeros(m_eq)
    μ = zeros(m_in)
    S = [zeros(k, k) for k in ks]
    if warm !== nothing
        wλ = getfield(warm, :λ)
        wμ = getfield(warm, :μ)
        if length(wλ) == m_eq && length(wμ) == m_in
            λ .= wλ
            μ .= wμ
        end
        wΛ = getfield(warm, :Λ)
        if wΛ !== nothing && !isempty(sd) && all(b -> b.idx <= length(wΛ), sd)
            for (j, b) in enumerate(sd)
                ks[j] == 0 && continue
                S[j] .= b.Q0' * wΛ[b.idx] * b.Q0
            end
        end
    end
    # The residual map: d = p + Aᵀλ + Cᵀμ + Σ J_jᵀ vec(Q0_j S_j Q0_jᵀ).
    function assemble(λ, μ, S)
        d = p .+ A' * λ .+ C' * μ
        for (j, b) in enumerate(sd)
            ks[j] == 0 && continue
            d .+= b.J' * vec(b.Q0 * S[j] * b.Q0')
        end
        return d
    end
    d = assemble(λ, μ, S)
    # ‖B_j‖ ≤ ‖J_j‖ for the tangent-cone lifting, so the Lipschitz constant
    # of the dual quadratic is bounded by (‖A‖ + ‖C‖ + Σ ‖J_j‖)².
    L = max((opnorm(A) + opnorm(C) + sum(b -> opnorm(b.J), sd))^2, 1e-12)
    for _ in 1:options.dual_solver_iters
        gλ = A * d
        gμ = C * d
        gS = [
            if ks[j] == 0
                zeros(0, 0)
            else
                _sympart(b.Q0' * reshape(b.J * d, b.dim, b.dim) * b.Q0)
            end for (j, b) in enumerate(sd)
        ]
        q = sum(abs2, d) / 2
        t = 1 / L
        λt, μt, St, dt = λ, μ, S, d
        while true
            λt = λ .- t .* gλ
            μt = max.(μ .- t .* gμ, 0.0)
            # projection onto the negative semidefinite cone
            St = [
                ks[j] == 0 ? zeros(0, 0) : .-project_psd(.-(S[j] .- t .* gS[j])) for
                j in eachindex(sd)
            ]
            dt = assemble(λt, μt, St)
            qt = sum(abs2, dt) / 2
            Δλ = λt .- λ
            Δμ = μt .- μ
            bound =
                q +
                dot(gλ, Δλ) +
                dot(gμ, Δμ) +
                (
                    sum(abs2, Δλ) +
                    sum(abs2, Δμ) +
                    sum(sj -> sum(abs2, St[sj] .- S[sj]), eachindex(sd))
                ) / (2t) +
                1e-16
            (qt <= bound || t < 1e-16) && break
            t /= 2
        end
        λ, μ, S, d = λt, μt, St, dt
        res = _conic_kkt_residual(d, A, C, μ, sd, S)
        (res <= 1e-8 || !isfinite(res)) && break
    end
    Λ = [ks[j] == 0 ? zeros(b.dim, b.dim) : b.Q0 * S[j] * b.Q0' for (j, b) in enumerate(sd)]
    W = findall(>(1e-10), μ)
    return d, λ, μ, Λ, W
end

_sympart(H) = (H + H') / 2

# The step-size cap for the violated semidefinite blocks: the per-step
# change of a block is limited to half of its violation scale (with a
# floor at a tenth of the infeasibility tolerance), so that near the PSD
# boundary the Gauss–Newton pull dominates the eigenvector-rotation
# regeneration of the violation (which otherwise produces a discrete
# 2-cycle around the boundary).
function _sd_step_cap(sd::Vector, d::AbstractVector, tol)
    cap = Inf
    for b in sd
        b.violated || continue
        v = norm(project_psd_neg(b.P))
        rate = norm(b.J * d)
        rate == 0.0 && continue
        cap = min(cap, 0.5 * max(v, 0.1 * tol.infeas) / rate)
    end
    return cap
end

# KKT residual of the conic dual QP: stationarity of the QP is
# (A d = 0, C d ≤ 0 ⊥ μ ≥ 0, Q0ᵀ (J_j d) Q0 ⪯ 0 ⊥ S_j ⪯ 0) with
# complementarity ⟨S_j, Q0ᵀ (J_j d) Q0⟩ = 0.
function _conic_kkt_residual(d, A, C, μ, sd, S)
    r1 = size(A, 1) == 0 ? 0.0 : maximum(abs, A * d)
    r2 = size(C, 1) == 0 ? 0.0 : maximum(max.(C * d, 0.0); init=0.0)
    r3 = isempty(μ) ? 0.0 : maximum(-μ; init=0.0)
    r4 = isempty(μ) ? 0.0 : maximum(μ .* max.(-(C * d), 0.0); init=0.0)
    r5 = 0.0
    r6 = 0.0
    for (j, b) in enumerate(sd)
        k = size(b.Q0, 2)
        k == 0 && continue
        H = _sympart(b.Q0' * reshape(b.J * d, b.dim, b.dim) * b.Q0)
        r5 = max(r5, maximum(eigvals(Symmetric(H)); init=0.0))
        r6 = max(r6, abs(tr(S[j] * H)))
    end
    return max(r1, r2, r3, r4, r5, r6)
end

# The semidefinite terms of the merit function (Lemma 4.1 extended to the
# PSD cone): the linear part Σ⟨Λ_j, P_j(x)⟩ (Λ padded with zeros for the
# blocks outside the tangency set) and the quadratic part
# Σ‖P_j(x) - Π_{S₊ⁿ}(P_j(x))‖²_F (nonzero exactly for the violated
# blocks, whose gradient is the PSD range-space step).
function _sd_merit_terms(m::VecModel, x, Λ::Vector)
    lin = 0.0
    quad = 0.0
    for (j, c) in enumerate(m.sd_constraints.fs)
        d = c.dim
        P = reshape(c(x), d, d)
        P = (P + P') / 2
        lin += dot(Λ[j], P)
        quad += sum(abs2, project_psd_neg(P))
    end
    return lin, quad
end

# No tangency multipliers (e.g. no active or violated blocks): the
# quadratic violation term remains, the linear term vanishes. More
# specific than the base `(m, x, ::Nothing)` method through the `VecModel`
# annotation.
function _sd_merit_terms(m::VecModel, x, ::Nothing)
    quad = 0.0
    for c in m.sd_constraints.fs
        d = c.dim
        P = reshape(c(x), d, d)
        P = (P + P') / 2
        quad += sum(abs2, project_psd_neg(P))
    end
    return 0.0, quad
end

end
