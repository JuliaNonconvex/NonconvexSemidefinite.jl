# Projections onto the positive/negative semidefinite cones.

"""
    project_psd(A::AbstractMatrix)

Orthogonal projection of the symmetric part of `A` onto the cone of positive
semidefinite matrices. If `A = U Λ Uᵀ` is a symmetric eigendecomposition, the
projection is `U max(Λ, 0) Uᵀ`.
"""
function project_psd(A::AbstractMatrix)
    S = (A + A') / 2
    F = eigen(Symmetric(S))
    U = F.vectors
    λ = max.(F.values, zero(eltype(F.values)))
    return U * Diagonal(λ) * U'
end

"""
    project_psd_neg(A::AbstractMatrix)

Orthogonal projection of the symmetric part of `A` onto the cone of negative
semidefinite matrices, computed via the Moreau decomposition
`Π_{S₋ⁿ}(A) = A - Π_{S₊ⁿ}(A)`.
"""
function project_psd_neg(A::AbstractMatrix)
    S = (A + A') / 2
    return S - project_psd(S)
end

"""
    psd_infeasibility(P::AbstractMatrix)

Distance of the symmetric part of `P` from the cone of positive semidefinite
matrices, i.e. `‖Π_{S₋ⁿ}(P)‖_F` (the Frobenius norm of the negative
eigenvalues). Returns `0` if and only if `P` is positive semidefinite.
"""
psd_infeasibility(P::AbstractMatrix) = norm(project_psd_neg(P))

# Analytic pullback of `project_psd` (spectral function differential, Lewis).
# For Z = U diag(θ) Uᵀ with θ = max(λ, 0), the pullback of ΔZ to the symmetric
# part of A is U (G ∘ (Uᵀ ΔZ_sym U)) Uᵀ with
#   G[i,i] = λ[i] > 0 ? 1 : 0
#   G[i,j] = (θ[i] - θ[j]) / (λ[i] - λ[j])     if λ[i] != λ[j]
#   G[i,j] = λ[i] > 0 ? 1 : 0                  if λ[i] == λ[j] (subgradient choice)
function ChainRulesCore.rrule(::typeof(project_psd), A::AbstractMatrix)
    S = (A + A') / 2
    F = eigen(Symmetric(S))
    λ = F.values
    U = F.vectors
    θ = max.(λ, zero(eltype(λ)))
    Z = U * Diagonal(θ) * U'
    function project_psd_pullback(ΔZ)
        if ΔZ isa ChainRulesCore.AbstractZero
            return NoTangent(), ZeroTangent()
        end
        ΔZ_sym = (ΔZ + ΔZ') / 2
        M = U' * ΔZ_sym * U
        G = similar(M)
        for j in axes(G, 2), i in axes(G, 1)
            if i == j || λ[i] == λ[j]
                G[i, j] = λ[i] > 0 ? one(eltype(G)) : zero(eltype(G))
            else
                G[i, j] = (θ[i] - θ[j]) / (λ[i] - λ[j])
            end
        end
        ΔS = U * (G .* M) * U'
        return NoTangent(), ΔS
    end
    return Z, project_psd_pullback
end
