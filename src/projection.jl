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

"""
    partition_blocks(P::AbstractMatrix, blocks)

The block partition of a matrix: `blocks` is a collection of index
vectors (one per subdomain/element), and the result is the collection of
principal submatrices `P[I, I]` for `I ∈ blocks`. Defined with an
analytic [`rrule`](@ref): the cotangents of the blocks scatter-add back
into a cotangent for `P` — a *sparse* cotangent when `P` is sparse, so
the reverse cost is the number of gathered entries, and overlapping
blocks accumulate.

This is the partition primitive behind the SDD/element relaxations of a
matrix-valued function: composing it with an rrule'd matrix function
propagates the block derivatives back to the sparse matrix in one rule,
without traced per-block loops.
"""
function partition_blocks(P::AbstractMatrix, blocks)
    return [P[I, I] for I in blocks]
end
function ChainRulesCore.rrule(::typeof(partition_blocks), P::AbstractMatrix, blocks)
    out = partition_blocks(P, blocks)
    function partition_blocks_pb(Δout)
        Δout = ChainRulesCore.unthunk(Δout)
        if Δout isa ChainRulesCore.AbstractZero
            return NoTangent(), ZeroTangent(), NoTangent()
        end
        # NEVER densify: for a sparse P the block cotangents scatter back
        # into a SPARSE ΔP (the gathered nnz's only); no n×n dense object
        # is created anywhere in this rule.
        ΔP = if P isa SparseMatrixCSC
            rows = Int[]
            cols = Int[]
            vals = eltype(P)[]
            for (ΔB, I) in zip(Δout, blocks)
                ΔB isa ChainRulesCore.AbstractZero && continue
                for (a, i) in enumerate(I), (b, j) in enumerate(I)
                    Δ = ΔB[a, b]
                    iszero(Δ) && continue
                    push!(rows, i)
                    push!(cols, j)
                    push!(vals, Δ)
                end
            end
            sparse(rows, cols, vals, size(P, 1), size(P, 2))
        else
            ΔP = zeros(eltype(P), size(P))
            for (ΔB, I) in zip(Δout, blocks)
                ΔB isa ChainRulesCore.AbstractZero && continue
                for (a, i) in enumerate(I), (b, j) in enumerate(I)
                    ΔP[i, j] += ΔB[a, b]
                end
            end
            ΔP
        end
        return NoTangent(), ΔP, NoTangent()
    end
    return out, partition_blocks_pb
end

"""
    project_psd_batch(A::AbstractArray{T,3})

Batched orthogonal projection onto the PSD cone: projects every slice
`A[:, :, k]` with [`project_psd`](@ref). One primitive with one analytic
[`rrule`](@ref) so reverse-mode AD sees a single node: the pullback
propagates the cotangents to *all* slices at once (each slice uses the
spectral differential of its own decomposition).
"""
function project_psd_batch(A::AbstractArray{T,3}) where {T}
    Z = similar(A)
    for k in axes(A, 3)
        Z[:, :, k] .= project_psd(@view A[:, :, k])
    end
    return Z
end
function ChainRulesCore.rrule(::typeof(project_psd_batch), A::AbstractArray{T,3}) where {T}
    Z = similar(A)
    pbs = Vector{Any}(undef, size(A, 3))
    for k in axes(A, 3)
        Z[:, :, k], pbs[k] = ChainRulesCore.rrule(project_psd, @view A[:, :, k])
    end
    function project_psd_batch_pb(ΔZ)
        ΔZ = ChainRulesCore.unthunk(ΔZ)
        if ΔZ isa ChainRulesCore.AbstractZero
            return NoTangent(), ZeroTangent()
        end
        ΔA = similar(A)
        for k in axes(A, 3)
            ΔA[:, :, k] .= pbs[k](ΔZ[:, :, k])[2]
        end
        return NoTangent(), ΔA
    end
    return Z, project_psd_batch_pb
end

"""
    psd_neg_eig(W::AbstractMatrix)

The negative-eigenvalue part of the symmetric part of `W` as low-rank
factors `(U₋, λ₋)`: `Π_{S₋ⁿ}(W) = U₋·Diagonal(max(λ₋, 0))·U₋ᵀ`. No dense
projection matrix is formed — only the negative eigenpairs are kept, which
is all the PSD violation metric and its derivatives need (the metric is
`‖λ₋‖₂` and its cotangent is the low-rank matrix `2·Π₋(W)`).
"""
function psd_neg_eig(W::AbstractMatrix)
    F = eigen(Symmetric((W + W') / 2))
    neg = findall(<(0), F.values)
    λ₋ = F.values[neg]
    return Matrix(F.vectors[:, neg]), λ₋
end

"""
    psd_infeasibility_batch(W::AbstractArray{T,3})

Batched PSD violation metric: `‖Π_{S₋ⁿ}(W[:, :, k])‖_F` for every slice,
computed from the low-rank negative eigenpairs of each slice (see
[`psd_neg_eig`](@ref)) — no full projection matrices are materialized.
"""
function psd_infeasibility_batch(W::AbstractArray{T,3}) where {T}
    g = zeros(T, size(W, 3))
    for k in axes(W, 3)
        _, λ₋ = psd_neg_eig(@view W[:, :, k])
        g[k] = norm(λ₋)
    end
    return g
end
