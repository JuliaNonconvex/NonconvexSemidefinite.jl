# Chordal decomposition utilities.
#
# By Grone-Johnson-Sá-Wolkowicz (1984), a symmetric matrix with a sparsity
# pattern whose graph has a chordal extension is positive semidefinite if and
# only if all its principal submatrices indexed by the maximal cliques of the
# extension are positive semidefinite. This allows replacing one large `n×n`
# PSD constraint with many small PSD constraints on the cliques.

using SparseArrays: SparseMatrixCSC, sparse, nzrange, rowvals, nonzeros
import Zygote

"""
    symmetric_pattern(pattern::SparseMatrixCSC)

Symmetrized structural sparsity pattern of `pattern`, augmented with the
diagonal.
"""
function symmetric_pattern(pattern::SparseMatrixCSC)
    n = LinearAlgebra.checksquare(pattern)
    I = Int[]
    J = Int[]
    for j in 1:n
        for k in nzrange(pattern, j)
            i = rowvals(pattern)[k]
            iszero(nonzeros(pattern)[k]) && continue
            push!(I, i)
            push!(J, j)
            push!(I, j)
            push!(J, i)
        end
        push!(I, j)
        push!(J, j)
    end
    return sparse(I, J, trues(length(I)), n, n)
end

"""
    chordal_extension(pattern::SparseMatrixCSC)

Chordal extension of the graph of the sparsity `pattern`, computed with a
sparse Cholesky factorization (AMD ordering). Returns the maximal cliques of
the extended graph as a vector of index vectors in the original vertex
ordering. Each clique is fully connected in `pattern`, so
`symmetric_pattern(pattern)` restricted to any clique is dense.
"""
function chordal_extension(pattern::SparseMatrixCSC)
    n = LinearAlgebra.checksquare(pattern)
    P = symmetric_pattern(pattern)
    # Cholesky of a diagonally dominant matrix with the same pattern: the fill
    # pattern of L is the chordal completion, while diagonal dominance makes
    # the factorization numerically valid.
    A = convert(SparseMatrixCSC{Float64,Int}, P)
    for i in 1:n
        A[i, i] += n
    end
    F = cholesky(A, check = false)
    L = sparse(F.L)
    cliques = Vector{Vector{Int}}()
    for j in 1:n
        C = Int[j]
        for k in nzrange(L, j)
            i = rowvals(L)[k]
            i > j && push!(C, i)
        end
        push!(cliques, C)
    end
    maxcliques = _maximal_cliques(cliques)
    # Map the clique vertices back to the original ordering. The cliques of
    # the chordal graph are given in the ordering used by the factorization.
    unpermuted = map(maxcliques) do C
        sort!([F.p[i] for i in C])
    end
    return unpermuted
end

function _maximal_cliques(cliques::Vector{Vector{Int}})
    out = Vector{Vector{Int}}()
    for C in cliques
        maximal = true
        for C2 in cliques
            if length(C2) > length(C) && all(in(C2), C)
                maximal = false
                break
            end
        end
        maximal && !(C in out) && push!(out, C)
    end
    return out
end

"""
    aggregate_sparsity(f, x0::AbstractVector, dim::Int)

Structural sparsity pattern of the matrix-valued function `f` (returning a
`dim×dim` matrix) at `x0`, computed with reverse-mode AD: entry `(i, j)` is
structurally nonzero if `f(x)[i, j]` depends on any variable.
"""
function aggregate_sparsity(f, x0::AbstractVector, dim::Int)
    out, pb = Zygote.pullback(f, x0)
    length(out) == dim^2 || throw(
        ArgumentError(
            "Expected a $(dim)×$(dim) matrix-valued function but got an output of length $(length(out)).",
        ),
    )
    pattern = falses(dim, dim)
    for j in 1:dim, i in 1:dim
        ΔM = zeros(eltype(out), dim, dim)
        ΔM[i, j] = one(eltype(ΔM))
        Δx = pb(ΔM)
        Δx = Δx isa Tuple ? Δx[1] : Δx
        # `nothing`/AbstractZero cotangents mean the output entry does not
        # depend on the variables (e.g. constant terms).
        if Δx !== nothing &&
           !(Δx isa ChainRulesCore.AbstractZero) &&
           any(!iszero, Δx)
            pattern[i, j] = true
        end
    end
    return symmetric_pattern(sparse(pattern))
end

"""
    chordal_decompose(model::VecModel, patterns; x0 = nothing)
    chordal_decompose(model::VecModel; x0)

Replace each semidefinite constraint `P(x) ⪰ 0` of `model` with the clique
constraints `P(x)[C, C] ⪰ 0` for every maximal clique `C` of a chordal
extension of the constraint's sparsity pattern. The two models are equivalent
(Grone-Johnson-Sá-Wolkowicz).

`patterns` is a vector with one sparsity pattern per semidefinite constraint
(or a single pattern when the model has one constraint). Entries may be
`nothing` to probe the sparsity of the constraint at `x0` (the union of the
structural AD dependence pattern and the numerical support of `P(x0)`).

If `x0` is provided, every nonzero entry of `P(x0)` must lie inside the
pattern, otherwise an error is thrown: sparsity exploitation is only valid
when the pattern covers all entries the constraint can have.
"""
function chordal_decompose(model::VecModel, patterns; x0 = nothing)
    n_sd = length(model.sd_constraints.fs)
    pats = patterns isa AbstractVector ? patterns : [patterns]
    length(pats) == n_sd || throw(
        ArgumentError(
            "Got $(length(pats)) sparsity patterns for $n_sd semidefinite constraints; provide one per constraint.",
        ),
    )
    sd_fs = SDConstraint[]
    for (k, sd) in enumerate(model.sd_constraints.fs)
        f, dim = sd.f, sd.dim
        pat = pats[k]
        if pat === nothing
            x0 === nothing && throw(
                ArgumentError(
                    "To probe the sparsity pattern of semidefinite constraint $k, an initial solution `x0` is required.",
                ),
            )
            pat = aggregate_sparsity(f, x0, dim)
            P0 = f(x0)
            pat = symmetric_pattern(sparse(Matrix(pat) .| (P0 .!= 0)))
        else
            LinearAlgebra.checksquare(pat) == dim || throw(
                ArgumentError(
                    "The sparsity pattern of semidefinite constraint $k has size $(size(pat, 1)) but the constraint dimension is $dim.",
                ),
            )
            if x0 !== nothing
                P0 = f(x0)
                spat = symmetric_pattern(pat)
                for i in CartesianIndices(P0)
                    if !iszero(P0[i]) && !spat[i]
                        throw(
                            ArgumentError(
                                "The entry $(Tuple(i)) of semidefinite constraint $k is nonzero at `x0` " *
                                "but lies outside the supplied sparsity pattern; the decomposition " *
                                "would not be equivalent to the original constraint.",
                            ),
                        )
                    end
                end
            end
        end
        cliques = chordal_extension(pat)
        for C in cliques
            push!(sd_fs, SDConstraint(x -> f(x)[C, C], length(C)))
        end
    end
    return VecModel(
        model.objective,
        model.eq_constraints,
        model.ineq_constraints,
        VectorOfFunctions(sd_fs),
        model.box_min,
        model.box_max,
        model.init,
        model.integer,
    )
end
function chordal_decompose(model::VecModel; x0)
    return chordal_decompose(model, [nothing for _ in model.sd_constraints.fs]; x0 = x0)
end
