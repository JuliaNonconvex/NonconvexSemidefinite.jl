# Chordal decomposition: replacing one large semidefinite constraint with
# many small ones.
#
# The exact form (Fukuda's "conversion"): a symmetric matrix P with a
# chordal sparsity pattern is positive semidefinite if and only if
#
#     P = Σₖ E(Cₖ)ᵀ Yₖ E(Cₖ)          with Yₖ ⪰ 0,
#
# where the Cₖ are the maximal cliques of a chordal extension and E(C) is
# the 0/1 embedding of the clique into the n×n matrix. `chordal_transform`
# rewrites the PSD components of a model this way: one small PSD variable
# block per clique, coupled to P by linear equalities on the entries of the
# extension pattern (so the blocks can reuse the sparsity of P — element
# stiffness matrices, FE connectivity, …).
#
# The relaxation form: P ⪰ 0 implies every maximal-clique principal
# submatrix P[C,C] ⪰ 0, but the converse fails in general — the tridiagonal
# matrix tridiag(1, 1, 1) has PSD 2×2 clique submatrices yet is indefinite.
# `chordal_decompose` replaces a PSD constraint with these clique-submatrix
# constraints: that is a RELAXATION (the relaxed feasible set contains the
# original one; for minimization it yields a lower bound). It is kept as a
# labeled relaxation, e.g. for element-assembled matrices whose element
# blocks are rank-one (truss stiffness matrices), where the element-wise
# form is exact.

using SparseArrays: SparseMatrixCSC, sparse, nzrange, rowvals, nonzeros
using AMD: AMD
using LinearAlgebra
using Zygote: Zygote

using NonconvexCore: NonconvexCore
using NonconvexCore: VecModel, FunctionInSet, VariableInSet, VectorOfFunctions, Objective

"""
    symmetric_pattern(pattern::SparseMatrixCSC)

Symmetrized structural sparsity pattern of `pattern`, augmented with the
diagonal: entry `(i, j)` is true whenever `pattern[i, j]` or `pattern[j, i]`
is structurally nonzero, or `i == j`.
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
    AbstractChordalAlgorithm

The algorithms for computing the chordal extension of a sparsity pattern.
The built-in, dependency-light ones are [`AMDAlg`](@ref) (the default:
SuiteSparse's approximate minimum degree ordering) and [`MCSAlg`](@ref)
(maximum cardinality search); [`MFAlg`](@ref), [`METISAlg`](@ref) and
[`BTAlg`](@ref) use the optional CliqueTrees.jl dependency.
"""
abstract type AbstractChordalAlgorithm end

"""
    AMDAlg()

The default chordal-extension algorithm: SuiteSparse's approximate minimum
degree ordering (AMD.jl) followed by the symbolic Cholesky elimination.
"""
struct AMDAlg <: AbstractChordalAlgorithm end

"""
    MCSAlg()

Maximum cardinality search ordering (Tarjan–Yannakakis) followed by the
symbolic Cholesky elimination. Built in: a chordal graph's extension is
itself (no fill), since the reverse of an MCS numbering is a perfect
elimination ordering.
"""
struct MCSAlg <: AbstractChordalAlgorithm end

"""
    MFAlg()

Minimum fill heuristic (CliqueTrees.jl): usually smaller fill than AMD at a
higher cost. Requires the optional CliqueTrees.jl dependency.
"""
struct MFAlg <: AbstractChordalAlgorithm end

"""
    METISAlg()

Nested dissection via the METIS library (CliqueTrees.jl): often the smallest
cliques on large 2D/3D mesh patterns. Requires the optional CliqueTrees.jl
dependency.
"""
struct METISAlg <: AbstractChordalAlgorithm end

"""
    BTAlg()

An exact (optimal treewidth) algorithm (CliqueTrees.jl): the smallest
maximal cliques, at exponential worst-case cost. Requires the optional
CliqueTrees.jl dependency.
"""
struct BTAlg <: AbstractChordalAlgorithm end

"""
    chordal_extension(pattern::SparseMatrixCSC; alg = AMDAlg())

The maximal cliques of a chordal extension of the graph of the sparsity
`pattern`, as a vector of index vectors in the original vertex ordering.
Every edge of `pattern` is covered by at least one clique, and the clique
graph is chordal. On a chordal pattern the extension is the pattern itself
(no fill). The `alg` keyword selects the ordering (see
[`AbstractChordalAlgorithm`](@ref)).
"""
function chordal_extension(pattern::SparseMatrixCSC; alg::AbstractChordalAlgorithm=AMDAlg())
    return _chordal_extension(pattern, alg)
end

# The dispatch hook: the optional CliqueTrees extension specializes this
# with methods for the CliqueTrees-only algorithms (which then win whenever
# it is loaded); the built-ins fall back to `_chordal_extension_default`.
function _chordal_extension(pattern::SparseMatrixCSC, alg::AbstractChordalAlgorithm)
    return _chordal_extension_default(pattern, alg)
end

function _chordal_extension_default(pattern, alg)
    return throw(
        ArgumentError(
            "The chordal algorithm $(nameof(typeof(alg))) requires the optional " *
            "CliqueTrees.jl dependency, which is not loaded. Run `using CliqueTrees` " *
            "to enable it, or use the built-in AMDAlg() or MCSAlg() algorithms.",
        ),
    )
end
function _chordal_extension_default(pattern::SparseMatrixCSC, ::AMDAlg)
    return _symbolic_chordal_extension(pattern, AMD.amd(symmetric_pattern(pattern)))
end
function _chordal_extension_default(pattern::SparseMatrixCSC, ::MCSAlg)
    return _symbolic_chordal_extension(pattern, reverse(_mcs_order(pattern)[1]))
end

# The chordal completion from an elimination ordering: the fill pattern of
# the symbolic Cholesky elimination on the permuted pattern, and the maximal
# cliques in the original vertex ordering.
function _symbolic_chordal_extension(pattern::SparseMatrixCSC, p::AbstractVector{Int})
    n = LinearAlgebra.checksquare(pattern)
    P = symmetric_pattern(pattern)
    Ap = P[p, p]            # pattern in the elimination ordering
    # Symbolic Cholesky (up-looking, processed ascending): L_j is the full
    # column pattern of j plus, transitively, the columns of its upper
    # entries (k < j, the Rose–Tarjan fixpoint) — the fill pattern of the
    # factor. The upper entries act as merge sources and are then discarded,
    # so the final column keeps rows ≥ j only.
    L = [Int[] for _ in 1:n]
    for j in 1:n
        col = Int[j]
        for k in nzrange(Ap, j)
            i = rowvals(Ap)[k]
            i != j && push!(col, i)
        end
        merged = copy(col)
        seen = Set{Int}()
        idx = 1
        while idx <= length(merged)
            k = merged[idx]
            if k < j && !(k in seen)
                append!(merged, L[k])
                push!(seen, k)
            end
            idx += 1
        end
        sort!(merged)
        unique!(merged)
        L[j] = [i for i in merged if i >= j]
    end
    # The cliques of the completion: each vertex with its lower neighbors in L.
    cliques = Vector{Vector{Int}}()
    for j in 1:n
        C = Int[j]
        for k in L[j]
            k > j && push!(C, k)
        end
        push!(cliques, C)
    end
    maxcliques = _maximal_cliques(cliques)
    # Map back to the original vertex ordering (p[j] is the original index
    # of vertex j in the elimination ordering).
    return [sort!([p[i] for i in C]) for C in maxcliques]
end

function _maximal_cliques(cliques::Vector{Vector{Int}})
    # A clique is maximal unless contained in a larger one; check only
    # against larger cliques (sorted descending) with an early exit.
    order = sortperm(cliques; by=length, rev=true)
    sorted = cliques[order]
    out = Vector{Vector{Int}}()
    for (i, C) in enumerate(sorted)
        maximal = true
        for C2 in sorted[1:(i - 1)]
            if all(in(C2), C)
                maximal = false
                break
            end
        end
        maximal && push!(out, C)
    end
    return out
end

"""
    is_chordal(pattern::SparseMatrixCSC)

Whether the graph of the sparsity `pattern` (symmetrized, diagonal included)
is chordal: every cycle of length ≥ 4 has a chord. Implemented with Maximum
Cardinality Search (Tarjan–Yannakakis 1984): the reverse of an MCS numbering
is a perfect elimination ordering exactly when the graph is chordal, so the
test is to run MCS and verify that for every vertex its earlier MCS neighbors
form a clique. O(n + m) up to the clique checks. Loading the optional
CliqueTrees.jl package uses its `ischordal` instead.

A chordal pattern's chordal extension (see [`chordal_extension`](@ref)) is
itself — the decomposition introduces no fill and the maximal cliques are
those of the original graph.
"""
is_chordal(pattern::SparseMatrixCSC) = _is_chordal(pattern)

# The dispatch hook: the optional CliqueTrees extension specializes this.
_is_chordal(pattern) = _mcs_is_chordal(pattern)

# The Maximum Cardinality Search numbering of the pattern's graph (its
# reverse is a perfect elimination ordering when the graph is chordal).
function _mcs_order(pattern::SparseMatrixCSC)
    n = LinearAlgebra.checksquare(pattern)
    P = symmetric_pattern(pattern)
    adj = [Int[] for _ in 1:n]
    for j in 1:n
        for k in nzrange(P, j)
            i = rowvals(P)[k]
            i != j && push!(adj[j], i)
        end
    end
    # Maximum Cardinality Search with label buckets (O(n + m) amortized).
    labels = zeros(Int, n)
    processed = falses(n)
    buckets = [Int[] for _ in 0:n]
    for v in 1:n
        push!(buckets[1], v)  # all vertices start with label 0
    end
    order = Int[]
    maxlabel = 0
    while length(order) < n
        maxlabel >= 0 || break
        while maxlabel >= 0 && isempty(buckets[maxlabel + 1])
            maxlabel -= 1
        end
        maxlabel >= 0 || break
        v = pop!(buckets[maxlabel + 1])
        processed[v] && continue       # already processed
        labels[v] == maxlabel || continue  # stale bucket entry (re-labeled)
        processed[v] = true
        push!(order, v)
        for u in adj[v]
            processed[u] && continue
            labels[u] += 1
            push!(buckets[labels[u] + 1], u)
            maxlabel = max(maxlabel, labels[u])
        end
    end
    return order, adj
end

function _mcs_is_chordal(pattern::SparseMatrixCSC)
    order, adj = _mcs_order(pattern)
    # Tarjan–Yannakakis: the REVERSE of an MCS numbering is a perfect
    # elimination ordering exactly when the graph is chordal. In an MCS
    # numbering the first vertices are eliminated last, so the PEO condition
    # is: for every vertex, its EARLIER neighbors (smaller MCS position)
    # form a clique.
    n = LinearAlgebra.checksquare(pattern)
    adjset = [Set(adj[i]) for i in 1:n]
    pos = zeros(Int, n)
    for (i, v) in enumerate(order)
        pos[v] = i
    end
    for v in order
        earlier = [u for u in adj[v] if pos[u] < pos[v]]
        m = length(earlier)
        for a in 1:m
            for b in (a + 1):m
                earlier[b] in adjset[earlier[a]] || return false
            end
        end
    end
    return true
end

"""
    assemble_chordal(Ys, cliques, n)

Assembles the clique matrices `Ys` (the `|Cₖ| × |Cₖ|` block of the maximal
clique `cliques[k]` of a chordal graph on `n` vertices) into the global
`n × n` matrix

    A = Σₖ E(Cₖ)ᵀ Yₖ E(Cₖ),

the coupling map of the Fukuda chordal decomposition. Entries shared by
several cliques sum the contributions. The adjoint (an analytic ChainRules
`rrule`) scatters the cotangent back to the blocks, so the map
differentiates efficiently in both directions.
"""
function assemble_chordal(Ys, cliques, n::Int)
    T = promote_type(map(eltype, Ys)...)
    A = zeros(T, n, n)
    for (Y, C) in zip(Ys, cliques)
        for (a, i) in enumerate(C)
            for (b, j) in enumerate(C)
                A[i, j] += Y[a, b]
            end
        end
    end
    return A
end

function ChainRulesCore.rrule(::typeof(assemble_chordal), Ys, cliques, n)
    A = assemble_chordal(Ys, cliques, n)
    function assemble_pullback(Δ)
        Δ = ChainRulesCore.unthunk(Δ)
        ΔYs = [zero(Y) for Y in Ys]
        for (k, C) in enumerate(cliques)
            for (a, i) in enumerate(C)
                for (b, j) in enumerate(C)
                    ΔYs[k][a, b] = (Δ[i, j] + Δ[j, i]) / 2
                end
            end
        end
        return ChainRulesCore.NoTangent(),
        ΔYs, ChainRulesCore.NoTangent(),
        ChainRulesCore.NoTangent()
    end
    return A, assemble_pullback
end

# Normalize a label collection to a Set{Symbol}.
function _chordal_labels(labels)
    return if labels isa AbstractSet
        Set{Symbol}(labels)
    else
        Set{Symbol}(Symbol.(collect(labels)))
    end
end

# The PSD components of a VecModel's function-in-set constraints, in order:
# (constraint, component index, PSDCone) triples.
function _psd_components(model::VecModel)
    comps = Tuple{FunctionInSet,Int,PSDCone}[]
    for c in model.set_constraints.fs
        has_psd_component(c.set) || continue
        for (k, s) in enumerate(NonconvexCore.components(c.set))
            s isa PSDCone && push!(comps, (c, k, s))
        end
    end
    return comps
end

"""
    chordal_transform(model::VecModel, patterns; x0 = model.init, labels = nothing)

The exact chordal decomposition of the semidefinite components of `model`'s
function-in-set constraints: each selected `PSDCone` component with a
chordal sparsity pattern is rewritten with the Fukuda conversion —
one small PSD variable block per maximal clique of the chordal extension,
coupled to the original matrix by linear equalities on the entries of the
extension pattern (`P = Σₖ E(Cₖ)ᵀ Yₖ E(Cₖ)`, `Yₖ ⪰ 0`). The reformulation is
equivalent to `P(x) ⪰ 0`.

`patterns` is a vector with one sparsity pattern per PSD component of the
model's set constraints, in order (an entry may be `nothing` to probe the
pattern at `x0`); each pattern's graph is completed to a chordal extension
with [`chordal_extension`](@ref). Every nonzero entry of the component at
`x0` must lie inside the chordal extension, otherwise an error is thrown:
sparsity exploitation is only valid when the pattern covers all entries the
component can have.

`labels` selects which constraints to decompose: only PSD components whose
constraint flags intersect `labels` are rewritten (`labels = nothing`
selects all). The non-PSD components and the unselected constraints are
kept.

Returns `(dmodel, d0, mapper)` where `dmodel` is the transformed
[`VecModel`](@ref) (the original variables first, then one `PSDCone` block
per clique), `d0` its initial point, and `mapper` extracts the original
variables from a transformed point.
"""
function chordal_transform(model::VecModel, patterns; x0=model.init, labels=nothing)
    n0 = length(model.box_min)
    pats = patterns isa AbstractVector ? patterns : [patterns]
    # One pattern per PSD component of the set constraints, in order.
    comps0 = _psd_components(model)
    length(pats) == length(comps0) || throw(
        ArgumentError(
            "Got $(length(pats)) sparsity patterns for $(length(comps0)) PSD " *
            "components of the model's set constraints; provide one per component.",
        ),
    )
    sel = if labels === nothing
        trues(length(comps0))
    else
        begin
            labs = _chordal_labels(labels)
            [!isdisjoint(c.flags, labs) for (c, _, _) in comps0]
        end
    end
    # Auxiliary PSD blocks (new variables come after the original ones).
    block_ranges = UnitRange{Int}[]
    cliques_per = Vector{Vector{Int}}[]
    block_inits = Float64[]
    block_lbs = Float64[]
    block_ubs = Float64[]
    new_vars = VariableInSet[]
    synthetic_cs = FunctionInSet[]
    off = n0
    for (idx, ((c, k, s), pat)) in enumerate(zip(comps0, pats))
        sel[idx] || continue
        patc = if pat === nothing
            aggregate_sparsity(x -> NonconvexCore.set_value(c, x)[k], x0, s.dim)
        else
            pat
        end
        LinearAlgebra.checksquare(patc) == s.dim || throw(
            DimensionMismatch(
                "The sparsity pattern of PSD component $idx has size " *
                "$(size(patc, 1)) but the component dimension is $(s.dim).",
            ),
        )
        P0 = NonconvexCore.set_value(c, x0)[k]
        cliques = chordal_extension(patc)
        compl = _clique_pattern(cliques, s.dim)
        for i in 1:s.dim, j in 1:s.dim
            iszero(P0[i, j]) && continue
            compl[i, j] || throw(
                ArgumentError(
                    "The entry ($i, $j) of PSD component $idx is nonzero at `x0` " *
                    "but lies outside the chordal extension of its sparsity " *
                    "pattern; the decomposition would not be equivalent to the " *
                    "original constraint.",
                ),
            )
        end
        push!(cliques_per, cliques)
        for C in cliques
            sz = length(C)
            r = (off + 1):(off + sz ^ 2)
            blk = VariableInSet(PSDCone(sz), r; flags=c.flags)
            push!(new_vars, blk)
            push!(block_ranges, r)
            # The cone part of the block is added as an explicit synthetic
            # PSD constraint: solvers that consume the model directly (no
            # `tovecmodel` conversion) read set constraints, not the
            # variable blocks.
            push!(synthetic_cs, NonconvexCore.variable_set_constraints(blk)...)
            append!(block_lbs, fill(-Inf, sz^2))
            append!(block_ubs, fill(Inf, sz^2))
            append!(block_inits, vec(Matrix{Float64}(I, sz, sz)))
            off += sz^2
        end
    end
    # The coupling equalities (one per decomposed component, first in the
    # constraint list) and the kept constraints.
    set_cs = FunctionInSet[]
    ci = 0
    for (idx, ((c, k, s), pat)) in enumerate(zip(comps0, pats))
        sel[idx] || continue
        ci += 1
        cliques = cliques_per[ci]
        ranges = block_ranges[((ci - 1) * length(cliques) + 1):(ci * length(cliques))]
        entries = _coupling_entries(cliques, s.dim)
        fun = _coupling_fun(c, k, n0, cliques, ranges, entries, s.dim)
        push!(
            set_cs,
            FunctionInSet(fun, NonconvexCore.EqualTo(zeros(length(entries))), c.flags),
        )
    end
    # The synthetic PSD constraints of the clique variable blocks.
    append!(set_cs, synthetic_cs)
    for (idx, (c, _, _)) in enumerate(comps0)
        # Decomposed PSD components are replaced; the non-PSD components of
        # their constraints and all other constraints are kept (each PSD
        # component is dropped, the rest of the bundle is preserved).
        if sel[idx]
            for (k, s) in enumerate(NonconvexCore.components(c.set))
                s isa PSDCone && continue
                push!(
                    set_cs, FunctionInSet(x -> NonconvexCore.set_value(c, x)[k], s, c.flags)
                )
            end
        else
            push!(set_cs, FunctionInSet(x -> c(x), c.set, c.flags))
        end
    end
    d0 = vcat(float.(x0), block_inits)
    dm = VecModel(
        # objective (the original variables are the first n0 entries)
        Objective(
            y -> model.objective(y[1:n0]), model.objective.multiple, model.objective.flags
        ),
        # eq_constraints
        VectorOfFunctions([
            NonconvexCore.EqConstraint(y -> c(y[1:n0]), c.rhs, c.dim, c.flags) for
            c in model.eq_constraints.fs
        ]),
        # ineq_constraints
        VectorOfFunctions([
            NonconvexCore.IneqConstraint(y -> c(y[1:n0]), c.rhs, c.dim, c.flags) for
            c in model.ineq_constraints.fs
        ]),
        # sd_constraints
        VectorOfFunctions([
            NonconvexCore.SDConstraint(y -> c(y[1:n0]), c.dim) for
            c in model.sd_constraints.fs
        ]),
        # set_constraints: the coupling equalities first, then the kept ones
        VectorOfFunctions(set_cs),
        # set_variables: the new clique PSD blocks first, then the originals
        vcat(new_vars, model.set_variables),
        # box / init / integer
        vcat(model.box_min, block_lbs),
        vcat(model.box_max, block_ubs),
        vcat(float.(model.init), block_inits),
        vcat(model.integer, falses(length(block_inits))),
    )
    mapper = y -> y[1:n0]
    return dm, d0, mapper
end

# The symmetric pattern of the clique union (the chordal completion).
function _clique_pattern(cliques, n::Int)
    I = Int[]
    J = Int[]
    for C in cliques, i in C, j in C
        push!(I, i)
        push!(J, j)
    end
    return symmetric_pattern(sparse(I, J, trues(length(I)), n, n))
end

# The symmetric index pairs (i ≤ j) of the chordal completion: the entries
# on which the coupling P = ΣEᵀYE is enforced.
function _coupling_entries(cliques, n::Int)
    compl = _clique_pattern(cliques, n)
    return [(i, j) for j in 1:n for i in 1:j if compl[i, j]]
end

# The coupling residual P - ΣEᵀYE on the extension-pattern entries: the
# bundle is evaluated once, the clique blocks are read from y.
function _coupling_fun(c::FunctionInSet, k::Int, n0::Int, cliques, ranges, entries, n::Int)
    return function (y)
        P = NonconvexCore.set_value(c, y[1:n0])[k]
        A = assemble_chordal(
            [reshape(view(y, r), length(C), length(C)) for (r, C) in zip(ranges, cliques)],
            cliques,
            n,
        )
        return [P[i, j] - A[i, j] for (i, j) in entries]
    end
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
        if Δx !== nothing && !(Δx isa ChainRulesCore.AbstractZero) && any(!iszero, Δx)
            pattern[i, j] = true
        end
    end
    return symmetric_pattern(sparse(pattern))
end

"""
    BlockSDConstraint

A semidefinite constraint whose function returns a vector of matrices
(one per block) from a single evaluation. `chordal_decompose` uses it to
evaluate the matrix-valued function once and slice every clique block
from that single evaluation instead of re-evaluating it per clique.

The function must return a `Vector` (or tuple) of symmetric matrices of
sizes `dims`.
"""
struct BlockSDConstraint{F} <: NonconvexCore.AbstractConstraint
    f::F
    dims::Vector{Int}
end
(c::BlockSDConstraint)(args...; kwargs...) = c.f(args...; kwargs...)
NonconvexCore.getdim(c::BlockSDConstraint) = sum(c.dims)
getdims(c::BlockSDConstraint) = c.dims

# View any semidefinite constraint as its list of matrix blocks.
_sd_dims(c::NonconvexCore.SDConstraint) = [c.dim]
_sd_dims(c::BlockSDConstraint) = c.dims
_sd_blocks(c::NonconvexCore.SDConstraint, x) = (c(x),)
_sd_blocks(c::BlockSDConstraint, x) = Tuple(c(x))

"""
    chordal_decompose(model::VecModel, patterns; x0 = nothing)
    chordal_decompose(model::VecModel; x0)

Replace each semidefinite constraint `P(x) ⪰ 0` of `model` with a single
[`BlockSDConstraint`](@ref) whose function evaluates `P` once and returns
every maximal-clique submatrix `P(x)[C, C]` of a chordal extension of the
constraint's sparsity pattern.

!!! warning "This is a relaxation"
    The clique-submatrix constraints are only *implied* by `P ⪰ 0`; the
    converse fails in general (e.g. `tridiag(1, 1, 1)` has positive
    semidefinite 2×2 clique submatrices yet is indefinite). The rewritten
    model is therefore a RELAXATION: its feasible set contains the original
    one, and for minimization its optimal value is a lower bound. For
    element-assembled matrices whose element blocks are rank-one (e.g. truss
    stiffness matrices) the element-wise form is exact. Use
    [`chordal_transform`](@ref) for the exact Fukuda conversion.

`patterns` is a vector with one sparsity pattern per semidefinite constraint
(or a single pattern when the model has one constraint). Entries may be
`nothing` to probe the sparsity of the constraint at `x0` (the union of the
structural AD dependence pattern and the numerical support of `P(x0)`).

If `x0` is provided, every nonzero entry of `P(x0)` must lie inside the
pattern, otherwise an error is thrown: sparsity exploitation is only valid
when the pattern covers all entries the constraint can have.
"""
function chordal_decompose(model::VecModel, patterns; x0=nothing)
    n_sd = length(model.sd_constraints.fs)
    pats = patterns isa AbstractVector ? patterns : [patterns]
    length(pats) == n_sd || throw(
        ArgumentError(
            "Got $(length(pats)) sparsity patterns for $n_sd semidefinite constraints; provide one per constraint.",
        ),
    )
    sd_fs = NonconvexCore.AbstractConstraint[]
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
        dims = Int[length(C) for C in cliques]
        # A single constraint whose function evaluates `f` once and returns
        # every clique block of that one evaluation.
        push!(sd_fs, BlockSDConstraint(x -> begin
            P = f(x)
            return [P[C, C] for C in cliques]
        end, dims))
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
    return chordal_decompose(model, [nothing for _ in model.sd_constraints.fs]; x0=x0)
end
