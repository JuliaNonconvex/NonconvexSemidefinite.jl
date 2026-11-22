# Kočvara's structured domain decomposition (SDD) of large semidefinite
# constraints on finite-element meshes.
#
# A matrix assembled element-wise on a mesh is bordered block-diagonal
# ("arrow-type") after a subdomain reordering: with the interior dofs of
# each subdomain first and the interface dofs (shared by several subdomains)
# last,
#
#     P = [ P₁  0  ⋯  0  B₁
#           0  P₂ ⋯  0  B₂
#           ⋮     ⋱     ⋮
#           B₁ᵀ B₂ᵀ ⋯  Bₖᵀ P_Γ ]
#
# where Pₖ are the subdomain interior blocks, Bₖ the interior-interface
# couplings and P_Γ the interface block. By the Schur-complement theorem,
# such a P is positive semidefinite if and only if every interior block is
# positive semidefinite AND the interface Schur complement
#
#     S = P_Γ − Σₖ Bₖᵀ Pₖ⁻¹ Bₖ
#
# is positive semidefinite. `sdd_partition` finds the subdomain partition of
# a general mesh (geometric recursive bisection of the cell centroids — any
# cell type works); `sdd_transform` rewrites a large PSD constraint
# `P(x) ⪰ 0` into the small subdomain-block PSD constraints and the
# interface Schur-complement constraint, evaluated as ONE bundle so the
# element assembly runs once per call. The transformation is exact.

using LinearAlgebra

using NonconvexCore: NonconvexCore
using NonconvexCore: VecModel, FunctionInSet, VectorOfFunctions, Objective

"""
    SDDPartition(interiors, interface, n)

A subdomain partition of `n` degrees of freedom: `interiors[k]` are the dofs
interior to subdomain `k` (in the original numbering) and `interface` the
dofs shared by several subdomains. See [`sdd_partition`](@ref).
"""
struct SDDPartition
    interiors::Vector{Vector{Int}}
    interface::Vector{Int}
    n::Int
end
nsubdomains(p::SDDPartition) = length(p.interiors)

"""
    sdd_partition(coords::AbstractMatrix, cell_dofs; n_subdomains = 2)

Partitions a mesh into `n_subdomains` subdomains for the structured domain
decomposition (Kočvara's SDD): `coords` holds the centroid coordinates of
each cell (one row per cell, `size(coords, 1) == length(cell_dofs)`) and
`cell_dofs[c]` the degrees of freedom of cell `c`. The cells are split by
recursive bisection along the longest coordinate axis; the dofs shared by
cells of several subdomains become the interface, the others the interior of
their subdomain. Works for any mesh (any cell type); load the optional
Ferrite.jl dependency for the `sdd_partition(grid, dh)` convenience method
that reads a Ferrite grid and dof handler.
"""
function sdd_partition(
    coords::AbstractMatrix,
    cell_dofs::AbstractVector{<:AbstractVector{Int}};
    n_subdomains::Int=2,
)
    n_subdomains >= 1 || throw(ArgumentError("n_subdomains must be at least 1. "))
    ncells = length(cell_dofs)
    size(coords, 1) == ncells || throw(
        DimensionMismatch(
            "Got $(size(coords, 1)) coordinate rows for $ncells cells; provide one " *
            "cell centroid per cell.",
        ),
    )
    # Recursive coordinate bisection of the cells.
    owner = zeros(Int, ncells)
    function bisect(cells::Vector{Int}, s0::Int, ns::Int)
        if ns == 1
            for c in cells
                owner[c] = s0
            end
            return nothing
        end
        n_left = ns ÷ 2
        n_right = ns - n_left
        pts = coords[cells, :]
        ext = vec(maximum(pts; dims=1) - minimum(pts; dims=1))
        ax = argmax(ext)
        ord = sortperm(cells; by=c -> coords[c, ax])
        k = max(1, round(Int, length(cells) * n_left / ns))
        k = min(k, length(cells) - 1)
        bisect(cells[ord[1:k]], s0, n_left)
        bisect(cells[ord[(k + 1):end]], s0 + n_left, n_right)
        return nothing
    end
    ncells >= n_subdomains || throw(
        ArgumentError("Cannot partition $ncells cells into $n_subdomains subdomains. ")
    )
    bisect(collect(1:ncells), 1, n_subdomains)
    # Dof classification: a dof of cells in several subdomains is an
    # interface dof; the others are interior to their subdomain.
    ndofs = isempty(cell_dofs) ? 0 : maximum(maximum, cell_dofs)
    dof_owner = [Set{Int}() for _ in 1:ndofs]
    for (c, dofs) in enumerate(cell_dofs)
        for d in dofs
            push!(dof_owner[d], owner[c])
        end
    end
    interiors = [Int[] for _ in 1:n_subdomains]
    interface = Int[]
    for d in 1:ndofs
        if length(dof_owner[d]) > 1
            push!(interface, d)
        elseif !isempty(dof_owner[d])
            push!(interiors[only(dof_owner[d])], d)
        end
    end
    filter!(!isempty, interiors)
    sort!(interface)
    return SDDPartition(interiors, interface, ndofs)
end

# The SDD bundle of a matrix function: the subdomain interior blocks and
# the interface Schur complement, from one evaluation of P.
function _sdd_bundle(Pfun, partition::SDDPartition)
    I = partition.interiors
    Γ = partition.interface
    return function (x)
        P = Pfun(x)
        blocks = [P[I_k, I_k] for I_k in I]
        nΓ = length(Γ)
        if nΓ == 0
            return Tuple(blocks)
        end
        Pgg = Matrix(P[Γ, Γ])
        for I_k in I
            B = P[Γ, I_k]             # nΓ × |I_k|
            X = P[I_k, I_k] \ B'      # |I_k| × nΓ
            Pgg = Pgg - B * X         # B Pₖₖ⁻¹ Bᵀ
        end
        return (blocks..., Pgg)
    end
end

"""
    sdd_transform(model::VecModel, partition; x0 = model.init, labels = nothing)

Kočvara's structured domain decomposition of the semidefinite components of
`model`'s function-in-set constraints: each selected `PSDCone` component
`P(x) ⪰ 0` whose dimension matches `partition` is rewritten as a bundle
constraint — the subdomain interior blocks `P(x)[I_k, I_k] ⪰ 0` and the
interface Schur complement `S(x) ⪰ 0` as the PSD components of one
[`SetProduct`](@ref), so the element assembly behind `P` runs once per
evaluation. The transformation is exact for matrices with the bordered
block-diagonal structure of the partition (Schur-complement theorem).

`labels` selects which constraints to decompose: only PSD components whose
constraint flags intersect `labels` are rewritten (`labels = nothing`
selects all); the other constraints are kept. Returns `(dmodel, d0,
mapper)`; no auxiliary variables are introduced, so `d0` is the original
initial point and `mapper` the identity.
"""
function sdd_transform(
    model::VecModel, partition::SDDPartition; x0=model.init, labels=nothing
)
    comps0 = _psd_components(model)
    sel = if labels === nothing
        trues(length(comps0))
    else
        begin
            labs = _chordal_labels(labels)
            [!isdisjoint(c.flags, labs) for (c, _, _) in comps0]
        end
    end
    set_cs = FunctionInSet[]
    for (idx, (c, k, s)) in enumerate(comps0)
        if sel[idx]
            s.dim == partition.n || throw(
                DimensionMismatch(
                    "The SDD partition has $(partition.n) degrees of freedom but PSD " *
                    "component $idx has dimension $(s.dim).",
                ),
            )
            bundle = _sdd_bundle(x -> NonconvexCore.set_value(c, x)[k], partition)
            sets = NonconvexCore.AbstractSet[
                PSDCone(length(I_k)) for I_k in partition.interiors
            ]
            isempty(partition.interface) ||
                push!(sets, PSDCone(length(partition.interface)))
            push!(set_cs, FunctionInSet(bundle, NonconvexCore.SetProduct(sets), c.flags))
        else
            push!(set_cs, FunctionInSet(x -> c(x), c.set, c.flags))
        end
    end
    n0 = length(model.box_min)
    dm = VecModel(
        Objective(
            y -> model.objective(y[1:n0]), model.objective.multiple, model.objective.flags
        ),
        VectorOfFunctions([
            NonconvexCore.EqConstraint(y -> c(y[1:n0]), c.rhs, c.dim, c.flags) for
            c in model.eq_constraints.fs
        ]),
        VectorOfFunctions([
            NonconvexCore.IneqConstraint(y -> c(y[1:n0]), c.rhs, c.dim, c.flags) for
            c in model.ineq_constraints.fs
        ]),
        VectorOfFunctions([
            NonconvexCore.SDConstraint(y -> c(y[1:n0]), c.dim) for
            c in model.sd_constraints.fs
        ]),
        VectorOfFunctions(set_cs),
        model.set_variables,
        model.box_min,
        model.box_max,
        model.init,
        model.integer,
    )
    return dm, float.(x0), y -> y
end

# ------------------------------------------------------------------
# Kočvara's arrow decomposition (M. Kočvara, "Decomposition of arrow type
# positive semidefinite matrices with application to topology optimization",
# Math. Program. 190:105–134, 2021, Theorem 2.3 + Section 4, Case B).
#
# The minimum-compliance topology optimization SDP
#
#     min γ  s.t.  Z(x, γ) = [ K(x)  f ; fᵀ  γ ] ⪰ 0,  Σxᵢ ≤ V,  x̄ ≤ x ≤ x̄
#
# with K(x) = Σᵢ xᵢKᵢ assembled element-wise is rewritten with overlapping
# subdomains: each subdomain k (dofs Iₖ, elements Dₖ) gets its own PSD
# constraint
#
#     Z⁽ᵏ⁾_B = [ K⁽ᵏ⁾(x)  f⁽ᵏ⁾ + g⁽ᵏ⁾ ; (f⁽ᵏ⁾ + g⁽ᵏ⁾)ᵀ  γₖ ] ⪰ 0
#
# on Iₖ ∪ {γ}, where K⁽ᵏ⁾(x) = Σ_{i∈Dₖ} xᵢKᵢ, the fictitious loads
#
#     g⁽ᵏ⁾ = -Σ_{ℓ<k} g_{ℓ,k} + Σ_{ℓ>k} g_{k,ℓ}
#
# live on the interface I_{k,ℓ} = Iₖ ∩ I_ℓ (vector variables, not matrix
# ones), and the γₖ sum to γ. The transformation is exact (Theorem 2.3):
# Z ⪰ 0 ⟺ all Z⁽ᵏ⁾_B ⪰ 0, since Σₖ Z⁽ᵏ⁾_B = Z(x, Σₖγₖ).
# ------------------------------------------------------------------

"""
    ArrowSubdomain(Kfun, f, I)

One subdomain of the arrow decomposition ([`arrow_decompose`](@ref)): `Kfun`
maps the design variables to the subdomain stiffness matrix
`K⁽ᵏ⁾(x) = Σ_{i∈Dₖ} xᵢKᵢ` (the elements of the subdomain only, an
`|I| × |I|` matrix in the global dof ordering), `f` is the load vector
restricted to the subdomain's degrees of freedom `I` (global dof indices).
"""
struct ArrowSubdomain{F}
    Kfun::F
    f::Vector{Float64}
    I::Vector{Int}
end

"""
    ArrowIntersection(k, l, dofs)

An interface of the arrow decomposition: `dofs` are the global degrees of
freedom shared by subdomains `k` and `l` (`k < l`), i.e. `I_k ∩ I_l`; the
fictitious load `g_{k,l} ∈ ℝ^{|dofs|}` lives on it.
"""
struct ArrowIntersection
    k::Int
    l::Int
    dofs::Vector{Int}
end

"""
    arrow_decompose(model::VecModel, subdomains, intersections;
                    x0 = model.init, labels = nothing)

Kočvara's arrow decomposition (Theorem 2.3 of Kočvara, Math. Program. 2021):
rewrites the minimum-compliance SDP constraint `Z(x, γ) = [K(x) f; fᵀ γ] ⪰ 0`
(dimension `n+1`, with the design `x` first and `γ` the last variable) into
one PSD constraint per subdomain,

    [ K⁽ᵏ⁾(x)  f⁽ᵏ⁾ + g⁽ᵏ⁾ ; (f⁽ᵏ⁾ + g⁽ᵏ⁾)ᵀ  γₖ ] ⪰ 0,

replacing the matrix-sized coupling variables of the chordal decomposition
by the vector fictitious loads `g_{k,ℓ}` on the interfaces and the scalars
`γₖ` (the objective becomes `Σₖ γₖ`). The transformation is exact:
`Σₖ Z⁽ᵏ⁾_B = Z(x, Σₖγₖ)`.

`subdomains` is a vector of [`ArrowSubdomain`](@ref)s (their dof sets may
overlap on the interfaces) and `intersections` the [`ArrowIntersection`](@ref)s
with `k < l`. `labels` selects the PSD constraint to decompose (exactly one
must be selected; `labels = nothing` selects all, which must then be a
single PSD component). Returns `(dmodel, d0, mapper)` with the new variables
(`γₖ` then the `g_{k,ℓ}`) appended after the original ones.
"""
function arrow_decompose(
    model::VecModel, subdomains, intersections; x0=model.init, labels=nothing
)
    p = length(subdomains)
    p >= 2 || throw(ArgumentError("The arrow decomposition needs at least 2 subdomains. "))
    comps0 = _psd_components(model)
    sel = if labels === nothing
        trues(length(comps0))
    else
        [!isdisjoint(c.flags, _chordal_labels(labels)) for (c, _, _) in comps0]
    end
    count(sel) == 1 || throw(
        ArgumentError(
            "arrow_decompose requires exactly one selected PSD component (the " *
            "Z(x, γ) constraint); got $(count(sel)).",
        ),
    )
    ci = findfirst(sel)
    c, k0, s = comps0[ci]
    n0 = length(model.box_min)
    ndofs = s.dim - 1
    n_x = n0 - 1   # the design variables (γ is the last original variable)
    n_x >= 1 || throw(
        ArgumentError(
            "arrow_decompose expects the model variables to be the design x " *
            "followed by the compliance γ; got $n0 variables.",
        ),
    )
    # Validate the dof coverage: the union of the subdomain dof sets is 1:ndofs.
    covered = sort!(unique!(vcat((sd.I for sd in subdomains)...)))
    covered == collect(1:ndofs) || throw(
        ArgumentError(
            "The subdomain dof sets must cover 1:$ndofs; the union is missing " *
            "$(setdiff(collect(1:ndofs), covered)).",
        ),
    )
    # Local positions of each interface dof within its two subdomains.
    pos = Dict{Tuple{Int,Int},Vector{Int}}()  # (k, l) -> positions of g_{k,l} in I_k
    for it in intersections
        (it.k < it.l && it.k in 1:p && it.l in 1:p) || throw(
            ArgumentError(
                "ArrowIntersection must have 1 ≤ k < l ≤ p; got ($(it.k), $(it.l))."
            ),
        )
        Ik = subdomains[it.k].I
        Il = subdomains[it.l].I
        posk = Int[]
        posl = Int[]
        for d in it.dofs
            a = findfirst(==(d), Ik)
            b = findfirst(==(d), Il)
            (a === nothing || b === nothing) && throw(
                ArgumentError(
                    "Interface dof $d of subdomains ($(it.k), $(it.l)) is not in " *
                    "both dof sets.",
                ),
            )
            push!(posk, a)
            push!(posl, b)
        end
        pos[(it.k, it.l)] = posk
        pos[(it.l, it.k)] = posl
    end
    # New variables: γₖ (p), then the fictitious loads g_{k,l}.
    γ0 = x0[n0]
    ranges = UnitRange{Int}[]
    γ_ranges = Vector{UnitRange{Int}}(undef, p)
    off = n0
    for k in 1:p
        γ_ranges[k] = (off + 1):(off + 1)
        off += 1
    end
    g_ranges = Dict{Tuple{Int,Int},UnitRange{Int}}()
    for it in intersections
        r = (off + 1):(off + length(it.dofs))
        g_ranges[(it.k, it.l)] = r
        push!(ranges, r)
        off += length(it.dofs)
    end
    d0 = vcat(float.(x0), fill(γ0 / p, p), zeros(off - n0 - p))
    # The decomposed constraints: one PSD constraint per subdomain.
    set_cs = FunctionInSet[]
    for (k, sd) in enumerate(subdomains)
        f_k = sd.f
        Ik = sd.I
        nk = length(Ik)
        fun = function (v)
            Kk = sd.Kfun(v[1:n_x])
            Z = zeros(Float64, nk + 1, nk + 1)
            Z[1:nk, 1:nk] .= Kk
            last = copy(f_k)
            for it in intersections
                k2, l2 = it.k, it.l
                k2 == k && (last[pos[(k2, l2)]] .+= v[g_ranges[(k2, l2)]])
                l2 == k && (last[pos[(l2, k2)]] .-= v[g_ranges[(k2, l2)]])
            end
            Z[1:nk, nk + 1] .= last
            Z[nk + 1, 1:nk] .= last
            Z[nk + 1, nk + 1] = v[γ_ranges[k]][1]
            return Z
        end
        push!(set_cs, FunctionInSet(fun, PSDCone(nk + 1), c.flags))
    end
    # The other constraints pass through unchanged (the variables are only
    # extended, so they keep reading their original prefix).
    for (idx, (cc, _, _)) in enumerate(comps0)
        idx == ci && continue
        push!(set_cs, FunctionInSet(y -> cc(y), cc.set, cc.flags))
    end
    g0 = zeros(off - n0 - p)
    dm = VecModel(
        Objective(
            y -> sum(y[γ_ranges[k]] for k in 1:p),
            model.objective.multiple,
            model.objective.flags,
        ),
        VectorOfFunctions([
            NonconvexCore.EqConstraint(y -> c(y[1:n0]), c.rhs, c.dim, c.flags) for
            c in model.eq_constraints.fs
        ]),
        VectorOfFunctions([
            NonconvexCore.IneqConstraint(y -> c(y[1:n0]), c.rhs, c.dim, c.flags) for
            c in model.ineq_constraints.fs
        ]),
        VectorOfFunctions([
            NonconvexCore.SDConstraint(y -> c(y[1:n0]), c.dim) for
            c in model.sd_constraints.fs
        ]),
        VectorOfFunctions(set_cs),
        model.set_variables,
        vcat(model.box_min, fill(-Inf, off - n0)),
        vcat(model.box_max, fill(Inf, off - n0)),
        vcat(float.(model.init), fill(γ0 / p, p), g0),
        vcat(model.integer, falses(off - n0)),
    )
    return dm, d0, y -> y[1:n0]
end
