# Cutting-plane method for semidefinite constraints.
#
# The PSD constraints are dropped from the NLP subproblems and reintroduced
# as linear cuts v'P(x)v ≥ 0 taken from the most negative eigenpair of each
# violated matrix, computed with an iterative eigensolver (ARPACK): no dense
# eigendecomposition of the full matrix is ever needed. Every cut is a valid
# inequality for P(x) ⪰ 0, so the iterates converge to a PSD-feasible point
# (up to `eig_tol`).
#
# `block_size` restricts the eigen-solve to diagonal blocks of the matrix.
# Block cuts remain valid inequalities (a PSD matrix has PSD principal
# submatrices) and keep each eigen-solve at O(block) work; with blocks
# matching a chordal clique structure the enforced condition is the
# clique-submatrix one — a relaxation, like `chordal_decompose`.

using Arpack: Arpack
using LinearAlgebra

using NonconvexCore: NonconvexCore
using NonconvexCore:
    VecModel, VectorOfFunctions, IneqConstraint, AbstractModel, AbstractOptimizer

"""
    SDCuttingPlaneAlg(sub_alg)

Cutting-plane meta-algorithm for models with semidefinite constraints
([`SDConstraint`](@ref) and `PSDCone` components of function-in-set
constraints): the PSD constraints are dropped from the subproblems and
reintroduced as linear cuts `v'P(x)v ≥ 0` from the most negative eigenpair
of each violated matrix, computed with an iterative eigensolver (ARPACK).
`sub_alg` solves the NLP subproblems (e.g. `IpoptAlg()`); it must handle the
model's equality, inequality and box constraints.
"""
struct SDCuttingPlaneAlg{S} <: AbstractOptimizer
    sub_alg::S
end
function SDCuttingPlaneAlg()
    return throw(
        ArgumentError(
            "SDCuttingPlaneAlg needs a sub-solver for the NLP subproblems, e.g. " *
            "SDCuttingPlaneAlg(IpoptAlg()).",
        ),
    )
end

"""
    SDCuttingPlaneOptions(; eig_tol = 1e-5, max_cuts = 100, max_iterations = 10,
                            block_size = nothing, sub_options = NamedTuple())

Options of [`SDCuttingPlaneAlg`](@ref):

- `eig_tol`: eigenvalue feasibility tolerance — the algorithm stops once
  every PSD constraint (block) has smallest eigenvalue ≥ `-eig_tol`.
- `max_cuts`: the maximum total number of cuts added.
- `max_iterations`: the maximum number of subproblem solves.
- `block_size`: the size of the diagonal blocks the iterative eigensolver is
  applied to (`nothing` = the whole matrix). Block cuts are valid
  inequalities; with blocks matching a chordal clique structure the enforced
  condition is the clique-submatrix (relaxation) one.
- `sub_options`: the options passed to the sub-solver.
"""
mutable struct SDCuttingPlaneOptions{SO}
    eig_tol::Float64
    max_cuts::Int
    max_iterations::Int
    block_size::Union{Nothing,Int}
    sub_options::SO
end
function SDCuttingPlaneOptions(;
    eig_tol=1e-5,
    max_cuts=100,
    max_iterations=10,
    block_size=nothing,
    sub_options=NamedTuple(),
)
    @assert eig_tol > 0 "eig_tol should be positive. "
    @assert max_cuts >= 0 "max_cuts should be non-negative. "
    @assert max_iterations >= 1 "max_iterations should be at least 1. "
    block_size === nothing || @assert block_size >= 1 "block_size should be at least 1. "
    return SDCuttingPlaneOptions(eig_tol, max_cuts, max_iterations, block_size, sub_options)
end

"""
    SDCuttingPlaneResult

The result of [`SDCuttingPlaneAlg`](@ref) with fields:

- `minimum`: the objective value at the returned minimizer.
- `minimizer`: the point found by the last subproblem solve.
- `n_cuts`: the total number of cuts added.
- `min_eigvals`: the worst eigenvalue (over all PSD constraints/blocks) after
  each iteration; the last entry is ≥ `-eig_tol` when the algorithm
  converged.
"""
struct SDCuttingPlaneResult{M1,M2,E}
    minimum::M1
    minimizer::M2
    n_cuts::Int
    min_eigvals::E
end

mutable struct SDCuttingPlaneWorkspace{
    M<:VecModel,X<:AbstractVector,O<:SDCuttingPlaneOptions,S<:AbstractOptimizer
} <: NonconvexCore.Workspace
    model::M
    x0::X
    options::O
    sub_alg::S
end

function NonconvexCore.Workspace(
    model::VecModel, optimizer::SDCuttingPlaneAlg, x0, args...; options, kwargs...
)
    return SDCuttingPlaneWorkspace(model, copy(x0), options, optimizer.sub_alg)
end

# The semidefinite constraints of a VecModel as (matrix function, dim) pairs:
# the classic SD constraints plus the PSD components of function-in-set
# constraints (including products and the synthetic blocks of PSD variables).
function _psd_funs(model::VecModel)
    fs = Tuple{Any,Int}[]
    for sd in model.sd_constraints.fs
        push!(fs, (sd.f, sd.dim))
    end
    for c in model.set_constraints.fs
        for (k, s) in enumerate(NonconvexCore.components(c.set))
            s isa PSDCone || continue
            push!(fs, (x -> NonconvexCore.set_value(c, x)[k], s.dim))
        end
    end
    return fs
end

# The most negative eigenpair of a symmetric matrix, via the dense
# eigendecomposition for small blocks and the iterative (ARPACK) solver for
# larger ones.
function _min_eigenpair(S::Symmetric)
    n = size(S, 1)
    if n <= 6
        F = eigen(Symmetric(Matrix(S)))
        i = argmin(F.values)
        return F.values[i], collect(F.vectors[:, i])
    else
        d, V, nconv = Arpack.eigs(S; nev=1, which=:SR, tol=1e-10)
        nconv >= 1 ||
            @warn "The iterative eigensolver did not converge; using the last Ritz pair."
        return d[1], collect(V[:, 1])
    end
end

# All (block, eigenvalue, eigenvector) triples to check for P.
function _min_eigpairs(P::AbstractMatrix, block_size)
    n = size(P, 1)
    S = Symmetric(Matrix(P))
    if block_size === nothing
        λ, v = _min_eigenpair(S)
        return [(λ, v, 1:n)]
    end
    out = Tuple{Float64,Vector{Float64},UnitRange{Int}}[]
    for b in 1:block_size:n
        r = b:min(b + block_size - 1, n)
        λ, v = _min_eigenpair(Symmetric(Matrix(S[r, r])))
        push!(out, (λ, v, r))
    end
    return out
end

# The cut v'P(x)[r,r]v ≥ 0 as an inequality function (f ≤ 0).
function _cut_function(Pfun, v::AbstractVector, r::UnitRange{Int})
    return function (x)
        B = Pfun(x)
        Bb = length(r) == size(B, 1) ? B : B[r, r]
        return -(v' * Bb * v)
    end
end

function _with_cuts(model::VecModel, cuts)
    fs = NonconvexCore.AbstractConstraint[model.ineq_constraints.fs...; cuts...]
    return VecModel(
        model.objective,
        model.eq_constraints,
        VectorOfFunctions(fs),
        model.sd_constraints,
        model.set_constraints,
        model.set_variables,
        model.box_min,
        model.box_max,
        model.init,
        model.integer,
    )
end

function NonconvexCore.optimize!(workspace::SDCuttingPlaneWorkspace)
    @unpack model, x0, options, sub_alg = workspace
    @unpack eig_tol, max_cuts, max_iterations, block_size, sub_options = options
    psd = _psd_funs(model)
    isempty(psd) && throw(
        ArgumentError("SDCuttingPlaneAlg requires at least one semidefinite constraint."),
    )
    cuts = IneqConstraint[]
    cutvecs = Vector{Float64}[]
    x = copy(x0)
    min_eigs = Float64[]
    n_cuts = 0
    for _ in 1:max_iterations
        m = _with_cuts(model, cuts)
        r = optimize(m, sub_alg, x; options=sub_options)
        x = r.minimizer
        worst = 0.0
        violated = Tuple{Vector{Float64},UnitRange{Int},Any}[]
        for (Pfun, dim) in psd
            P = Pfun(x)
            size(P) == (dim, dim) || throw(
                DimensionMismatch(
                    "A semidefinite constraint function returned a $(size(P, 1))×$(size(P, 2)) " *
                    "matrix but a $dim×$dim one was expected.",
                ),
            )
            for (λ, v, r) in _min_eigpairs(P, block_size)
                λ < -eig_tol || continue
                worst = min(worst, λ)
                push!(violated, (v, r, Pfun))
            end
        end
        push!(min_eigs, worst)
        isempty(violated) && break
        added = 0
        for (v, r, Pfun) in violated
            n_cuts >= max_cuts && break
            # skip cuts (anti)parallel to an earlier one: they add nothing
            any(pv -> length(pv) == length(v) && abs(dot(pv, v)) > 1 - 1e-10, cutvecs) &&
                continue
            push!(cuts, IneqConstraint(_cut_function(Pfun, v, r), 0.0))
            push!(cutvecs, v)
            n_cuts += 1
            added += 1
        end
        added == 0 && break   # every violated direction is already cut
    end
    obj = getobjective(model)
    obj === nothing && throw(ArgumentError("The model has no objective."))
    return SDCuttingPlaneResult(obj(x), x, n_cuts, min_eigs)
end

# The semidefinite constraints are handled by the cutting planes; the
# remaining capabilities are those of the sub-solver.
function NonconvexCore.solver_capabilities(alg::SDCuttingPlaneAlg)
    sub = solver_capabilities(alg.sub_alg)
    return merge_capabilities(
        sub; linear_sd=true, nonlinear_sd=true, sets_sd=true, variable_sets_sd=true
    )
end

function NonconvexCore._optimize_precheck(
    model::AbstractModel, optimizer::SDCuttingPlaneAlg, args...; kwargs...
)
    return nothing
end
