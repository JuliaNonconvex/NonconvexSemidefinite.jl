# Quadratic re-parameterization of semidefinite constraints.
#
# Each semidefinite constraint matrix A(x) ⪰ 0 whose variables follow the
# (x_L, x_D) decompress layout is re-parameterized as A = L Lᵀ (full-rank
# Cholesky factor with diag(L) ≥ diag_floor) or A = R Rᵀ (Burer-Monteiro
# low-rank factor R ∈ R^{n×r}). The semidefinite constraint is eliminated and
# the sub-problem (unconstrained/box-constrained in the factor variables) is
# solved with `sub_alg`.

"""
    SDPQuadOptions(; sub_options, rank = nothing, diag_floor = 1e-6)

Options of [`SDPQuadAlg`](@ref): `rank = nothing` selects the full-rank
Cholesky re-parameterization `A = L Lᵀ` (with `diag(L) ≥ diag_floor` bounds);
`rank = r` selects the Burer-Monteiro low-rank re-parameterization
`A = R Rᵀ` with `R ∈ R^{n×r}`.
"""
mutable struct SDPQuadOptions{SO}
    sub_options::SO
    rank::Union{Nothing,Int}
    diag_floor::Real
end
function SDPQuadOptions(; sub_options, rank=nothing, diag_floor=1e-6)
    (rank === nothing || rank >= 1) ||
        throw(ArgumentError("rank must be nothing (full rank) or a positive integer"))
    diag_floor >= 0 || throw(ArgumentError("diag_floor must be non-negative"))
    return SDPQuadOptions(sub_options, rank, diag_floor)
end

"""
    SDPQuadAlg(sub_alg)

Meta-algorithm eliminating semidefinite constraints through quadratic
re-parameterization. Every semidefinite constraint of the model must be a
`decompress_symmetric`-style matrix variable chunk in the `(x_L, x_D)` layout,
and the model variables must consist exactly of those chunks. The
re-parameterized problem is solved with `sub_alg`.
"""
struct SDPQuadAlg{Alg<:AbstractOptimizer} <: AbstractOptimizer
    sub_alg::Alg
end
function SDPQuadAlg(; sub_alg)
    return SDPQuadAlg(sub_alg)
end

function NonconvexCore.solver_capabilities(alg::SDPQuadAlg)
    sub = solver_capabilities(alg.sub_alg)
    return merge_capabilities(sub; linear_sd=true, nonlinear_sd=true)
end

function _optimize_precheck(model::AbstractModel, optimizer::SDPQuadAlg, args...; kwargs...)
    return nothing
end

struct SDPQuadResult{M1,M2}
    minimum::M1
    minimizer::M2
end

mutable struct SDPQuadWorkspace{
    M<:VecModel,X<:AbstractVector,O<:SDPQuadOptions,S<:AbstractOptimizer
} <: Workspace
    model::M
    x0::X
    options::O
    sub_alg::S
end

# layout of the sd-matrix chunks: (offset, k, n) with k = n(n-1)/2
function _sd_layout(model::VecModel)
    offset = 0
    layouts = Tuple{Int,Int,Int}[]
    for sd in model.sd_constraints.fs
        n = sd.dim
        k = n * (n - 1) ÷ 2
        push!(layouts, (offset, k, n))
        offset += k + n
    end
    return layouts
end

function _check_layout(model::VecModel)
    layouts = _sd_layout(model)
    total = isempty(layouts) ? 0 : last(layouts)[1] + last(layouts)[2] + last(layouts)[3]
    total == length(model.box_min) || throw(
        ArgumentError(
            "SDPQuadAlg requires the model variables to consist exactly of the " *
            "semidefinite matrix chunks in the (x_L, x_D) layout ($total entries " *
            "expected, got $(length(model.box_min))).",
        ),
    )
    return layouts
end

# Lower-triangular L from a flat chunk [strict lower entries; diagonal],
# built functionally (comprehension) so that every AD backend (including
# ReverseDiff) can differentiate through the re-parameterization. The strict
# lower chunk index of (i, j) is column-major: (j-1)*n - (j-1)j/2 + (i-j).
function _cholesky_param(chunk, n, k)
    return [
        if i > j
            chunk[(j - 1) * n - (j - 1) * j ÷ 2 + (i - j)]
        elseif i == j
            chunk[k + i]
        else
            zero(eltype(chunk))
        end for i in 1:n, j in 1:n
    ]
end

function _unpack_lowrank(chunk, n, r)
    return reshape(@view(chunk[1:(n * r)]), n, r)
end

function _factor_link(layouts, rank, nvars_new)
    # maps the new (factor) variable vector to the old (x_L, x_D) variable
    # vector
    return function _link(fx)
        # fully functional (no in-place operations) so that every AD backend
        # can differentiate through the re-parameterization
        return reduce(
            vcat,
            (
                begin
                    if rank === nothing
                        L = _cholesky_param(@view(fx[(offset + 1):(offset + k + n)]), n, k)
                        A = L * L'
                    else
                        R = _unpack_lowrank(
                            @view(fx[(offset + 1):(offset + n * rank)]), n, rank
                        )
                        A = R * R'
                    end
                    vcat([A[lin] for lin in lowertriangind(A)], [A[i, i] for i in 1:n])
                end for (offset, k, n) in layouts
            ),
        )
    end
end

function _factor_chunk_size(n, rank)
    k = n * (n - 1) ÷ 2
    return rank === nothing ? k + n : n * rank
end

function _initial_factors(model::VecModel, layouts, rank, diag_floor)
    fx = Float64[]
    for (offset, k, n) in layouts
        A0 = Symmetric(
            decompress_symmetric(
                @view(model.init[(offset + 1):(offset + k)]),
                @view(model.init[(offset + k + 1):(offset + k + n)]),
            ),
        )
        if rank === nothing
            F = cholesky(Matrix(A0) + 1e-6 * I; check=false)
            L = issuccess(F) ? F.L : Matrix(1e-3 * I, n, n)
            chunk = [L[lowertriangind(L)]; diag(L)]
            chunk[k .+ (1:n)] .= max.(diag(L), diag_floor)
            append!(fx, chunk)
        else
            F = eigen(Matrix(A0))
            r = min(rank, n)
            v = max.(F.values, 0.0)
            R = F.vectors[:, 1:r] * Diagonal(sqrt.(v[1:r]))
            append!(fx, vec(R))
        end
    end
    return fx
end

function _factor_bounds(layouts, rank, diag_floor)
    lbs = Float64[]
    ubs = Float64[]
    for (offset, k, n) in layouts
        m = _factor_chunk_size(n, rank)
        if rank === nothing
            append!(lbs, fill(-Inf, k))
            append!(lbs, fill(diag_floor, n))
        else
            append!(lbs, fill(-Inf, m))
        end
        append!(ubs, fill(Inf, m))
    end
    return lbs, ubs
end

function _wrap(f, link)
    return x -> f(link(x))
end

function Workspace(
    model::VecModel, optimizer::SDPQuadAlg, x0, args...; options::SDPQuadOptions, kwargs...
)
    return SDPQuadWorkspace(model, copy(x0), options, optimizer.sub_alg)
end

function optimize!(workspace::SDPQuadWorkspace)
    @unpack model, x0, options, sub_alg = workspace
    @unpack rank, diag_floor = options
    layouts = _check_layout(model)
    nvars_new = sum(l -> _factor_chunk_size(l[3], rank), layouts)
    link = _factor_link(layouts, rank, nvars_new)
    # build the factor-space sub-model: no semidefinite constraints
    obj = _wrap(model.objective, link)
    eq = VectorOfFunctions([
        EqConstraint(_wrap(c.f, link), c.rhs, c.dim, c.flags) for
        c in model.eq_constraints.fs
    ],)
    ineq = VectorOfFunctions([
        IneqConstraint(_wrap(c.f, link), c.rhs, c.dim, c.flags) for
        c in model.ineq_constraints.fs
    ],)
    lbs, ubs = _factor_bounds(layouts, rank, diag_floor)
    fx0 = _initial_factors(model, layouts, rank, diag_floor)
    submodel = VecModel(
        Objective(obj),
        eq,
        ineq,
        VectorOfFunctions(()),
        lbs,
        ubs,
        copy(fx0),
        falses(length(fx0)),
    )
    checksubproblem!(submodel, sub_alg; context="the re-parameterized subproblem")
    result_i = optimize(submodel, sub_alg, fx0; options=options.sub_options)
    minimizer = link(result_i.minimizer)
    obj = getobjective(model)
    obj === nothing && throw(ArgumentError("The model has no objective."))
    return SDPQuadResult(obj(minimizer), minimizer)
end
