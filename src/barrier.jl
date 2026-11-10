# Generalized primal barrier meta-algorithm.
#
# `BarrierAlg` converts inequality, bound and semidefinite constraints into
# log-barrier terms of the subproblem objective, decays the barrier coefficient
# across a sequence of subproblems and solves each one with `sub_alg`.
# Constraints that are not selected for barrierification are kept as hard
# constraints of the subproblem. `SDPBarrierAlg` is kept as an alias.

"""
    BarrierSelection(; ineq = false, box = false, sd = true)

Selects which constraint kinds are converted to barrier terms by
[`BarrierAlg`](@ref). Each entry is `true` (all constraints of that kind),
`false` (none) or a vector of constraint indices (those only). Equality
constraints can never be barrierified and are always kept for the sub-solver.

The default selects semidefinite constraints only, preserving the historical
`SDPBarrierAlg` behaviour; pass `BarrierSelection(ineq = true, box = true)`
to barrierify inequalities and variable bounds as well.
"""
struct BarrierSelection
    ineq::Union{Bool,AbstractVector{Int}}
    box::Bool
    sd::Union{Bool,AbstractVector{Int}}
end
function BarrierSelection(; ineq=false, box=false, sd=true)
    return BarrierSelection(ineq, box, sd)
end

_selected(sel::Bool, n::Int) = sel ? collect(1:n) : Int[]
_selected(sel::AbstractVector{Int}, n::Int) = sel
function _selected(sel, n::Int)
    return throw(
        ArgumentError(
            "Barrier selection entries must be `true`, `false` or a vector of constraint indices; got $(typeof(sel)).",
        ),
    )
end

"""
    BarrierAlg(sub_alg)

A meta-algorithm that handles inequality, bound and semidefinite constraints
using a primal log-barrier approach. The barrier coefficient is decayed across
a sequence of sub-problems (see [`BarrierOptions`](@ref)) and each sub-problem
is solved with `sub_alg`, which must handle the constraints that are not
barrierified. `SDPBarrierAlg` is an alias.
"""
struct BarrierAlg{Alg<:AbstractOptimizer} <: AbstractOptimizer
    sub_alg::Alg
end
function BarrierAlg(; sub_alg)
    return BarrierAlg(sub_alg)
end

"""
    SDPBarrierAlg(sub_alg)

Alias of [`BarrierAlg`](@ref). Historically the barrier meta-algorithm only
handled semidefinite constraints; the generalized [`BarrierAlg`](@ref) now
also barrierifies inequalities and variable bounds, selected through
[`BarrierSelection`](@ref). Both names construct the same meta-algorithm.
"""
const SDPBarrierAlg = BarrierAlg

# The barrier meta-algorithm can barrierify ineq/box/sd constraints itself;
# equality constraints are forwarded to the sub-solver.
function NonconvexCore.solver_capabilities(alg::BarrierAlg)
    sub = solver_capabilities(alg.sub_alg)
    return merge_capabilities(
        sub;
        linear_ineq=true,
        nonlinear_ineq=true,
        linear_sd=true,
        nonlinear_sd=true,
        box=true,
    )
end

function _optimize_precheck(model::AbstractModel, optimizer::BarrierAlg, args...; kwargs...)
    return nothing
end

"""
    BarrierOptions(; kwargs...)

The keyword arguments which can be specified are:
- `c_init`: (default 1.0) initial value for the barrier coefficient, a real
  number or a vector with one entry per barrierified constraint group.
- `c_decr`: (default 0.1) decreasing rate (< 1) multiplying the barrier
  coefficient every iteration, a real number or a vector.
- `n_iter`: (default 20) maximum number of sub-problems to solve.
- `sub_options`: options for the sub-problem's solver.
- `keep_all`: (default `false`) if set to `true`, the result stores the
  sub-problem results from all the iterations.
- `barrier`: a [`BarrierSelection`](@ref) (default: barrierify the
  semidefinite constraints only).
- `adaptive`: (default `false`) decay the barrier coefficient with a
  superlinear schedule targeting `tol` within `n_iter` iterations instead of
  the fixed `c_decr` factor.
- `tol`: (default `nothing`) stop early when the estimated duality gap
  `c ⋅ m` (barrier coefficient times the number of barrier terms) falls below
  this value.
"""
mutable struct BarrierOptions{
    C1<:Union{Real,AbstractArray},C2<:Union{Real,AbstractArray},SO
}
    c_init::C1
    c_decr::C2
    n_iter::Int
    sub_options::SO
    keep_all::Bool
    barrier::BarrierSelection
    adaptive::Bool
    tol::Union{Nothing,Real}
end
function BarrierOptions(c_init, c_decr, n_iter; sub_options, keep_all=false)
    @assert all(0 .< c_decr .< 1) "c_decr should be between 0 and 1. "
    @assert all(c_init .> 0) "c_init should be larger than 0. "
    return BarrierOptions(
        c_init, c_decr, n_iter, sub_options, keep_all, BarrierSelection(), false, nothing
    )
end
function BarrierOptions(;
    sub_options,
    c_init=1.0,
    c_decr=0.1,
    n_iter=20,
    keep_all=false,
    barrier=BarrierSelection(),
    adaptive=false,
    tol=nothing,
)
    @assert all(0 .< c_decr .< 1) "c_decr should be between 0 and 1. "
    @assert all(c_init .> 0) "c_init should be larger than 0. "
    return BarrierOptions(
        c_init, c_decr, n_iter, sub_options, keep_all, barrier, adaptive, tol
    )
end

"""
    SDPBarrierOptions(; kwargs...)

Alias of [`BarrierOptions`](@ref) (all keyword arguments are shared).
"""
const SDPBarrierOptions = BarrierOptions

# Result

"""
    SDPBarrierResult

The result of [`BarrierAlg`](@ref) (or its alias [`SDPBarrierAlg`](@ref))
with fields:

- `minimum`: the best objective value over the sub-problem sequence.
- `minimizer`: the minimizer attaining `minimum`.
- `results`: if `keep_all = true` in [`BarrierOptions`](@ref), the full
  sub-problem trajectory as a vector of `(objective, minimizer)` pairs;
  `nothing` otherwise.
- `optimal_ind`: if `keep_all = true`, the index of the best sub-problem
  result in `results`; `nothing` otherwise.
"""
struct SDPBarrierResult{M1,M2,R,O}
    minimum::M1
    minimizer::M2
    results::R
    optimal_ind::O
end

"""
    BarrierResult

Alias of [`SDPBarrierResult`](@ref).
"""
const BarrierResult = SDPBarrierResult

struct BarrierWorkspace{
    M<:VecModel,X<:AbstractVector,O<:BarrierOptions,S<:AbstractOptimizer
} <: Workspace
    model::M
    x0::X
    options::O
    sub_alg::S
end
const SDPBarrierWorkspace = BarrierWorkspace

function Workspace(model::VecModel, optimizer::BarrierAlg, x0, args...; options, kwargs...)
    @unpack c_init, c_decr, barrier = options
    # Strict feasibility of the barrierified constraints at the initial point.
    for i in _selected(barrier.ineq, length(model.ineq_constraints.fs))
        vals = model.ineq_constraints.fs[i](x0)
        if !all(vals .< 0)
            worst = maximum(vals)
            throw(
                ArgumentError(
                    "The initial point must strictly satisfy the barrierified " *
                    "inequality constraint $i (values ≤ 0 required; the worst " *
                    "value is $worst).",
                ),
            )
        end
    end
    if barrier.box
        for j in eachindex(x0)
            lo, hi = model.box_min[j], model.box_max[j]
            if isfinite(lo) && !(x0[j] > lo)
                throw(
                    ArgumentError(
                        "The initial point must strictly satisfy the lower " *
                        "bound on variable $j (x0[$j] = $(x0[j]) must be > $lo).",
                    ),
                )
            end
            if isfinite(hi) && !(x0[j] < hi)
                throw(
                    ArgumentError(
                        "The initial point must strictly satisfy the upper " *
                        "bound on variable $j (x0[$j] = $(x0[j]) must be < $hi).",
                    ),
                )
            end
        end
    end
    for i in _selected(barrier.sd, length(model.sd_constraints.fs))
        if !isposdef(model.sd_constraints.fs[i](x0))
            throw(
                ArgumentError(
                    "The initial point must be strictly interior to the " *
                    "barrierified semidefinite constraint $i (the matrix must " *
                    "be positive definite).",
                ),
            )
        end
    end
    nunits =
        length(_selected(barrier.ineq, length(model.ineq_constraints.fs))) +
        (barrier.box ? 1 : 0) +
        length(_selected(barrier.sd, length(model.sd_constraints.fs)))
    for (name, val) in (("c_init", c_init), ("c_decr", c_decr))
        if val isa AbstractArray
            @assert length(val) == nunits "$name should have one entry per barrierified constraint group ($nunits), got $(length(val))."
        end
    end
    return BarrierWorkspace(model, copy(x0), options, optimizer.sub_alg)
end

function safe_logdet(A::AbstractMatrix)
    c = cholesky(A; check=false)
    if issuccess(c)
        return logdet(c)
    else
        return -Inf
    end
end

# C² smooth extension of the log barrier -log(-t) beyond t = -ε, so that
# sub-solver line searches can probe points outside the strict interior.
function _log_barrier(t::Real, ε::Real)
    t < -ε && return -log(-t)
    return -log(ε) + (t + ε) / ε + (t + ε)^2 / (2ε^2)
end

function _barrier_units(model, barrier)
    ineq_sel = _selected(barrier.ineq, length(model.ineq_constraints.fs))
    sd_sel = _selected(barrier.sd, length(model.sd_constraints.fs))
    # number of barrier terms per unit (for the duality gap estimate)
    ms = Int[]
    for i in ineq_sel
        push!(ms, model.ineq_constraints.fs[i].dim)
    end
    if barrier.box
        nfinite = 0
        for j in eachindex(model.box_min)
            nfinite += isfinite(model.box_min[j]) ? 1 : 0
            nfinite += isfinite(model.box_max[j]) ? 1 : 0
        end
        push!(ms, nfinite)
    end
    for i in sd_sel
        push!(ms, model.sd_constraints.fs[i].dim)
    end
    return ineq_sel, sd_sel, ms
end

function _barrier_objective(objective0, model, c::Tuple, barrier, ε)
    ineq_sel, sd_sel, _ = _barrier_units(model, barrier)
    function _objective(x)
        target = objective0(x)
        u = 1
        for i in ineq_sel
            for v in model.ineq_constraints.fs[i](x)
                target += c[u] * _log_barrier(v, ε)
            end
            u += 1
        end
        if barrier.box
            for j in eachindex(x)
                lo, hi = model.box_min[j], model.box_max[j]
                isfinite(lo) && (target += c[u] * _log_barrier(lo - x[j], ε))
                isfinite(hi) && (target += c[u] * _log_barrier(x[j] - hi, ε))
            end
            u += 1
        end
        for i in sd_sel
            target += c[u] * -safe_logdet(model.sd_constraints.fs[i](x))
            u += 1
        end
        return target
    end
    return _objective
end

"""
    to_barrier(model, c; selection = BarrierSelection(ineq = false, box = false, sd = true), ε = 1e-6)

Build the sub-model of the barrier method: the selected constraints are
removed and encoded in the log-barrier objective (with coefficient vector
`c`, one entry per selected constraint group). The default selection matches
the historical `SDPBarrierAlg` behaviour (semidefinite constraints only).
"""
function to_barrier(
    model::VecModel,
    c::Tuple;
    selection=BarrierSelection(; ineq=false, box=false, sd=true),
    ε=1e-6,
)
    ineq_sel, sd_sel, _ = _barrier_units(model, selection)
    kept_ineq = VectorOfFunctions(_kept(model.ineq_constraints.fs, ineq_sel))
    kept_sd = VectorOfFunctions(_kept(model.sd_constraints.fs, sd_sel))
    return VecModel(
        Objective(_barrier_objective(model.objective, model, c, selection, ε)),
        model.eq_constraints,
        kept_ineq,
        kept_sd,
        model.box_min,
        model.box_max,
        model.init,
        model.integer,
    )
end

# Constraints not selected for barrierification, collected with the concrete
# element type of `fs`. A filtered comprehension over an all-rejected
# `Vector{SDConstraint}` widens to `Vector{Any}`, which `VectorOfFunctions`
# rejects; this helper keeps the concrete element type in every case.
function _kept(fs, sel::AbstractVector{Int})
    kept = eltype(fs)[]
    for (i, cons) in enumerate(fs)
        i in sel || push!(kept, cons)
    end
    return kept
end

function _next_c(c, ms, tol, k, n_iter, c_decr, adaptive)
    if !adaptive
        return c .* c_decr
    end
    gap = dot(c, ms)
    remaining = max(1, n_iter - k)
    target = max(tol / maximum(ms), eps(Float64))
    ratio = max(minimum(c_decr), (target / maximum(c))^(1 / remaining))
    return c .* ratio
end

function optimize!(workspace::BarrierWorkspace)
    @unpack model, x0, options, sub_alg = workspace
    @unpack c_init, c_decr, n_iter, sub_options, keep_all = options
    @unpack barrier, adaptive, tol = options
    _c_decr = if c_decr isa Real
        ntuple(_ -> c_decr, length(_barrier_units(model, barrier)[3]))
    else
        Tuple(c_decr)
    end
    ineq_sel, sd_sel, ms = _barrier_units(model, barrier)
    objective0 = getobjective(model)
    objective0 === nothing && throw(ArgumentError("The model has no objective."))
    x = copy(x0)
    obj0 = objective0(x)
    c = c_init isa Real ? ntuple(_ -> c_init, length(ms)) : Tuple(c_init)
    results = Tuple{typeof(obj0),typeof(x)}[]
    for k in 1:n_iter
        gap = dot(c, ms)
        tol !== nothing && gap <= tol && break
        # The smooth-extension region must shrink with the barrier
        # coefficient, otherwise infeasible points leak through at small c.
        ε = clamp(Base.minimum(c) / 10, 1e-9, 1e-4)
        model_i = to_barrier(model, c; selection=barrier, ε=ε)
        result_i = optimize(model_i, sub_alg, x; options=sub_options)
        minimizer_i = result_i.minimizer
        push!(results, (objective0(minimizer_i), minimizer_i))
        c = _next_c(c, ms, tol === nothing ? 0.0 : tol, k, n_iter, _c_decr, adaptive)
        x = copy(minimizer_i)
    end
    optimal_ind = argmin(eachindex(results)) do i
        v = first(results[i])
        return isfinite(v) ? v : Inf
    end
    minimum, minimizer = results[optimal_ind]
    if keep_all
        return SDPBarrierResult(minimum, minimizer, results, optimal_ind)
    else
        return SDPBarrierResult(minimum, minimizer, nothing, nothing)
    end
end
