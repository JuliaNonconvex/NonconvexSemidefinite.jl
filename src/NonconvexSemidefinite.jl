module NonconvexSemidefinite

export SDPBarrierOptions, SDPBarrierAlg, decompress_symmetric

using ChainRulesCore, Reexport, Parameters
@reexport using NonconvexCore
using NonconvexCore: VecModel, AbstractOptimizer, AbstractModel, set_objective
import NonconvexCore: Workspace, optimize!, _optimize_precheck

# Semidefinite programming

# Decompress
function lowertriangind(mat::Matrix)
    indices = [i for i in CartesianIndices(mat) if i[1]>i[2]]
    return LinearIndices(mat)[indices]
end

function rearrange_x(x_L::AbstractVector, x_D::AbstractVector)
    mat_dim = length(x_D)
    L = zeros(mat_dim, mat_dim)
    L[lowertriangind(L)] .= x_L
    D = diagm(x_D)
    return (L, D)
end

function decompress_symmetric(L::Matrix, D::Matrix)
    Symmetric(L + D + L')
end

"""
    decompress_symmetric

For example: 
    a 3*3 positive semidefinite matrix: 
    `[  a       b       d; 
        b       c       e; 
        d       e       f       ]` 
    represents by: 
    `[  x_D[1]  x_L[3]  x_L[2]; 
        x_L[1]  x_D[2]  x_L[1]; 
        x_L[2]  x_L[3]  x_D[3]  ]`
- `x_L::AbstractArray`: representing lower triangular part of a `n*n` matrix, length should be `(n^2-n)÷2`
- `x_D::AbstractArray`: representing diagonal part of a `n*n` matrix, length should be `n`
"""
function decompress_symmetric(x_L::AbstractArray, x_D::AbstractArray)
    L, D = rearrange_x(x_L, x_D)
    return decompress_symmetric(L, D)
end

function ChainRulesCore.rrule(::typeof(rearrange_x), x_L::AbstractVector, x_D::AbstractVector)
    function pullback((ΔL, ΔD))
        Δx_L = ΔL[lowertriangind(ΔL)]
        Δx_D = diag(ΔD)
        NoTangent(), Δx_L, Δx_D 
    end
    return rearrange_x(x_L, x_D), pullback
end

"""
    SDPBarrierAlg(sub_alg)

A meta-algorithm that handles semidefinite constraints on nonlinear functions using a barrier approach. The coefficient of the barrier term is exponentially decayed and the sub-problems are solved using `sub_alg`. `sub_alg` can be any other compatible solver from `Nonconvex.jl`. The solver must be able to solve the sub-problem after removing the semidefinite constraints. The options to the solver should be pased to the [`SDPBarrierOptions`](@ref) struct and passed in as the options to the `optimize` function. Call `? SDPBarrierOptions` to check all the different options that can be set.
"""
struct SDPBarrierAlg{Alg <: AbstractOptimizer} <: AbstractOptimizer
    sub_alg::Alg
end
function SDPBarrierAlg(;sub_alg)
    return SDPBarrierAlg(sub_alg)
end

function _optimize_precheck(model::AbstractModel, optimizer::SDPBarrierAlg, args...; kwargs...)
    nothing
end


"""
    SDPBarrierOptions(; kwargs...)

The keyword arguments which can be specified are:
- `c_init`: (default 1.0) initial value for the coefficient `c` that is multiplied by the barrier term, could be a real number or vector in the case of multiple semidefinite constraints.
- `c_decr`: (default 0.1) decreasing rate (< 1) that multiplies the barrier term in every iteration, could be either a real number or a vector in the case of multiple semidefinite constraints.
- `n_iter`: (default 20) number of sub-problems to solve in the barrier method.
- `sub_options`: options for the sub-problem's solver
- `keep_all`: (default `falue`) if set to `true`, `SDPBarrierResult` stores the results from all the iterations
"""
mutable struct SDPBarrierOptions{C1 <: Union{Real, AbstractArray}, C2 <: Union{Real, AbstractArray}}
    # Dimension of objective matrix
    # Hyperparameters 
    # Initial value of `c` in barrier method: 
    # `Real` for using same value for all `sd_constraints`, `AbstractArray` for assign them respectively
    c_init::C1
    # Decrease rate of `c` for every epoch, same as above
    c_decr::C2
    n_iter::Int
    # sub_option to solve (in)equality constraints
    sub_options
    # Keep all results or not
    keep_all::Bool
end
function SDPBarrierOptions(c_init, c_decr, n_iter; sub_options, keep_all=false)
    @assert all(0 .< c_decr .< 1) "c_decr should be between 0 and 1. "
    @assert all(c_init .> 0) "c_init shoule be larger than 0. "
    SDPBarrierOptions(c_init, c_decr, n_iter, sub_options, keep_all)
end
function SDPBarrierOptions(;sub_options, c_init=1.0, c_decr=0.1, n_iter=20, keep_all=false)
    SDPBarrierOptions(c_init, c_decr, n_iter, sub_options=sub_options, keep_all=keep_all)
end

# Result
struct SDPBarrierResult{M1, M2, R, O}
    minimum::M1
    minimizer::M2
    results::R
    optimal_ind::O
end

struct SDPBarrierWorkspace{M <: VecModel, X <: AbstractVector, O <: SDPBarrierOptions, S <: AbstractOptimizer} <: Workspace
    model::M
    x0::X
    options::O
    sub_alg::S
end

function Workspace(model::VecModel, optimizer::SDPBarrierAlg, x0, args...; options, kwargs...,)
    @unpack c_init, c_decr = options
    for c in model.sd_constraints.fs
        @assert isposdef(c(x0)) "Initial matrix should be positive definite. "
    end
    if c_init isa AbstractArray
        @assert length(model.sd_constraints.fs) == length(c_init) "c_init should be same length with number of `sd_constraints` when using array. "
    end
    if c_decr isa AbstractArray
        @assert length(model.sd_constraints.fs) == length(c_decr) "c_decr should be same length with number of `sd_constraints` when using array. "
    end
    if c_init isa AbstractArray && c_decr isa AbstractArray
        @assert length(c_init) == length(c_decr) "c_decr should be same length with c_init. "
    end
    return SDPBarrierWorkspace(model, copy(x0), options, optimizer.sub_alg)
end

function safe_logdet(A::AbstractMatrix)
    c = cholesky(A, check = false)
    if issuccess(c)
        return logdet(c)
    else
        return -Inf
    end
end

function sd_objective(objective0, sd_constraints, c::AbstractArray)
    function _objective(args)
        target = objective0(args)
        barrier = sum(c .* -safe_logdet.(map(f -> f(args), sd_constraints.fs)))
        return target + barrier
    end
    return _objective
end

function to_barrier(model, c::AbstractArray)
    sd_constraints, objective0 = model.sd_constraints, model.objective
    _model = set_objective(model, sd_objective(objective0, sd_constraints, c))
    return _model
end

function optimize!(workspace::SDPBarrierWorkspace)
    @unpack model, x0, options, sub_alg = workspace
    @unpack c_init, c_decr, n_iter, sub_options, keep_all = options
    objective0 = model.objective
    x = copy(x0)
    c = c_init isa Real ? ([c_init for _ in 1:length(model.sd_constraints.fs)]) : c_init
    results = []
    for _ in 1:n_iter
        model_i = to_barrier(model, c)
        result_i = optimize(model_i, sub_alg, x, options = sub_options)
        minimizer_i = result_i.minimizer
        @info NonconvexCore.getobjective(model_i)(result_i.minimizer)
        push!(results, (objective0(minimizer_i), minimizer_i))
        c = c .* c_decr
        x = copy(minimizer_i)
    end
    optimal_ind = argmin(first.(results))
    minimum, minimizer = results[optimal_ind]
    if keep_all
        return SDPBarrierResult(minimum, minimizer, results, optimal_ind)
    else
        return SDPBarrierResult(minimum, minimizer, nothing, nothing)
    end
end

end
