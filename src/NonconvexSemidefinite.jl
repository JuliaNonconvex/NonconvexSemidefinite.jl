module NonconvexSemidefinite

export SDPBarrierOptions, SDPBarrierAlg, decompress_symmetric
export BarrierAlg, BarrierOptions, BarrierSelection, BarrierResult
export SDPQuadAlg, SDPQuadOptions, SDPQuadResult
export project_psd, project_psd_neg, psd_infeasibility
export symmetric_pattern, chordal_extension, aggregate_sparsity, chordal_decompose
export PSDMatrixView, add_psd_variable!

using ChainRulesCore, Reexport, Parameters
@reexport using NonconvexCore
using NonconvexCore: VecModel, AbstractOptimizer, AbstractModel
using NonconvexCore:
    VectorOfFunctions, SDConstraint, AbstractConstraint, Objective, EqConstraint, IneqConstraint
import NonconvexCore: Workspace, optimize!, _optimize_precheck

# Semidefinite programming

# Decompress
function lowertriangind(mat::AbstractMatrix)
    indices = [i for i in CartesianIndices(mat) if i[1] > i[2]]
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
    return Symmetric(L + D + L')
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

function ChainRulesCore.rrule(
    ::typeof(rearrange_x), x_L::AbstractVector, x_D::AbstractVector
)
    function pullback((ΔL, ΔD))
        if ΔL isa ChainRulesCore.AbstractZero || ΔD isa ChainRulesCore.AbstractZero
            return NoTangent(), ZeroTangent(), ZeroTangent()
        end
        Δx_L = ΔL[lowertriangind(ΔL)]
        Δx_D = diag(ΔD)
        return NoTangent(), Δx_L, Δx_D
    end
    return rearrange_x(x_L, x_D), pullback
end

include("projection.jl")
include("chordal.jl")
include("modeling.jl")

include("barrier.jl")
include("reparam.jl")

end
