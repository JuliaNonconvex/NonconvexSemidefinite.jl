module NonconvexSemidefinite

export SDPBarrierOptions, SDPBarrierAlg, decompress_symmetric
export BarrierAlg, BarrierOptions, BarrierSelection, BarrierResult
export SDPQuadAlg, SDPQuadOptions, SDPQuadResult
export project_psd,
    project_psd_neg,
    psd_infeasibility,
    project_psd_batch,
    psd_neg_eig,
    psd_infeasibility_batch,
    partition_blocks
export symmetric_pattern, chordal_extension, aggregate_sparsity, chordal_decompose
export AbstractChordalAlgorithm, AMDAlg, MCSAlg, MFAlg, METISAlg, BTAlg
export is_chordal, chordal_transform, assemble_chordal
export PSDCone, has_psd_component
export block_relaxation
export chordal_projection
export element_blocks
export SDDPartition, sdd_partition, sdd_transform, nsubdomains
export ArrowSubdomain, ArrowIntersection, arrow_decompose
export SDCuttingPlaneAlg, SDCuttingPlaneOptions, SDCuttingPlaneResult
export PSDMatrixView, add_psd_variable!
export ks_aggregate
export PSDBarrierOracle,
    psd_barrier_value, psd_barrier_grad!, psd_hess_dense!, psd_hess_prod!
export logdet_estimate,
    logdet_estimate_of,
    slq_logdet,
    chebyshev_logdet,
    hutchpp_logdet,
    rational_logdet,
    taylor_logdet,
    trace_integration_logdet,
    lowrank_logdet,
    lanczos_bounds

using ChainRulesCore, Reexport, Parameters
import IterativeSolvers: lobpcg
@reexport using NonconvexCore
using NonconvexCore: VecModel, AbstractOptimizer, AbstractModel
using NonconvexCore:
    VectorOfFunctions,
    SDConstraint,
    AbstractConstraint,
    Objective,
    EqConstraint,
    IneqConstraint
import NonconvexCore: Workspace, optimize!, _optimize_precheck

# Semidefinite programming

# Decompress
function lowertriangind(mat::AbstractMatrix)
    indices = [i for i in CartesianIndices(mat) if i[1] > i[2]]
    return LinearIndices(mat)[indices]
end

function rearrange_x(x_L::AbstractVector, x_D::AbstractVector)
    mat_dim = length(x_D)
    # Preserve the element type of the inputs (e.g. ForwardDiff.Dual) instead
    # of forcing Float64, which strips or errors on AD dual numbers.
    # `Matrix{T}` (rather than `zeros(T, ...)`) keeps the construction
    # unambiguous for static analysis even when T is abstract.
    T = promote_type(eltype(x_L), eltype(x_D))
    L = Matrix{T}(undef, mat_dim, mat_dim)
    fill!(L, 0)
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
    `[  x_D[1]  x_L[1]  x_L[2]; 
        x_L[1]  x_D[2]  x_L[3]; 
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
        # Handle each cotangent independently: a zero cotangent in one output
        # must not discard the nonzero cotangent of the other output.
        Δx_L = ΔL isa ChainRulesCore.AbstractZero ? ZeroTangent() : ΔL[lowertriangind(ΔL)]
        Δx_D = ΔD isa ChainRulesCore.AbstractZero ? ZeroTangent() : diag(ΔD)
        return NoTangent(), Δx_L, Δx_D
    end
    return rearrange_x(x_L, x_D), pullback
end

include("projection.jl")
include("ks.jl")
include("chordal.jl")
include("completion.jl")
include("modeling.jl")

include("barrier.jl")
include("logdet_estimation.jl")
include("ipm_oracles.jl")
include("reparam.jl")
include("cutting_plane.jl")
include("sdd.jl")

end
