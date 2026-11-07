# Modeling conveniences for semidefinite variables.

"""
    PSDMatrixView(offset, dim)

A view of an `dim×dim` positive semidefinite matrix variable added to a model
with [`add_psd_variable!`](@ref). Calling `M(x)` decompresses the symmetric
matrix from `x` using [`decompress_symmetric`](@ref):

- if `length(x) == nvars(M)`, `x` is the variable chunk `[x_L; x_D]` itself;
- otherwise `x` is the full flattened variable vector and the chunk is read at
  the offset recorded when the variable was added.
"""
struct PSDMatrixView
    offset::Int
    dim::Int
    nvars::Int
end
function PSDMatrixView(offset::Int, dim::Int)
    return PSDMatrixView(offset, dim, dim * (dim - 1) ÷ 2 + dim)
end

Base.size(m::PSDMatrixView) = (m.dim, m.dim)
Base.size(m::PSDMatrixView, d::Integer) = d == 1 || d == 2 ? m.dim : 1

function (m::PSDMatrixView)(x::AbstractVector)
    k = m.dim * (m.dim - 1) ÷ 2
    chunk = if length(x) == m.nvars
        x
    elseif length(x) >= m.offset + m.nvars
        @view x[m.offset+1:m.offset+m.nvars]
    else
        throw(
            ArgumentError(
                "The variable vector has length $(length(x)) which does not match the " *
                "PSD variable chunk length $(m.nvars) nor the full flattened model " *
                "($(m.offset + m.nvars) variables).",
            ),
        )
    end
    x_L = @view chunk[1:k]
    x_D = @view chunk[k+1:k+m.dim]
    return decompress_symmetric(x_L, x_D)
end

function _psd_chunk(init::AbstractMatrix, dim::Int)
    size(init) == (dim, dim) || throw(
        ArgumentError(
            "The initial PSD matrix must be $(dim)×$(dim), got $(size(init)).",
        ),
    )
    k = dim * (dim - 1) ÷ 2
    return vcat(init[lowertriangind(init)], diag(init))
end

"""
    add_psd_variable!(model, dim; init = I(dim))

Add the variables of a symmetric `dim×dim` positive semidefinite matrix to the
model as one chunk in the `(x_L, x_D)` layout (the strict lower-triangular
entries followed by the diagonal), with the diagonal constrained to be
non-negative. Returns a [`PSDMatrixView`](@ref) that decompresses the matrix
from a variable vector.

```julia
model = Model()
M = add_psd_variable!(model, 3)
model.objective = x -> -loglikelihood(MvNormal(μ, M(x[1])), samples)
```
"""
function add_psd_variable!(
    model::Model,
    dim::Int;
    init::AbstractMatrix = Matrix{Float64}(I, dim, dim),
)
    offset = length(NonconvexCore.flatten(getmin(model))[1])
    chunk = _psd_chunk(init, dim)
    k = dim * (dim - 1) ÷ 2
    addvar!(
        model,
        vcat(fill(-Inf, k), zeros(dim)),
        fill(Inf, length(chunk));
        init = chunk,
    )
    return PSDMatrixView(offset, dim)
end

function add_psd_variable!(
    model::VecModel,
    dim::Int;
    init::AbstractMatrix = Matrix{Float64}(I, dim, dim),
)
    offset = length(getmin(model))
    chunk = _psd_chunk(init, dim)
    k = dim * (dim - 1) ÷ 2
    addvar!(
        model,
        vcat(fill(-Inf, k), zeros(dim)),
        fill(Inf, length(chunk));
        init = chunk,
    )
    return PSDMatrixView(offset, dim)
end

function add_psd_variable!(
    model::DictModel,
    name::Union{Symbol,String},
    dim::Int;
    init::AbstractMatrix = Matrix{Float64}(I, dim, dim),
)
    offset = sum(v -> length(NonconvexCore.flatten(v)[1]), values(getmin(model)))
    chunk = _psd_chunk(init, dim)
    k = dim * (dim - 1) ÷ 2
    addvar!(
        model,
        name,
        vcat(fill(-Inf, k), zeros(dim)),
        fill(Inf, length(chunk));
        init = chunk,
    )
    return PSDMatrixView(offset, dim)
end
