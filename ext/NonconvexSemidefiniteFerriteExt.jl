# Ferrite.jl glue for the structured domain decomposition (SDD) and the
# element-based relaxation: loading Ferrite enables `sdd_partition(grid, dh)`
# for a general Ferrite mesh (the cell centroids and the per-cell degrees of
# freedom are read and handed to the mesh-agnostic
# `NonconvexSemidefinite.sdd_partition(coords, cell_dofs)`) and
# `element_blocks(dh)` for [`block_relaxation`](@ref).

module NonconvexSemidefiniteFerriteExt

using NonconvexSemidefinite: NonconvexSemidefinite
import NonconvexSemidefinite: sdd_partition, element_blocks
using Ferrite: Ferrite

# Cell centroids of a Ferrite grid (any cell type, any dimension).
function _cell_centroids(grid)
    ncells = Ferrite.getncells(grid)
    ncells == 0 && return zeros(0, 0)
    dim = length(first(Ferrite.getcoordinates(grid, 1)))
    coords = Matrix{Float64}(undef, ncells, dim)
    for i in 1:ncells
        xs = Ferrite.getcoordinates(grid, i)
        coords[i, :] = collect(sum(xs) / length(xs))
    end
    return coords
end

"""
    sdd_partition(grid::Ferrite.AbstractGrid, dh::Ferrite.AbstractDofHandler;
                  n_subdomains = 2)

The structured-domain-decomposition partition (Kočvara's SDD) of a general
Ferrite mesh: the cell centroids of `grid` and the per-cell degrees of
freedom of `dh` are read and partitioned by recursive coordinate bisection
into `n_subdomains`; the dofs shared by several subdomains become the
interface, the others the interiors. See
[`sdd_partition(coords, cell_dofs)`](@ref sdd_partition) for the
mesh-agnostic core and [`sdd_transform`](@ref) for the transformation of a
large semidefinite constraint.
"""
function sdd_partition(
    grid::Ferrite.AbstractGrid, dh::Ferrite.AbstractDofHandler; n_subdomains::Int=2
)
    ncells = Ferrite.getncells(grid)
    coords = _cell_centroids(grid)
    cell_dofs = [collect(Ferrite.celldofs(dh, i)) for i in 1:ncells]
    return sdd_partition(coords, cell_dofs; n_subdomains=n_subdomains)
end

function element_blocks(dh::Ferrite.AbstractDofHandler)
    ncells = Ferrite.getncells(Ferrite.get_grid(dh))
    return [collect(Ferrite.celldofs(dh, i)) for i in 1:ncells]
end
function element_blocks(grid::Ferrite.AbstractGrid, dh::Ferrite.AbstractDofHandler)
    return element_blocks(dh)
end

end
