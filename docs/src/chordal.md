## Chordal decomposition utilities

A semidefinite constraint with a sparse (chordal) pattern can be decomposed
into several small PSD constraints on the maximal cliques of a chordal
extension of its pattern — an exact reformulation (Grone–Johnson–Sá–Wolkowicz).
`symmetric_pattern` symmetrizes a structural sparsity pattern and completes
it with the diagonal, `chordal_extension` returns the maximal cliques of the
extended graph, and `aggregate_sparsity` probes the structural dependence of
a matrix-valued function on the variables with reverse-mode AD.
`chordal_decompose` replaces each semidefinite constraint of a model by the
clique constraints, either from user-supplied patterns or by probing at `x0`.

```julia
using NonconvexSemidefinite, NonconvexCore, SparseArrays, LinearAlgebra

# A tridiagonal 4×4 constraint matrix and its sparsity pattern.
P0 = [2.0 0.5 0.0 0.0; 0.5 2.0 0.5 0.0; 0.0 0.5 2.0 0.5; 0.0 0.0 0.5 2.0]
pattern = sparse(P0 .!= 0)

symmetric_pattern(pattern)  # pattern ∪ pattern' ∪ diagonal
cliques = chordal_extension(pattern)  # maximal cliques of the extended graph
# AD-probed dependence pattern of x -> diagm(x) at zeros(3):
aggregate_sparsity(x -> diagm(x), zeros(3), 3)

model = Model(x -> sum(abs2, x .- 3.0))
addvar!(model, fill(-10.0, 4), fill(10.0, 4))
add_sd_constraint!(
    model,
    x -> [x[1] 0.5 0.0 0.0; 0.5 x[2] 0.5 0.0; 0.0 0.5 x[3] 0.5; 0.0 0.0 0.5 x[4]],
    4,
)
vecmodel, x0, _ = NonconvexCore.tovecmodel(model)
decomposed = chordal_decompose(vecmodel, pattern)
# one 4×4 PSD constraint became three 2×2 clique constraints:
length(decomposed.sd_constraints.fs)  # 3
```

### Function-in-set transforms

The function-in-set framework (NonconvexCore 2.3) has two model transforms
of its own:

- [`chordal_transform`](@ref) is the exact Fukuda conversion: each PSD
  component with a chordal sparsity pattern is rewritten as one small PSD
  variable block per maximal clique, coupled to the original matrix by
  linear equalities (`P = Σₖ E(Cₖ)ᵀYₖE(Cₖ)`, `Yₖ ⪰ 0`). The reformulation is
  equivalent to `P(x) ⪰ 0`.
- [`block_relaxation`](@ref) is the relaxation: each PSD component is
  rewritten as per-block [`PSDCone`](@ref) constraints on the principal
  submatrices `P(x)[C, C]` for user-supplied index blocks `C`. The blocks
  may come from the maximal cliques of a chordal extension or from the
  cells of a finite-element mesh (see below).

!!! warning "`block_relaxation` is a relaxation"
    `P(x)[C, C] ⪰ 0` for every block is only *implied* by `P(x) ⪰ 0`; the
    converse fails in general. The rewritten model's feasible set contains
    the original one, and for minimization its optimal value is a lower
    bound.

### Ferrite element blocks

With Ferrite loaded, `element_blocks(dh)` returns the per-cell degree of
freedom blocks of a finite-element mesh. Feeding them to
[`block_relaxation`](@ref) gives the element-based semidefinite relaxation
of an assembled constraint — one small PSD block per element, evaluated
from a single assembly:

```julia
using Ferrite, NonconvexSemidefinite

# ... a model whose set constraint bundles (g(x), K(x) + λ̄Kσ(x)) ...
dm = block_relaxation(vecmodel, element_blocks(dh))
# dm enforces (K(x) + λ̄Kσ(x))[C, C] ⪰ 0 on every element block C
```


### Chordal completion projection

[`chordal_projection`](@ref) completes a symmetric matrix to the closest
PSD matrix on a chordal pattern — the fixed-point iteration of
Grone–Johnson–Sá–Wolkowicz — with an implicit (fixed-point) adjoint:

```julia
using NonconvexSemidefinite, LinearAlgebra

cliques = [[1, 2], [2, 3]]             # maximal cliques of a tridiagonal graph
P = [2.0 1.0 0.0; 1.0 2.0 1.0; 0.0 1.0 2.0]
A = chordal_projection(P, cliques)     # PSD, and A ≈ P (P is already PSD)
minimum(eigvals(Symmetric(A)))         # ≥ 0

# reverse-mode AD through the projection (implicit adjoint):
using Zygote
grad = Zygote.gradient(P -> tr(chordal_projection(P, cliques)), P)[1]
```
