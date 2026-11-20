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
