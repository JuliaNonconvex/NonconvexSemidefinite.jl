# Interior point meta-algorithm for handling nonlinear semidefinite constraints

![JuliaNonconvex logo](https://raw.githubusercontent.com/JuliaNonconvex/NonconvexSemidefinite.jl/master/docs/src/assets/logo.png)

## Description

If you need to keep your any matrix-valued function of the decision variables positive semidefinite, Nonconvex supports an interface for the [barrier method for semidefinite programming](http://eaton.math.rpi.edu/faculty/Mitchell/courses/matp6640/notes/24A_SDPbarrierbeamer.pdf), which is a meta-algorithm transforming the optimization problem to a series of nonlinear programming problems and solving them using the pre-specified `sub_alg` and `sub_options`.

## Quick start

Optimizing over a multivariate gaussian distribution with artificial samples using `Ipopt`:

```julia
using Distributionsusing NonconvexSemidefinite, NonconvexIpopt

# Draw random multivariate gaussian samples
# Random groundtruth
mat_dim = 3
μ = randn(mat_dim)
Σ = rand(mat_dim, mat_dim)
Σ = Σ + Σ' + 2I
# Generate
n_sample = 1000
samples = rand(MvNormal(μ, Σ), n_sample)

# Define objective function
function f((x_L, x_D))
    return -loglikelihood(MvNormal(μ, decompress_symmetric(x_L, x_D)), samples)
end
# Define the matrix-valued function
function sd_constraint((x_L, x_D))
    return decompress_symmetric(x_L, x_D)
end

# Define settings
model = Model(f)
mat_x0 = rand(mat_dim, mat_dim)
mat_x0 = mat_x0 + mat_x0' + I

x0 = [mat_x0[NonconvexSemidefinite.lowertriangind(mat_x0)], diag(mat_x0)]
lbs = [fill(-Inf, length(x0[1])), zeros(length(x0[2]))]
ubs = [fill(Inf, length(x0[1])), fill(Inf, length(x0[2]))]
addvar!(model, lbs, ubs)
add_sd_constraint!(model, sd_constraint)
alg = SDPBarrierAlg(sub_alg=IpoptAlg())
options = SDPBarrierOptions(sub_options=IpoptOptions(max_iter=200), n_iter = 20)

# Optimize
result = optimize(model, alg, x0, options = options)

# Relative error norm
norm(sd_constraint(result.minimizer) - Σ) / norm(Σ)
```

## Quick start: quadratic re-parameterization

Instead of the barrier method, the semidefinite constraint can be eliminated
entirely by substituting a factorization of the matrix variable: `SDPQuadAlg`
re-parameterizes `A = L Lᵀ` (full-rank Cholesky with `diag(L) ≥ diag_floor`)
or `A = R Rᵀ` (Burer–Monteiro with `R ∈ R^{n×r}` when `rank = r` is set) and
solves the resulting smooth, box-constrained subproblem with `sub_alg`. The
model variables must consist exactly of the semidefinite matrix chunks in the
`(x_L, x_D)` layout.

```julia
using NonconvexSemidefinite, NonconvexIpopt

# The decision variables are the (x_L, x_D) chunk of a 3×3 PSD matrix:
# the 3 strict lower-triangular entries followed by the 3 diagonal entries.
mat_dim = 3
k = mat_dim * (mat_dim - 1) ÷ 2
function objective(x)
    A = decompress_symmetric(@view(x[1:k]), @view(x[(k + 1):end]))
    return (A[1, 1] - 3.0)^2 + (A[2, 2] - 4.0)^2 + (A[3, 3] - 5.0)^2
end
function sd_constraint(x)
    return decompress_symmetric(@view(x[1:k]), @view(x[(k + 1):end]))
end

model = Model(objective)
x0 = [0.0, 0.0, 0.0, 3.0, 4.0, 5.0]
addvar!(model, vcat(fill(-Inf, k), zeros(mat_dim)), fill(Inf, k + mat_dim); init=x0)
add_sd_constraint!(model, sd_constraint)
alg = SDPQuadAlg(sub_alg=IpoptAlg())
options = SDPQuadOptions(sub_options=IpoptOptions(max_iter=200))

# Optimize
result = optimize(model, alg, x0, options=options)
A = sd_constraint(result.minimizer)  # ≈ diagm([3.0, 4.0, 5.0])
```

The full-rank re-parameterization is surjective: every PSD matrix has a
Cholesky factor, so no solution is lost. To search only the rank-`r` matrices
instead, pass `rank=r` to `SDPQuadOptions` (`r ≤ mat_dim`); see the theory
page for the trade-offs of the Burer–Monteiro factor.

## The general barrier method

`BarrierAlg` turns a problem with inequality, bound and semidefinite
constraints into a sequence of subproblems whose objective is augmented with
log-barrier terms. In each iteration the barrier coefficient decays and the
subproblem is solved with `sub_alg` (e.g. `IpoptAlg()`); constraints that are
not barrierified are passed to the sub-solver as hard constraints. Equality
constraints are never barrierified.

Which constraints are barrierified is controlled by `BarrierSelection`: each
of `ineq`, `box` and `sd` can be `false` (keep those constraints hard),
`true` (barrierify all of them) or a vector of constraint indices
(barrierify those only). The default
`BarrierSelection(ineq = false, box = false, sd = true)` barrierifies the
semidefinite constraints only, which reproduces the historical
`SDPBarrierAlg` behaviour.

The schedule is controlled by `BarrierOptions`: `c_init` is the initial
barrier coefficient, `c_decr` the decay factor (`c ← c_decr · c`, fixed
schedule) and `n_iter` the maximum number of subproblems. With
`adaptive = true` the coefficient instead follows a superlinear schedule
targeting the requested duality gap `tol` within the remaining iterations,
and the loop terminates early once the gap `c ⋅ m` (coefficient times the
number of barrier terms) falls below `tol`. `c_init` and `c_decr` accept
scalars or vectors with one entry per barrierified constraint group (ordered
inequalities, box, semidefinite). With `keep_all = true` the result stores
the full subproblem trajectory.

### Barrierifying inequalities and bounds (non-SDP problems)

`BarrierAlg` is not limited to semidefinite constraints — it can barrierify
plain inequality constraints and variable bounds as well:

```julia
using NonconvexSemidefinite, NonconvexIpopt

# min (x₁ - 1)² + (x₂ - 1)²  s.t.  2 - x₁ - x₂ ≤ 0  (optimum (1, 1), value 2)
model = Model(x -> (x[1] - 1.0)^2 + (x[2] - 1.0)^2)
addvar!(model, [-5.0, -5.0], [5.0, 5.0])
add_ineq_constraint!(model, x -> 2 - x[1] - x[2])

alg = BarrierAlg(sub_alg=IpoptAlg())
options = BarrierOptions(
    sub_options=IpoptOptions(max_iter=100),
    barrier=BarrierSelection(ineq=true, box=false, sd=false),
    c_init=1.0,
    c_decr=0.1,
    n_iter=15,
)
result = optimize(model, alg, [1.5, 1.5], options=options)
result.minimizer  # ≈ [1.0, 1.0]
```

The initial point must strictly satisfy every barrierified constraint (the
barrier is infinite on the boundary), otherwise an `ArgumentError` naming the
violated constraint is thrown. When bounds are barrierified (`box = true`),
only the finite bound sides contribute a barrier term.

## Options and optimizer

`SDPBarrierOptions` is an alias of `BarrierOptions` and `SDPBarrierAlg` an
alias of `BarrierAlg` (they share the keyword arguments and the selection
semantics described above). The full reference of every documented name,
including `BarrierOptions`, `BarrierAlg`, `BarrierSelection`,
`SDPBarrierOptions`, `SDPBarrierAlg` and the `BarrierResult`/`SDPBarrierResult`
result types, lives on the Docstrings page of this documentation.

## Matrix interfaces

For every `n*n` real positive semidefinite matrix that optimization objective contains, please have two inputs `x_L` and `x_D` representing the lower-triangular and the diagonal part of it. In the function, call `decompress_symmetric(x_L, x_d)` to represent that matrix, which will be handled by Nonocnvex automatically

## Projection utilities

The projection utilities move matrices onto the positive semidefinite cone
and measure the distance to it. `project_psd` returns the closest PSD matrix
to the symmetric part of its argument (eigenvalues clamped at zero),
`project_psd_neg` returns the closest negative semidefinite matrix and
`psd_infeasibility` returns the distance of a matrix from the PSD cone.

```julia
using NonconvexSemidefinite, LinearAlgebra

A = [1.0 2.0 0.0; 2.0 1.0 0.0; 0.0 0.0 -1.0]  # indefinite
P = project_psd(A)       # closest PSD matrix to A
N = project_psd_neg(A)   # closest NSD matrix to A
A ≈ P + N                # Moreau decomposition of the PSD/NSD cone pair
psd_infeasibility(A) ≈ norm(N)  # distance of A from the PSD cone
psd_infeasibility(P) ≈ 0.0
```

`project_psd` ships a ChainRules `rrule` (the Lewis spectral differential),
so reverse-mode AD differentiates through the projection, e.g. inside the
augmented Lagrangian solvers of the Nonconvex ecosystem.

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

## Modeling helpers

`add_psd_variable!` appends the variables of a symmetric `dim × dim`
positive semidefinite matrix to a model as one chunk in the `(x_L, x_D)`
layout (the strict lower-triangular entries followed by the diagonal), with
the diagonal bounded below by zero. It returns a `PSDMatrixView` which
decompresses the matrix from a variable vector: call it with the chunk itself
or with the full flattened variable vector (the chunk is read at the recorded
offset). Combine the view with an explicit semidefinite constraint to keep
the full PSD condition:

```julia
using NonconvexSemidefinite, NonconvexIpopt, LinearAlgebra

model = Model()
M = add_psd_variable!(model, 2)
add_sd_constraint!(model, x -> M(x))
set_objective!(model, x -> sum(abs2, M(x) .- [2.0 0.5; 0.5 3.0]))

alg = SDPBarrierAlg(sub_alg=IpoptAlg())
options = SDPBarrierOptions(sub_options=IpoptOptions(max_iter=200))
result = optimize(model, alg, [0.0, 1.0, 1.0], options=options)
M(result.minimizer)  # ≈ [2.0 0.5; 0.5 3.0]
```

## Full API reference

Every exported name of `NonconvexSemidefinite` is documented on the
Docstrings page: the projection utilities (`project_psd`, `project_psd_neg`,
`psd_infeasibility`), the chordal decomposition utilities (`symmetric_pattern`,
`chordal_extension`, `aggregate_sparsity`, `chordal_decompose`), the modeling
helpers (`PSDMatrixView`, `add_psd_variable!`, `decompress_symmetric`), and
the meta-algorithms with their options, selections and result types
(`BarrierAlg`/`SDPBarrierAlg`, `BarrierOptions`/`SDPBarrierOptions`,
`BarrierSelection`, `BarrierResult`, `SDPQuadAlg`, `SDPQuadOptions`,
`SDPQuadResult`).
