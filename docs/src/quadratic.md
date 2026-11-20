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
