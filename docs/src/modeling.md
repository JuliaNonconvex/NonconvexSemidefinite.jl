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
