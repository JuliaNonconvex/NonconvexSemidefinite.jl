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
