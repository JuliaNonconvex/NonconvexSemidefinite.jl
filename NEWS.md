# NEWS

## 0.4.2

- Fixed ForwardDiff through `decompress_symmetric`, an rrule cotangent loss
  and the Burer-Monteiro warm start; scheduled formatter workflow.

## 0.4.0

- Block semidefinite constraints (`BlockSDConstraint`): chordal-decomposed
  matrix functions are evaluated once; README and QA parity (docs badges,
  Aqua/JET, formatting), organization logo.

## 0.3.0

- NonconvexCore 2 with strict solver capability declarations.
- SDP domain-layer primitives: PSD projection with an analytic rrule,
  `psd_infeasibility`, chordal decomposition utilities and PSD modeling
  conveniences.
- General `BarrierAlg` (inequality/bound/semidefinite barrierification with
  adaptive schedule and selection policy) and `SDPQuadAlg` (full-rank
  Cholesky and Burer-Monteiro re-parameterizations).
- DocumenterFragments docs with a theory page; tests migrated to TestItems
  with JET, Aqua and formatting quality gates.

## 0.2.1

- The "Different algorithm" test now accepts both the warning (NonconvexCore
  < 1.7) and the fail-fast error (NonconvexCore >= 1.7) when a solver that
  cannot handle semidefinite constraints receives a model with them.
- Added a regression test that the barrier model handed to the sub-solver
  carries no semidefinite constraints.
