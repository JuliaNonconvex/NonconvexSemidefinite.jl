# NEWS

## Unreleased

- `NonconvexSemidefiniteGradientFlowExt`: the semidefinite support for
  `NullSpaceAlg` (NonconvexGradientFlow.jl) now lives here, as a weak-dep
  extension loaded when NonconvexGradientFlow is loaded — the conic
  null-space projection of the Feppon–Allaire–Dapogny gradient flow:
  tangent-cone construction from `project_psd` eigendecompositions, the
  conic dual QP over the negative polar cone of the PSD tangent cone
  (projected gradient), the exact tangency projection, signed boundary
  stabilization and a violation-scale step cap.

## 0.4.6

- Docs: the logo now appears only in the sidebar, not in the page body.

## 0.5.0

- The semidefinite cone `PSDCone` now lives in NonconvexCore (which defines
  all constraint sets); NonconvexSemidefinite re-exports it and provides the
  semidefinite algorithms.
- Chordal decomposition rebuilt: `chordal_extension` (AMD and MCS orderings
  built in, MF/METIS/BT via the optional CliqueTrees.jl extension),
  `is_chordal`, `assemble_chordal` with an analytic rrule, and the exact
  Fukuda-conversion `chordal_transform` (clique PSD variable blocks coupled
  by pattern equalities). `chordal_decompose` remains as the clique-submatrix
  form, now documented as a relaxation.
- Kočvara's structured domain decomposition: `sdd_partition`/
  `sdd_transform` (Schur complement on the subdomain interface, with a
  Ferrite.jl extension for general meshes) and the arrow decomposition
  `arrow_decompose` (Theorem 2.3 of Kočvara, Math. Program. 2021).
- `SDCuttingPlaneAlg`: a cutting-plane solver for semidefinite constraints
  using the iterative (ARPACK) eigensolver, with a `block_size` option.
- Requires NonconvexCore 2.3 (the function-in-set framework).
 (Chordal, SDD and cutting-plane algorithms for NonconvexCore 2.3 sets)
## 0.4.5

- Docs: package-name section titles and left-aligned logos (no code changes).
## 0.4.4

- `to_barrier` now removes barrierified box constraints from the sub-model (encoded in the log-barrier objective), so the smooth-extension region is reachable and the barrier terms are not duplicated.\n- `SDPQuadAlg` rejects `rank > dim` with a clear `ArgumentError` instead of handing the sub-solver an init/bounds size mismatch.
## 0.4.3

- Fixed the `decompress_symmetric` docstring example (the `x_L` mapping was
  wrong); documented the 0.3.x/0.4.x releases in NEWS.

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
