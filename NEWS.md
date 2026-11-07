# NEWS

## 0.2.1

- The "Different algorithm" test now accepts both the warning (NonconvexCore
  < 1.7) and the fail-fast error (NonconvexCore >= 1.7) when a solver that
  cannot handle semidefinite constraints receives a model with them.
- Added a regression test that the barrier model handed to the sub-solver
  carries no semidefinite constraints.
