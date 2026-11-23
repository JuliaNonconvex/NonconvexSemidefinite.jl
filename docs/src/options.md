## Options and optimizer

`SDPBarrierOptions` is an alias of `BarrierOptions` and `SDPBarrierAlg` an
alias of `BarrierAlg` (they share the keyword arguments and the selection
semantics described above). The full reference of every documented name,
including `BarrierOptions`, `BarrierAlg`, `BarrierSelection`,
`SDPBarrierOptions`, `SDPBarrierAlg` and the `BarrierResult`/`SDPBarrierResult`
result types, lives on the Docstrings page of this documentation.

## Cutting-plane solver

`SDCuttingPlaneAlg` enforces the semidefinite constraints by iteratively
appending the most-violated eigenvector cut (see the Theory page). It
takes a `sub_alg` for the NLP subproblems, e.g. `SDCuttingPlaneAlg(IpoptAlg())`,
and `SDCuttingPlaneOptions` exposes `block_size` — the eigen-solve for the
cut direction is restricted to diagonal blocks of the constraint matrix
when `block_size > 0` (a cheap relaxation of the cut direction).
