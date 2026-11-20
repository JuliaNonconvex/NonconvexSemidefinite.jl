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
