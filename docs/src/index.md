# NonconvexSemidefinite



## Description

If you need to keep your any matrix-valued function of the decision variables positive semidefinite, Nonconvex supports an interface for the [barrier method for semidefinite programming](http://eaton.math.rpi.edu/faculty/Mitchell/courses/matp6640/notes/24A_SDPbarrierbeamer.pdf), which is a meta-algorithm transforming the optimization problem to a series of nonlinear programming problems and solving them using the pre-specified `sub_alg` and `sub_options`.

See [Quick start](quickstart.md) for the first steps, the [quadratic
re-parameterization](quadratic.md), the [general barrier
method](barrier.md) and the [cutting-plane
solver](options.md) for the algorithm internals, [Options and
optimizer](options.md) for the configuration, the [matrix
interfaces](interfaces.md), [projection utilities](projection.md),
[chordal decomposition utilities](chordal.md) and [modeling
helpers](modeling.md) for the API, [Theory](theory.md) for the
algorithmic background and [Docstrings](docstrings.md) for the full API
reference.
