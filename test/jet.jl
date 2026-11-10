using JET, NonconvexSemidefinite

# Run JET analysis on the package.
JET.test_package(NonconvexSemidefinite; target_modules=(NonconvexSemidefinite,))
