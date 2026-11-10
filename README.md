<p align="center">
  <img src="assets/logo.png" width="400" alt="NonconvexSemidefinite.jl logo">
</p>

[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://julianonconvex.github.io/NonconvexSemidefinite.jl/stable)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://julianonconvex.github.io/NonconvexSemidefinite.jl/dev)
[![Build Status](https://github.com/JuliaNonconvex/NonconvexSemidefinite.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JuliaNonconvex/NonconvexSemidefinite.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaNonconvex/NonconvexSemidefinite.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaNonconvex/NonconvexSemidefinite.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-Blue-4473c9.svg)](https://github.com/invenia/BlueStyle)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

# NonconvexSemidefinite.jl

Nonlinear semidefinite programming

Barrier and re-parameterization algorithms, PSD projections and chordal decomposition utilities.

NonconvexSemidefinite.jl is part of the [JuliaNonconvex](https://github.com/JuliaNonconvex) ecosystem built on [NonconvexCore.jl](https://github.com/JuliaNonconvex/NonconvexCore.jl).

## Quick start

```julia
using Distributionsusing NonconvexSemidefinite, NonconvexIpopt

# Draw random multivariate gaussian samples
# Random groundtruth
mat_dim = 3
μ = randn(mat_dim)
Σ = rand(mat_dim, mat_dim)
Σ = Σ + Σ' + 2I
# Generate
n_sample = 1000
samples = rand(MvNormal(μ, Σ), n_sample)

# Define objective function
function f((x_L, x_D))
    return -loglikelihood(MvNormal(μ, decompress_symmetric(x_L, x_D)), samples)
end
# Define the matrix-valued function
function sd_constraint((x_L, x_D))
    return decompress_symmetric(x_L, x_D)
end

# Define settings
model = Model(f)
mat_x0 = rand(mat_dim, mat_dim)
mat_x0 = mat_x0 + mat_x0' + I

x0 = [mat_x0[NonconvexSemidefinite.lowertriangind(mat_x0)], diag(mat_x0)]
lbs = [fill(-Inf, length(x0[1])), zeros(length(x0[2]))]
ubs = [fill(Inf, length(x0[1])), fill(Inf, length(x0[2]))]
addvar!(model, lbs, ubs)
add_sd_constraint!(model, sd_constraint)
alg = SDPBarrierAlg(sub_alg=IpoptAlg())
options = SDPBarrierOptions(sub_options=IpoptOptions(max_iter=200), n_iter = 20)

# Optimize
result = optimize(model, alg, x0, options = options)

# Relative error norm
norm(sd_constraint(result.minimizer) - Σ) / norm(Σ)
```

## Documentation

The full documentation is available at [https://julianonconvex.github.io/NonconvexSemidefinite.jl/stable](https://julianonconvex.github.io/NonconvexSemidefinite.jl/stable) (stable) and [https://julianonconvex.github.io/NonconvexSemidefinite.jl/dev](https://julianonconvex.github.io/NonconvexSemidefinite.jl/dev) (development).

## Installation

```julia
using Pkg
Pkg.add("NonconvexSemidefinite")
```
