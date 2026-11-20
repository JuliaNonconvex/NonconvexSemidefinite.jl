## Quick start

Optimizing over a multivariate gaussian distribution with artificial samples using `Ipopt`:

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
