# Test cases of semidefinite programming
using NonconvexSemidefinite, NonconvexIpopt, LinearAlgebra, Test
using Distributions, ChainRulesTestUtils, Random, ChainRulesCore
Random.seed!(1)

# Test setting
mat_dim = 2
mat_length = mat_dim^2
n_sample = 300

@testset "Test matrix to vectors function" begin
    test_rrule(
        NonconvexSemidefinite.rearrange_x,
        rand((mat_dim^2 - mat_dim) ÷ 2),
        rand(mat_dim),
    )
end

@testset "rearrange_x rrule with zero cotangents" begin
    # Regression test: the pullback used to call lowertriangind/diag on
    # AbstractZero cotangents, which threw MethodError.
    y, pb = ChainRulesCore.rrule(
        NonconvexSemidefinite.rearrange_x,
        [0.1, 0.2, 0.3],
        [1.0, 2.0, 3.0],
    )
    cotans = pb((ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent()))
    @test cotans[2] isa ChainRulesCore.AbstractZero
    @test cotans[3] isa ChainRulesCore.AbstractZero
end

function random_psd_mat(mat_dim)
    _mat = randn(mat_dim, mat_dim)
    return _mat' * _mat + I
end

function sd_constraint((x_L, x_D))
    decompress_symmetric(x_L, x_D)
end

@testset "Single semi-definite constraint" begin
    # Randomize groundtruth
    μ, Σ = randn(mat_dim), random_psd_mat(mat_dim)

    # Generate samples
    samples = rand(MvNormal(μ, Σ), n_sample)

    # Objective function
    function f((x_L, x_D))
        A = decompress_symmetric(x_L, x_D)
        # Guard against Ipopt probing trial points that are not positive
        # definite: the barrier handles PSD-ness, but the line search can
        # still evaluate indefinite iterates, which makes `PDMat` throw.
        F = cholesky(A, check = false)
        issuccess(F) || return Inf
        return -loglikelihood(MvNormal(μ, A), samples)
    end

    model = Model(f)

    mat_x0 = random_psd_mat(mat_dim)
    x0 = [mat_x0[NonconvexSemidefinite.lowertriangind(mat_x0)], diag(mat_x0)]
    lbs = [fill(-Inf, length(x0[1])), zeros(length(x0[2]))]
    ubs = [fill(Inf, length(x0[1])), fill(Inf, length(x0[2]))]
    addvar!(model, lbs, ubs, init = x0)

    add_sd_constraint!(model, sd_constraint)

    alg = SDPBarrierAlg(sub_alg = IpoptAlg())
    options = SDPBarrierOptions(sub_options = IpoptOptions(max_iter = 200))
    result = optimize(model, alg, x0, options = options)

    minimum, minimizer, optimal_ind = result.minimum, result.minimizer, result.optimal_ind
    _Σ = sd_constraint(minimizer)

    println("result: \n $result")

    println("minimum: \n $minimum")
    println("minimizer: \n $minimizer")
    println("_Σ: \n $(_Σ)")

    println("Σ: \n $Σ")
    println("abs(_Σ - Σ): \n $(abs.(_Σ - Σ))")
    println("mean(abs(_Σ - Σ)): \n $(mean(abs.(_Σ - Σ)))")

    @test Σ ≈ _Σ rtol = 0.1
end

@testset "Two semi-definite constraints" begin
    # Randomize groundtruth
    μ1, Σ1 = randn(mat_dim), random_psd_mat(mat_dim)
    μ2, Σ2 = randn(mat_dim), random_psd_mat(mat_dim)

    # Generate samples
    samples1 = rand(MvNormal(μ1, Σ1), n_sample)
    samples2 = rand(MvNormal(μ2, Σ2), n_sample)

    # Objective function
    function f((x_L1, x_D1, x_L2, x_D2))
        A1 = decompress_symmetric(x_L1, x_D1)
        A2 = decompress_symmetric(x_L2, x_D2)
        # Guard against Ipopt probing indefinite trial points (see above).
        (issuccess(cholesky(A1, check = false)) &&
         issuccess(cholesky(A2, check = false))) || return Inf
        return -loglikelihood(MvNormal(μ1, A1), samples1) -
               loglikelihood(MvNormal(μ2, A2), samples2)
    end
    model = Model(f)

    mat_x0 = random_psd_mat(mat_dim)
    x0 = [mat_x0[NonconvexSemidefinite.lowertriangind(mat_x0)], diag(mat_x0)]
    x0 = [copy.(x0)..., copy.(x0)...]

    lbs = [fill(-Inf, length(x0[1])), zeros(length(x0[2]))]
    lbs = [copy.(lbs)..., copy.(lbs)...]

    ubs = [fill(Inf, length(x0[1])), fill(Inf, length(x0[2]))]
    ubs = [copy.(ubs)..., copy.(ubs)...]

    addvar!(model, lbs, ubs, init = x0)

    add_sd_constraint!(model, x -> sd_constraint(x[1:2]))
    add_sd_constraint!(model, x -> sd_constraint(x[3:4]))

    alg = SDPBarrierAlg(sub_alg = IpoptAlg())
    options = SDPBarrierOptions(
        sub_options = IpoptOptions(max_iter = 200),
        keep_all = false,
        c_init = [1.5, 2.0],
        c_decr = [0.2, 0.1],
    )
    result = optimize(model, alg, x0, options = options)

    minimum, minimizer, optimal_ind = result.minimum, result.minimizer, result.optimal_ind
    _Σ1 = sd_constraint(minimizer[1:2])
    _Σ2 = sd_constraint(minimizer[3:4])

    println("result: \n $result")

    println("minimum: \n $minimum")
    println("minimizer: \n $minimizer")
    println("_Σ1: \n $(_Σ1)")
    println("_Σ2: \n $(_Σ2)")

    println("Σ1: \n $Σ1")
    println("Σ2: \n $Σ2")
    println("abs(_Σ1 - Σ1): \n $(abs.(_Σ1 - Σ1))")
    println("abs(_Σ2 - Σ2): \n $(abs.(_Σ2 - Σ2))")
    println("mean(abs(_Σ1 - Σ1)): \n $(mean(abs.(_Σ1 - Σ1)))")
    println("mean(abs(_Σ2 - Σ2)): \n $(mean(abs.(_Σ2 - Σ2)))")

    @test Σ1 ≈ _Σ1 rtol = 0.1
    @test Σ2 ≈ _Σ2 rtol = 0.1
end

@testset "Different algorithm" begin
    # Randomize groundtruth
    μ, Σ = randn(mat_dim), random_psd_mat(mat_dim)

    # Generate samples
    samples = rand(MvNormal(μ, Σ), n_sample)

    # Passing sd_constraint but not using SDPBarrierOptions
    # Objective function
    function f((x_L, x_D))
        A = decompress_symmetric(x_L, x_D)
        # Guard against Ipopt probing trial points that are not positive
        # definite: the barrier handles PSD-ness, but the line search can
        # still evaluate indefinite iterates, which makes `PDMat` throw.
        F = cholesky(A, check = false)
        issuccess(F) || return Inf
        return -loglikelihood(MvNormal(μ, A), samples)
    end
    model = Model(f)

    mat_x0 = random_psd_mat(mat_dim)
    x0 = [mat_x0[NonconvexSemidefinite.lowertriangind(mat_x0)], diag(mat_x0)]
    lbs = [fill(-Inf, length(x0[1])), zeros(length(x0[2]))]
    ubs = [fill(Inf, length(x0[1])), fill(Inf, length(x0[2]))]
    addvar!(model, lbs, ubs, init = x0)
    add_sd_constraint!(model, sd_constraint)

    # NonconvexCore < 1.7 silently drops semidefinite constraints with a
    # warning; NonconvexCore >= 1.7 fails fast with a descriptive error.
    # Accept both so this test works against the released and the
    # development version of NonconvexCore.
    err = try
        optimize(
            model,
            IpoptAlg(),
            x0,
            options = IpoptOptions(max_iter = 1, first_order = true),
        )
        nothing
    catch e
        e
    end
    if err !== nothing
        @test err isa ErrorException
        @test occursin("semidefinite", sprint(showerror, err))
    end
end
