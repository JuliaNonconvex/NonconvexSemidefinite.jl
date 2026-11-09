# Test cases of semidefinite programming
using NonconvexSemidefinite, NonconvexIpopt, LinearAlgebra, Test
using Distributions, ChainRulesTestUtils, Random, ChainRulesCore, SparseArrays
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
        @test err isa Union{ErrorException,ArgumentError}
        @test occursin("semidefinite", sprint(showerror, err))
    end
end

@testset "to_barrier removes semidefinite constraints" begin
    model = Model(x -> sum(x .^ 2))
    addvar!(model, fill(-2.0, 2), fill(2.0, 2))
    add_sd_constraint!(model, x -> diagm([x[1]^2 + 1.0, x[2]^2 + 1.0]))
    vm, x0, _ = NonconvexCore.tovecmodel(model)
    barrier_model = NonconvexSemidefinite.to_barrier(vm, (1.0,))
    @test isempty(barrier_model.sd_constraints.fs)
    # The barrier objective still evaluates to a finite value.
    @test isfinite(barrier_model.objective(x0))
end

@testset "PSD cone projections" begin
    # rrule correctness in the identity region (all eigenvalues positive)
    test_rrule(NonconvexSemidefinite.project_psd, randn(3, 3) + 2I)
    # rrule correctness with mixed-sign eigenvalues away from zero
    test_rrule(NonconvexSemidefinite.project_psd, diagm([-1.0, 1.0, 2.0]))
    A = randn(3, 3)
    A = A + A'
    Z = project_psd(A)
    @test Z ≈ Z'
    @test all(eigvals(Symmetric(Z)) .>= -1e-8)
    @test project_psd(Z) ≈ Z
    # Moreau decomposition: A = Π_{S₊}(A) + Π_{S₋}(A) for symmetric A
    @test project_psd(A) + project_psd_neg(A) ≈ A
    @test psd_infeasibility(Z) ≈ 0 atol = 1e-8
    @test psd_infeasibility(diagm([-2.0, 1.0])) ≈ 2 atol = 1e-8
    # Zero cotangent does not throw
    _, pb = ChainRulesCore.rrule(NonconvexSemidefinite.project_psd, A)
    cotans = pb(ChainRulesCore.ZeroTangent())
    @test cotans[1] isa ChainRulesCore.AbstractZero
    @test cotans[2] isa ChainRulesCore.AbstractZero
end

@testset "Chordal decomposition" begin
    # A path graph is already chordal; its maximal cliques are its edges.
    Ppath = spdiagm(0 => ones(4), 1 => ones(3), -1 => ones(3)) .!= 0
    cliques = chordal_extension(Ppath)
    @test length(cliques) == 3
    @test all(length(c) == 2 for c in cliques)
    @test symmetric_pattern(Ppath) == Ppath
    # Every edge of the pattern is covered by a clique.
    for j in 1:4, k in nzrange(Ppath, j)
        i = rowvals(Ppath)[k]
        i == j && continue
        @test any(c -> i in c && j in c, cliques)
    end
    # A 4-cycle needs one chord; the extension has two cliques of size 3.
    Pcycle = sparse([1, 2, 2, 3, 3, 4, 4, 1], [2, 1, 3, 2, 4, 3, 1, 4], trues(8), 4, 4)
    cliques4 = chordal_extension(Pcycle)
    @test length(cliques4) == 2
    @test all(length(c) == 3 for c in cliques4)
    # Structural sparsity probing (diagonal pattern).
    pat = aggregate_sparsity(x -> diagm(x), ones(3), 3)
    @test Matrix(pat) == Matrix(I, 3, 3)
    # chordal_decompose replaces one tridiagonal 4×4 PSD constraint with three
    # 2×2 clique constraints, all PSD (Grone-Johnson-Sá-Wolkowicz equivalence).
    P = [2.0 0.5 0.0 0.0; 0.5 2.0 0.5 0.0; 0.0 0.5 2.0 0.5; 0.0 0.0 0.5 2.0]
    model = Model(x -> 0.0)
    addvar!(model, -1.0, 1.0)
    add_sd_constraint!(model, x -> P, 4)
    vm, x0, _ = NonconvexCore.tovecmodel(model)
    dm = chordal_decompose(vm, sparse(P .!= 0))
    @test length(dm.sd_constraints.fs) == 3
    for sd in dm.sd_constraints.fs
        M = sd(x0)
        @test size(M) == (2, 2)
        @test minimum(eigvals(Symmetric(M))) > 0
    end
    # Auto-probed decomposition: the union of AD structure and value support
    # recovers the tridiagonal structure → three 2×2 clique constraints.
    dm_auto = chordal_decompose(vm; x0 = x0)
    @test length(dm_auto.sd_constraints.fs) == 3
    @test all(sd -> sd.dim == 2, dm_auto.sd_constraints.fs)
    # A dense constant matrix has full value support → a single 4×4 clique.
    Pdense = P + 0.1 * ones(4, 4)
    modeld = Model(x -> 0.0)
    addvar!(modeld, -1.0, 1.0)
    add_sd_constraint!(modeld, x -> Pdense, 4)
    vmd, x0d, _ = NonconvexCore.tovecmodel(modeld)
    dmd = chordal_decompose(vmd; x0 = x0d)
    @test length(dmd.sd_constraints.fs) == 1
    @test dmd.sd_constraints.fs[1].dim == 4
    # A user pattern that misses a nonzero entry of P(x0) is rejected.
    Pbad = copy(P)
    Pbad[1, 3] = 0.1
    Pbad[3, 1] = 0.1
    modelbad = Model(x -> 0.0)
    addvar!(modelbad, -1.0, 1.0)
    add_sd_constraint!(modelbad, x -> Pbad, 4)
    vmbad, x0bad, _ = NonconvexCore.tovecmodel(modelbad)
    @test_throws ArgumentError chordal_decompose(vmbad, sparse(Ppath); x0 = x0bad)
end

@testset "PSD variable modeling conveniences" begin
    model = Model()
    M = add_psd_variable!(model, 3)
    x0 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # x_L = [1, 2, 3], x_D = [4, 5, 6] in the (x_L, x_D) layout
    @test M(x0) ≈ [4.0 1.0 2.0; 1.0 5.0 3.0; 2.0 3.0 6.0]
    # Offset access when the PSD variable is not the first variable.
    model2 = Model()
    addvar!(model2, -1.0, 1.0)
    M2 = add_psd_variable!(model2, 2)
    xfull = [0.5, 1.0, 2.0, 3.0]
    @test M2(xfull) ≈ [2.0 1.0; 1.0 3.0]
    @test M2(xfull[2:4]) ≈ [2.0 1.0; 1.0 3.0]
    # VecModel support (via the public tovecmodel path)
    m = Model()
    addvar!(m, -1.0, 1.0)
    vm, _, _ = NonconvexCore.tovecmodel(m)
    Mv = add_psd_variable!(vm, 2; init = Matrix{Float64}(I, 2, 2))
    @test Mv(vm.init) ≈ Matrix{Float64}(I, 2, 2)
end


@testset "solver capabilities" begin
    caps = solver_capabilities(SDPBarrierAlg(sub_alg = IpoptAlg()))
    @test caps isa ConstraintCapabilities
    @test caps.sd === true
    @test caps.eq === true
    @test caps.ineq === true
    @test caps.box === true
    @test caps.integer === false
    @test caps.infinite_bounds === true  # forwarded from IpoptAlg's declaration
end

@testset "General barrier method" begin
    # Inequality-constrained problem: min x[1]^2 + x[2]^2 s.t. 2 - x[1] - x[2] <= 0
    # (the minimum on the feasible half-space is at x = [1, 1], value 2).
    function ineq_problem()
        m = Model(x -> sum(x .^ 2))
        addvar!(m, [-5.0, -5.0], [5.0, 5.0])
        add_ineq_constraint!(m, x -> 2 - x[1] - x[2])
        x0 = [1.5, 1.5]   # strictly feasible interior point
        return m, x0
    end
    m, x0 = ineq_problem()
    # Keep the box native for Ipopt (barrier only the inequality).
    alg = BarrierAlg(sub_alg = IpoptAlg())
    options = BarrierOptions(
        sub_options = IpoptOptions(max_iter = 100),
        c_init = 1.0,
        c_decr = 0.1,
        n_iter = 15,
        barrier = BarrierSelection(ineq = true, box = false, sd = false),
    )
    r = optimize(m, alg, x0, options = options)
    @test r.minimizer ≈ [1.0, 1.0] rtol = 1e-3
    @test r.minimum ≈ 2.0 rtol = 1e-3

    # Bounds-only barrierification: min (x - 3)^2 s.t. x <= 2 (active at the
    # optimum), from the strictly interior x0 = 0.
    mb = Model(x -> (x[1] - 3.0)^2)
    addvar!(mb, [-Inf], [2.0])
    alg2 = BarrierAlg(sub_alg = IpoptAlg())
    options2 = BarrierOptions(
        sub_options = IpoptOptions(max_iter = 100),
        n_iter = 12,
        barrier = BarrierSelection(ineq = false, box = true, sd = false),
    )
    rb = optimize(mb, alg2, [0.0], options = options2)
    @test rb.minimizer[1] ≈ 2.0 atol = 1e-3

    # Adaptive schedule + duality-gap termination on the ineq problem.
    options3 = BarrierOptions(
        sub_options = IpoptOptions(max_iter = 100),
        c_init = 1.0,
        n_iter = 50,
        adaptive = true,
        tol = 1e-6,
        barrier = BarrierSelection(ineq = true, box = false, sd = false),
    )
    r3 = optimize(m, alg, x0, options = options3)
    @test r3.minimizer ≈ [1.0, 1.0] rtol = 1e-3

    # Strict-feasibility validation errors.
    mbad, _ = ineq_problem()
    @test_throws ArgumentError optimize(
        mbad,
        alg,
        [0.0, 0.0],   # violates 2 - x1 - x2 <= 0
        options = options3,
    )

    # Barrierify everything: ineq + box + sd together, with sd kept feasible.
    # min -x[1] s.t. x[1]^2 <= 4 (box-like ineq), 1 <= x[1] <= 3 (box), and a
    # trivial 1x1 PSD constraint x[2] + 1 >= 0.
    mmix = Model(x -> -x[1])
    addvar!(mmix, [1.0, 1.0], [3.0, 3.0])
    add_ineq_constraint!(mmix, x -> x[1]^2 - 4)
    add_sd_constraint!(mmix, x -> [x[2] + 1.0;;])
    algmix = BarrierAlg(sub_alg = IpoptAlg())
    optionsmix = BarrierOptions(
        sub_options = IpoptOptions(max_iter = 100),
        n_iter = 15,
        barrier = BarrierSelection(ineq = true, box = true, sd = true),
    )
    rmix = optimize(mmix, algmix, [1.5, 1.5], options = optionsmix)
    # The feasible optimum is x[1] = 2 (ineq active), value -2.
    @test rmix.minimizer[1] ≈ 2.0 rtol = 1e-3
    @test rmix.minimum ≈ -2.0 rtol = 1e-3
end
