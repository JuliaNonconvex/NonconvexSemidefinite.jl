# End-to-end feature coverage for NonconvexSemidefinite.
#
# Complements test/semidefinite.jl and test/coverage_extra.jl: every exported
# feature is exercised through the public API, in particular the
# meta-algorithms driven end-to-end with Ipopt on small problems.

@testitem "SDPQuadAlg end-to-end: full-rank and rank-1 re-parameterization" begin
    using NonconvexSemidefinite, NonconvexIpopt, LinearAlgebra, Test

    # min over PSD 2×2 A of ‖A - T‖²_F with T = [4 2; 2 1] (rank 1).
    # The optimum is A* = T: minimum 0, rank 1 — representable both by the
    # full-rank Cholesky factor and by a Burer–Monteiro factor of rank 1 ≤ 2.
    T = [4.0 2.0; 2.0 1.0]
    function quad_problem()
        m = Model(x -> begin
            A = decompress_symmetric(@view(x[1:1]), @view(x[2:3]))
            return sum(abs2, A .- T)
        end)
        x0 = [0.0, 2.0, 1.0]  # (x_L, x_D) chunk of [2 0; 0 1] ⪰ 0
        addvar!(m, [-10.0, 0.0, 0.0], [10.0, 10.0, 10.0]; init=x0)
        add_sd_constraint!(m, x -> decompress_symmetric(@view(x[1:1]), @view(x[2:3])), 2)
        return m, x0
    end
    m, x0 = quad_problem()
    alg = SDPQuadAlg(sub_alg=IpoptAlg())

    # Full-rank Cholesky re-parameterization (rank = nothing).
    options = SDPQuadOptions(sub_options=IpoptOptions(max_iter=200))
    r = optimize(m, alg, x0, options=options)
    @test r isa SDPQuadResult
    A = decompress_symmetric(@view(r.minimizer[1:1]), @view(r.minimizer[2:3]))
    @test A ≈ T atol = 1e-4
    @test minimum(eigvals(Symmetric(A))) >= -1e-6
    @test r.minimum ≈ 0.0 atol = 1e-6
    @test length(r.minimizer) == 3

    # Burer–Monteiro with rank = 1 ≤ n = 2: the rank-1 optimum is representable.
    options_lr = SDPQuadOptions(sub_options=IpoptOptions(max_iter=200), rank=1)
    r_lr = optimize(m, alg, x0, options=options_lr)
    @test r_lr isa SDPQuadResult
    A_lr = decompress_symmetric(@view(r_lr.minimizer[1:1]), @view(r_lr.minimizer[2:3]))
    @test A_lr ≈ T atol = 1e-4
    @test minimum(eigvals(Symmetric(A_lr))) >= -1e-6
    @test r_lr.minimum ≈ 0.0 atol = 1e-6
end

@testitem "BarrierAlg end-to-end on a non-SDP problem (ineq + box)" begin
    using NonconvexSemidefinite, NonconvexIpopt, LinearAlgebra, Test

    # min x[1]^2 + x[2]^2  s.t.  2 - x[1] - x[2] <= 0  (x[1] + x[2] >= 2)
    # and box 0 <= x <= 3. The unconstrained optimum (0, 0) is infeasible; the
    # constrained optimum is (1, 1) with value 2, where the inequality is
    # active and the box is inactive. Both the inequality and the box are
    # barrierified through BarrierSelection.
    m = Model(x -> sum(abs2, x))
    addvar!(m, [0.0, 0.0], [3.0, 3.0])
    add_ineq_constraint!(m, x -> 2.0 - x[1] - x[2])
    alg = BarrierAlg(sub_alg=IpoptAlg())
    options = BarrierOptions(
        sub_options=IpoptOptions(max_iter=100),
        c_init=1.0,
        c_decr=0.1,
        n_iter=15,
        keep_all=true,
        barrier=BarrierSelection(ineq=true, box=true, sd=false),
    )
    r = optimize(m, alg, [1.5, 1.5], options=options)
    @test r isa BarrierResult
    @test r.minimizer ≈ [1.0, 1.0] rtol = 1e-3
    @test r.minimum ≈ 2.0 rtol = 1e-3
    # keep_all stores the full subproblem trajectory; optimal_ind points at
    # the best entry.
    @test r.results isa Vector
    @test !isempty(r.results)
    @test r.optimal_ind == argmin(first.(r.results))
    @test first(r.results[r.optimal_ind]) == r.minimum

    # Bounds-only barrierification through a full run: min (x - 3)^2 s.t.
    # x <= 2 (the upper bound is active at the optimum), from x0 = 0.
    mb = Model(x -> (x[1] - 3.0)^2)
    addvar!(mb, [-Inf], [2.0])
    optionsb = BarrierOptions(
        sub_options=IpoptOptions(max_iter=100),
        n_iter=12,
        barrier=BarrierSelection(ineq=false, box=true, sd=false),
    )
    rb = optimize(mb, BarrierAlg(sub_alg=IpoptAlg()), [0.0], options=optionsb)
    @test rb isa BarrierResult
    @test rb.minimizer[1] ≈ 2.0 atol = 1e-3
    @test rb.results === nothing  # keep_all defaults to false
    @test rb.optimal_ind === nothing
end

@testitem "Projection utilities on representative matrices" begin
    using NonconvexSemidefinite, LinearAlgebra, Test
    using ChainRulesCore, Random

    # Representative matrices: PSD, NSD, indefinite with mixed eigenvalues,
    # rank-deficient PSD and a non-symmetric input.
    for A in (
        [2.0 0.5; 0.5 1.0], # PSD
        [-3.0 0.1; 0.1 -2.0], # NSD
        [1.0 0.5; 0.5 -1.0], # indefinite
        [1.0 1.0; 1.0 1.0], # rank-deficient PSD
        [0.0 2.0; 0.0 0.0], # non-symmetric
        [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0], # 3×3 PSD (tridiagonal)
    )
        S = (A + A') / 2
        P = project_psd(A)
        N = project_psd_neg(A)
        @test P ≈ P'
        @test N ≈ N'
        # Moreau decomposition on the symmetric part.
        @test P + N ≈ S
        # Idempotence and cone membership (up to round-off: the idempotence
        # error of the near-zero projections is at the floating-point scale).
        @test project_psd(P) ≈ P atol = 1e-12
        @test project_psd_neg(N) ≈ N atol = 1e-12
        @test minimum(eigvals(Symmetric(P))) >= -1e-8
        @test maximum(eigvals(Symmetric(N))) <= 1e-8
        # The projection is the closest PSD matrix: its distance to the
        # symmetric part equals the infeasibility measure.
        @test norm(P - S) ≈ psd_infeasibility(A) atol = 1e-8
        # psd_infeasibility vanishes exactly on PSD matrices and equals the
        # norm of the negative-eigenvalue part otherwise.
        @test psd_infeasibility(P) ≈ 0.0 atol = 1e-8
        @test psd_infeasibility(A) ≈ norm(N) atol = 1e-8
    end

    # rrule consistency with central finite differences at a smooth point
    # (PSD matrix with distinct eigenvalues: the projection is the identity).
    Random.seed!(7)
    B = randn(4, 4)
    B = B + B' + 3.0 * I
    B[1, 1] += 1.0  # break accidental eigenvalue ties
    _, pb = ChainRulesCore.rrule(project_psd, B)
    E = randn(4, 4)
    E = (E + E') / 2
    Δfd = pb(E)[2]
    h = 1e-6
    fd = (project_psd(B + h .* E) - project_psd(B - h .* E)) / (2h)
    @test Δfd ≈ fd rtol = 1e-4
end

@testitem "Chordal decomposition end-to-end: dense vs chordal solve" begin
    using NonconvexSemidefinite, NonconvexIpopt, NonconvexCore, LinearAlgebra, Test
    using SparseArrays

    # min Σ (xᵢ - 1)²  s.t.  P(x) = tridiag(x, 0.5) ⪰ 0.
    # The tridiagonal 4×4 constraint has a chordal (path) pattern whose
    # maximal cliques are the three edges, so the dense 4×4 constraint is
    # exactly equivalent to the three 2×2 clique constraints.
    function sd(x)
        return [x[1] 0.5 0.0 0.0; 0.5 x[2] 0.5 0.0; 0.0 0.5 x[3] 0.5; 0.0 0.0 0.5 x[4]]
    end
    P0 = sd([2.0, 2.0, 2.0, 2.0])
    pattern = sparse(P0 .!= 0)

    function make_model()
        m = Model(x -> sum(abs2, x .- 1.0))
        addvar!(m, fill(-10.0, 4), fill(10.0, 4))
        add_sd_constraint!(m, sd, 4)
        return m
    end
    model = make_model()
    vm, _, _ = NonconvexCore.tovecmodel(model)
    dm = chordal_decompose(vm, pattern)
    # One 4×4 constraint becomes a single block constraint with three 2×2
    # clique blocks.
    @test length(dm.sd_constraints.fs) == 1
    @test dm.sd_constraints.fs[1] isa NonconvexSemidefinite.BlockSDConstraint
    @test NonconvexSemidefinite.getdims(dm.sd_constraints.fs[1]) == [2, 2, 2]
    # Every off-diagonal pattern entry is covered by some clique.
    cliques = chordal_extension(pattern)
    for j in 1:4, i in (j + 1):4
        if pattern[i, j]
            @test any(c -> i in c && j in c, cliques)
        end
    end

    options = SDPBarrierOptions(sub_options=IpoptOptions(max_iter=200), n_iter=10)
    alg = SDPBarrierAlg(sub_alg=IpoptAlg())

    # Dense solve and chordal solve of the equivalent model.
    x0 = [2.0, 2.0, 2.0, 2.0]  # strictly interior
    rdense = optimize(model, alg, x0, options=options)
    rchord = optimize(dm, alg, x0, options=options)

    @test rdense.minimizer ≈ [1.0, 1.0, 1.0, 1.0] atol = 1e-3
    @test rchord.minimizer ≈ [1.0, 1.0, 1.0, 1.0] atol = 1e-3
    @test rdense.minimizer ≈ rchord.minimizer atol = 1e-3
    @test rdense.minimum ≈ rchord.minimum atol = 1e-3
    # Every clique block value at the chordal minimizer is PSD.
    for B in dm.sd_constraints.fs[1](rchord.minimizer)
        @test minimum(eigvals(Symmetric(B))) > -1e-5
    end
end

@testitem "PSDMatrixView + add_psd_variable! through optimize" begin
    using NonconvexSemidefinite, NonconvexIpopt, LinearAlgebra, Test

    # Add a 2×2 PSD matrix variable, keep it PSD with an explicit
    # semidefinite constraint and optimize a least-squares objective over the
    # entries.
    target = [2.0 0.5; 0.5 3.0]
    model = Model()
    M = add_psd_variable!(model, 2)
    add_sd_constraint!(model, x -> M(x))
    set_objective!(model, x -> sum(abs2, M(x) .- target))

    @test size(M) == (2, 2)
    x0 = [0.0, 1.0, 1.0]  # chunk: x_L = [0], x_D = [1, 1] → I
    @test M(x0) ≈ Matrix{Float64}(I, 2, 2)

    alg = SDPBarrierAlg(sub_alg=IpoptAlg())
    options = SDPBarrierOptions(sub_options=IpoptOptions(max_iter=200), n_iter=10)
    r = optimize(model, alg, x0, options=options)
    A = M(r.minimizer)
    @test A ≈ target atol = 1e-3
    @test r.minimum ≈ 0.0 atol = 1e-4
    @test minimum(eigvals(Symmetric(A))) >= -1e-5

    # A PSD variable placed behind a plain variable: the recorded offset is
    # honored through a full optimize.
    model2 = Model()
    addvar!(model2, -1.0, 1.0; init=0.0)
    M2 = add_psd_variable!(model2, 2)
    add_sd_constraint!(model2, x -> M2(x))
    set_objective!(model2, x -> sum(abs2, M2(x) .- Matrix{Float64}(I, 2, 2)))
    r2 = optimize(model2, alg, [0.0, 0.0, 1.0, 1.0], options=options)
    @test M2(r2.minimizer) ≈ Matrix{Float64}(I, 2, 2) atol = 1e-3
    @test r2.minimizer[1] ≈ 0.0 atol = 1e-4
end

@testitem "decompress_symmetric round-trips" begin
    using NonconvexSemidefinite, LinearAlgebra, Test, Random

    Random.seed!(3)
    for dim in 1:5
        B = randn(dim, dim)
        A = B' * B + I  # symmetric PSD
        x_L = A[NonconvexSemidefinite.lowertriangind(A)]
        x_D = diag(A)
        R = decompress_symmetric(x_L, x_D)
        @test R ≈ A
        @test R isa Symmetric
        @test size(R) == (dim, dim)
        @test Matrix(R) ≈ Matrix(Symmetric(A))
        # The matrix form agrees with the vector form.
        L, D = NonconvexSemidefinite.rearrange_x(x_L, x_D)
        @test decompress_symmetric(L, D) ≈ R
        # The diagonal and strict lower part survive the round-trip exactly.
        @test diag(R) == x_D
        @test R[NonconvexSemidefinite.lowertriangind(Matrix(R))] == x_L
    end

    # Integer entries round-trip too.
    xL = [1, 2, 3]
    xD = [4, 5, 6]
    @test decompress_symmetric(xL, xD) ≈ [4 1 2; 1 5 3; 2 3 6]
    @test decompress_symmetric(xL, xD) isa Symmetric

    # 1×1 corner case: empty strict-lower chunk.
    @test decompress_symmetric(Float64[], [7.0]) ≈ fill(7.0, 1, 1)

    # Round-trip through add_psd_variable!-style chunks on random matrices.
    Random.seed!(4)
    for dim in 2:4
        B = randn(dim, dim)
        A = B' * B + 0.5 * I
        chunk = vcat(A[NonconvexSemidefinite.lowertriangind(A)], diag(A))
        k = dim * (dim - 1) ÷ 2
        R2 = decompress_symmetric(@view(chunk[1:k]), @view(chunk[(k + 1):end]))
        @test R2 ≈ A
    end
end
