@testitem "Log-determinant estimation" begin
    using NonconvexSemidefinite, NonconvexCore, LinearAlgebra, Random, Zygote

    n = 30
    A = Matrix(
        qr(randn(n, n)).Q * Diagonal(range(1.0, 3.0, length=n)) * Matrix(qr(randn(n, n)).Q)'
    )
    A = Symmetric(A + 5 * I) # SPD, condition number ≈ 8
    f! = (y, x) -> mul!(y, A, x)
    rng = MersenneTwister(11)
    lmin, lmax = eigmin(A) - 0.1, eigmax(A) + 0.1

    # values of all matrix-free methods
    @test NonconvexSemidefinite.slq_logdet(f!, n; lanczos_deg=60, nprobes=60, rng=rng) ≈
        logdet(A) rtol = 0.02
    @test NonconvexSemidefinite.chebyshev_logdet(
        f!, n; deg=60, nprobes=200, lmin=lmin, lmax=lmax, rng=rng
    ) ≈ logdet(A) rtol = 2e-2
    @test NonconvexSemidefinite.hutchpp_logdet(
        f!, n; k=20, lanczos_deg=30, nprobes=20, rng=rng
    ) ≈ logdet(A) rtol = 0.05
    @test NonconvexSemidefinite.rational_logdet(f!, n; nodes=30, nprobes=20, rng=rng) ≈
        logdet(A) rtol = 8e-2
    @test NonconvexSemidefinite.taylor_logdet(
        f!, n; terms=60, nprobes=200, lmin=lmin, lmax=lmax, rng=rng
    ) ≈ logdet(A) rtol = 1e-2
    @test NonconvexSemidefinite.trace_integration_logdet(
        f!, n; nodes=16, nprobes=200, rng=rng
    ) ≈ logdet(A) rtol = 3e-2

    # low-rank approximation on a near-identity matrix
    B = Symmetric(Matrix{Float64}(I, n, n) .+ 0.05 .* Matrix(A))
    fB = (y, x) -> mul!(y, B, x)
    @test NonconvexSemidefinite.lowrank_logdet(fB, n; r=30, rng=rng) ≈ logdet(B) rtol = 0.01

    # dispatcher (fresh rng for each call: the estimates share the seed but
    # the rng state advances between calls)
    rng1 = MersenneTwister(11)
    rng2 = MersenneTwister(11)
    e1 = NonconvexSemidefinite.logdet_estimate(f!, n; method=:slq, rng=rng1)
    e2 = NonconvexSemidefinite.slq_logdet(f!, n; rng=rng2)
    @test e1 == e2
    @test_throws ArgumentError NonconvexSemidefinite.logdet_estimate(f!, n; method=:bogus)

    # spectral bounds contain the spectrum
    a, b = NonconvexSemidefinite.lanczos_bounds(f!, n; k=10, rng=rng)
    @test a <= eigmin(A) + 0.5
    @test b >= eigmax(A) - 0.5

    # quadrature weights
    t, w = NonconvexSemidefinite._gauss_laguerre(20)
    @test sum(w .* t .^ 2) ≈ 2.0 rtol = 1e-10 # ∫₀^∞ t² e^{-t} dt = 2
    tl, wl = NonconvexSemidefinite._gauss_legendre_unit(20)
    @test sum(wl .* tl .^ 2) ≈ 1 / 3 rtol = 1e-10 # ∫₀¹ t² dt = 1/3

    # shifted CG accuracy
    σ = 0.7
    b = randn(n)
    u = NonconvexSemidefinite._shifted_cg(f!, b, σ)
    @test (Matrix(A) + σ * I) * u ≈ b rtol = 1e-5

    # rrule gradients: the reverse rule must match the finite-difference
    # gradient of the estimator itself (the rrule contract), using the same
    # probe draws for every evaluation. The spectral bounds of the
    # Chebyshev/Taylor methods are treated as stopped gradients, so fixed
    # bounds are passed to both the rule and the finite differences.
    c = x -> [x[1] x[2]; x[2] x[3]]
    x = [4.0, 1.0, 3.0]
    h = 1e-5
    ecols = eachcol(Matrix{Float64}(I, 3, 3))
    lmin, lmax = eigmin(Symmetric(c(x))) - 0.5, eigmax(Symmetric(c(x))) + 0.5
    for method in (:slq, :hutchpp, :rational, :integration)
        g = Zygote.gradient(
            z -> NonconvexSemidefinite.logdet_estimate_of(
                c, z; method=method, rng=copy(rng), nprobes=60
            ),
            x,
        )[1]
        gfd = [
            (
                NonconvexSemidefinite.logdet_estimate_of(
                    c, x .+ h .* e_i; method=method, rng=copy(rng), nprobes=60
                ) - NonconvexSemidefinite.logdet_estimate_of(
                    c, x .- h .* e_i; method=method, rng=copy(rng), nprobes=60
                )
            ) / (2h) for (i, e_i) in enumerate(ecols)
        ]
        @test g ≈ gfd rtol = 5e-2
    end
    # the polynomial-method reverse rules are not implemented yet (the
    # values remain available)
    for method in (:chebyshev, :taylor)
        @test_throws ArgumentError Zygote.gradient(
            z -> NonconvexSemidefinite.logdet_estimate_of(
                c, z; method=method, rng=copy(rng), nprobes=60
            ),
            x,
        )
    end
    # finite-difference check for the low-rank rrule on a near-identity map
    cB = x -> [x[1] x[2]; x[2] x[3]] + [0.5 0.0; 0.0 0.5]
    g = Zygote.gradient(
        z -> NonconvexSemidefinite.logdet_estimate_of(cB, z; method=:lowrank, r=3, rng=rng),
        x,
    )[1]
    h = 1e-6
    gfd = [
        (
            NonconvexSemidefinite.logdet_estimate_of(
                cB, x .+ h .* e_i; method=:lowrank, r=3, rng=rng
            ) - NonconvexSemidefinite.logdet_estimate_of(
                cB, x .- h .* e_i; method=:lowrank, r=3, rng=rng
            )
        ) / (2h) for (i, e_i) in enumerate(eachcol(Matrix{Float64}(I, 3, 3)))
    ]
    @test g ≈ gfd rtol = 1e-2

    # spectral-bound defaults: the Chebyshev/Taylor estimators estimate the
    # bounds internally with a short Lanczos run when lmin/lmax are absent
    @test NonconvexSemidefinite.chebyshev_logdet(
        f!, n; deg=60, nprobes=100, rng=copy(rng)
    ) ≈ logdet(A) rtol = 1e-1
    @test NonconvexSemidefinite.taylor_logdet(f!, n; terms=60, nprobes=100, rng=copy(rng)) ≈
        logdet(A) rtol = 1e-1

    # Hutch++ rrule with a nontrivial deflated range (n > k): exercises the
    # range-restricted SLQ adjoint and the deflated-operator adjoint
    c3 = z -> [z[1] z[2] 0.0; z[2] z[3] 0.5; 0.0 0.5 2.0]
    x3 = [4.0, 1.0, 3.0]
    ecols3 = eachcol(Matrix{Float64}(I, 3, 3))
    g = Zygote.gradient(
        z -> NonconvexSemidefinite.logdet_estimate_of(
            c3, z; method=:hutchpp, k=2, lanczos_deg=3, nprobes=40, rng=copy(rng)
        ),
        x3,
    )[1]
    gfd = [
        (
            NonconvexSemidefinite.logdet_estimate_of(
                c3,
                x3 .+ h .* e_i;
                method=:hutchpp,
                k=2,
                lanczos_deg=3,
                nprobes=40,
                rng=copy(rng),
            ) - NonconvexSemidefinite.logdet_estimate_of(
                c3,
                x3 .- h .* e_i;
                method=:hutchpp,
                k=2,
                lanczos_deg=3,
                nprobes=40,
                rng=copy(rng),
            )
        ) / (2h) for (i, e_i) in enumerate(ecols3)
    ]
    @test g ≈ gfd rtol = 1e-1

    # LinearFunction-backed constraints: structural dispatch in the value
    # and the reverse rule (X(x) = [x1 x2; x2 x1])
    Af = [1.0 0.0; 0.0 1.0; 0.0 1.0; 1.0 0.0]
    lf = NonconvexCore.LinearFunction(Af, zeros(4))
    x2 = [2.0, 1.0]
    est_lf = NonconvexSemidefinite.logdet_estimate_of(
        lf, x2; method=:slq, lanczos_deg=2, nprobes=2000, rng=copy(rng)
    )
    @test est_lf ≈ log(3.0) rtol = 0.05
    glf = Zygote.gradient(
        z -> NonconvexSemidefinite.logdet_estimate_of(
            lf, z; method=:slq, lanczos_deg=2, nprobes=2000, rng=copy(rng)
        ),
        x2,
    )[1]
    @test glf ≈ [4 / 3, -2 / 3] rtol = 0.05

    # unknown methods are rejected on the reverse path too
    @test_throws ArgumentError Zygote.gradient(
        z -> NonconvexSemidefinite.logdet_estimate_of(c, z; method=:bogus, rng=copy(rng)), x
    )
end
