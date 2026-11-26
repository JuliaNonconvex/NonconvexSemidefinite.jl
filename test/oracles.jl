@testitem "PSD barrier oracles" begin
    using NonconvexSemidefinite, NonconvexCore, LinearAlgebra, Random, ForwardDiff
    const NSD = NonconvexSemidefinite

    # Determinism for the internally randomized parts (Lanczos start vectors).
    Random.seed!(42)

    c = z -> [z[1] z[2]; z[2] z[3]]
    x = [4.0, 1.0, 3.0]
    X = [x[1] x[2]; x[2] x[3]]
    Xinv = inv(X)

    mk(; mode=:dense, est=nothing, rng=MersenneTwister(3), c2=c) =
        NSD.PSDBarrierOracle(mode, c2, 2, 20, 2, 1e-10, rng, est)

    # --- dense mode: value, gradient, Hessian (log det) ---
    o = mk()
    @test NSD.psd_barrier_value(o, x) ≈ -logdet(X)
    g = zeros(3)
    NSD.psd_barrier_grad!(o, x, g)
    @test g ≈ [-Xinv[1, 1], -2Xinv[1, 2], -Xinv[2, 2]] rtol = 1e-12
    @test o.Xinv ≈ Xinv
    H = zeros(3, 3)
    NSD.psd_hess_dense!(o, x, H)
    w = [0.3, -0.7, 0.4]
    y = zeros(3)
    NSD.psd_hess_prod!(o, x, w, y)
    @test y ≈ H * w rtol = 1e-12
    # Hessian correctness against finite differences of the gradient
    h = 1e-6
    for k in 1:3
        ek = [i == k ? 1.0 : 0.0 for i in 1:3]
        gp = zeros(3)
        gm = zeros(3)
        NSD.psd_barrier_grad!(mk(), x .+ h .* ek, gp)
        NSD.psd_barrier_grad!(mk(), x .- h .* ek, gm)
        @test H[:, k] ≈ (gp .- gm) ./ (2h) rtol = 1e-4
    end
    # non-PSD matrix: the value is -Inf (checked factorization)
    @test NSD.psd_barrier_value(mk(), [1.0, 2.0, 1.0]) == -Inf
    # vector-returning constraint map (reshape branch of _oracle_mat)
    cvec = z -> vec([z[1] z[2]; z[2] z[3]])
    @test NSD.psd_barrier_value(mk(c2=cvec), x) ≈ -logdet(X)

    # --- dense mode: type-aware barriers (:logdet, :inverse, :inverse_square) ---
    @test NSD.psd_barrier_value(mk(), x, :inverse) ≈ tr(Xinv)
    @test NSD.psd_barrier_value(mk(), x, :inverse_square) ≈ tr(Xinv * Xinv) / 2
    @test NSD.psd_barrier_value(mk(), x, :logdet) ≈ -logdet(X)
    for type in (:logdet, :inverse, :inverse_square)
        ot = mk()
        gt = zeros(3)
        NSD.psd_barrier_grad!(ot, x, gt, type)
        Ht = zeros(3, 3)
        NSD.psd_hess_dense!(ot, x, Ht, type)
        yt = zeros(3)
        NSD.psd_hess_prod!(ot, x, w, yt, type)
        @test yt ≈ Ht * w rtol = 1e-12
        # value and gradient against finite differences
        for k in 1:3
            ek = [i == k ? 1.0 : 0.0 for i in 1:3]
            vp = NSD.psd_barrier_value(mk(), x .+ h .* ek, type)
            vm = NSD.psd_barrier_value(mk(), x .- h .* ek, type)
            @test gt[k] ≈ (vp - vm) / (2h) rtol = 1e-5
            gp = zeros(3)
            gm = zeros(3)
            NSD.psd_barrier_grad!(mk(), x .+ h .* ek, gp, type)
            NSD.psd_barrier_grad!(mk(), x .- h .* ek, gm, type)
            @test Ht[:, k] ≈ (gp .- gm) ./ (2h) rtol = 1e-4
        end
    end
    # dual-safe factorization (no Float64 materialization in the AD path)
    gdual = ForwardDiff.gradient(z -> NSD.psd_barrier_value(mk(), z, :inverse), x)
    gdual2 = ForwardDiff.gradient(z -> tr(inv([z[1] z[2]; z[2] z[3]])), x)
    @test gdual ≈ gdual2 rtol = 1e-12
    # errors: unknown sd barrier type and non-dense modes with non-logdet types
    @test_throws ArgumentError NSD.psd_barrier_value(mk(), x, :bogus)
    @test_throws ArgumentError NSD.psd_barrier_grad!(mk(), x, zeros(3), :bogus)
    @test_throws ArgumentError NSD.psd_hess_dense!(mk(), x, zeros(3, 3), :bogus)
    @test_throws ArgumentError NSD.psd_hess_prod!(mk(), x, w, zeros(3), :bogus)
    for mode in (:matrixfree, :trace)
        for type in (:inverse, :inverse_square)
            @test_throws ArgumentError NSD.psd_barrier_value(mk(mode=mode), x, type)
            @test_throws ArgumentError NSD.psd_barrier_grad!(
                mk(mode=mode), x, zeros(3), type
            )
            @test_throws ArgumentError NSD.psd_hess_dense!(
                mk(mode=mode), x, zeros(3, 3), type
            )
            @test_throws ArgumentError NSD.psd_hess_prod!(
                mk(mode=mode), x, w, zeros(3), type
            )
        end
    end

    # --- LinearFunction-backed constraint: structural dispatch ---
    Af = [1.0 0.0; 0.0 1.0; 0.0 1.0; 1.0 0.0]
    lf = NonconvexCore.LinearFunction(Af, zeros(4))
    x2 = [2.0, 1.0]
    X2 = [2.0 1.0; 1.0 2.0]
    ol = NSD.PSDBarrierOracle(:dense, lf, 2, 20, 2, 1e-10, MersenneTwister(3), nothing)
    @test NSD.psd_barrier_value(ol, x2) ≈ -logdet(X2)
    gl = zeros(2)
    NSD.psd_barrier_grad!(ol, x2, gl)
    X2inv = inv(X2)
    @test gl ≈ [-X2inv[1, 1] - X2inv[2, 2], -2X2inv[1, 2]] rtol = 1e-12
    glt = zeros(2)
    NSD.psd_barrier_grad!(ol, x2, glt, :inverse)
    @test glt ≈ [
        -(X2inv[1, 1]^2 + 2X2inv[1, 2]^2 + X2inv[2, 2]^2),
        -2 * (X2inv[1, 1] + X2inv[2, 2]) * X2inv[1, 2],
    ] rtol = 1e-10
    # Hessian (structural dispatch in the derivative blocks): exact entries
    # tr(X⁻¹ ∂Xᵢ X⁻¹ ∂Xⱼ) with ∂X₁ = I and ∂X₂ = [0 1; 1 0]
    Hl = zeros(2, 2)
    NSD.psd_hess_dense!(ol, x2, Hl)
    @test Hl ≈ [
        X2inv[1, 1]^2 + 2X2inv[1, 2]^2 + X2inv[2, 2]^2
        2 * (X2inv[1, 1] + X2inv[2, 2]) * X2inv[1, 2];;
        2 * (X2inv[1, 1] + X2inv[2, 2]) * X2inv[1, 2]
        2X2inv[1, 2]^2 + 2X2inv[1, 1] * X2inv[2, 2]
    ] rtol = 1e-10
    yl = zeros(2)
    wl = [0.6, -0.2]
    NSD.psd_hess_prod!(ol, x2, wl, yl)
    @test yl ≈ Hl * wl rtol = 1e-10

    # --- matrixfree mode (Lanczos; exact for full-dimensional Krylov spaces) ---
    om = mk(mode=:matrixfree)
    @test NSD.psd_barrier_value(om, x) ≈ -logdet(X) rtol = 1e-10
    gm = zeros(3)
    NSD.psd_barrier_grad!(om, x, gm)
    @test gm ≈ [-Xinv[1, 1], -2Xinv[1, 2], -Xinv[2, 2]] rtol = 1e-10
    ym = zeros(3)
    NSD.psd_hess_prod!(om, x, w, ym) # cached Lanczos basis from the value call
    @test ym ≈ H * w rtol = 1e-10
    om2 = mk(mode=:matrixfree) # fresh oracle: recompute the basis inside
    NSD.psd_hess_prod!(om2, x, w, ym)
    @test ym ≈ H * w rtol = 1e-10
    @test NSD.psd_barrier_value(mk(mode=:matrixfree), [1.0, 2.0, 1.0]) == -Inf

    # --- trace mode: randomized value, gradient and Hessian products ---
    ot = mk(mode=:trace)
    @test NSD.psd_barrier_value(ot, x) ≈ -logdet(X) rtol = 0.1
    # custom log-det estimator closure
    exact_est = (f!, n, rng) -> logdet(Symmetric(X))
    oc = mk(mode=:trace, est=exact_est)
    @test NSD.psd_barrier_value(oc, x) ≈ -logdet(X)
    # gradient (Hutchinson with CG solves) against the exact gradient
    gtr = zeros(3)
    NSD.psd_barrier_grad!(
        NSD.PSDBarrierOracle(:trace, c, 2, 2000, 2, 1e-10, MersenneTwister(7), nothing),
        x,
        gtr,
    )
    @test gtr ≈ [-Xinv[1, 1], -2Xinv[1, 2], -Xinv[2, 2]] rtol = 5e-2
    # scalar trace mode: with Rademacher probes the biased-Hutchinson
    # Gauss-Newton Hessian product collapses to the exact w/x² (z⁴ = 1) and
    # the gradient expectation is -1/x
    c1 = z -> [z[1]]
    o1 = NSD.PSDBarrierOracle(:trace, c1, 1, 20000, 2, 1e-10, MersenneTwister(9), nothing)
    g1 = zeros(1)
    NSD.psd_barrier_grad!(o1, [2.0], g1)
    @test g1[1] ≈ -0.5 rtol = 5e-2
    y1 = zeros(1)
    NSD.psd_hess_prod!(o1, [2.0], [1.0], y1)
    @test y1[1] ≈ 1 / 4 rtol = 5e-2
end
