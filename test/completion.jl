@testitem "Chordal PSD projection and its implicit adjoint" begin
    using NonconvexSemidefinite, NonconvexCore, ChainRulesCore, LinearAlgebra, Test

    # Exactness: a tridiagonal PSD matrix is reproduced exactly (Theorem 2.2).
    cl = [[1, 2], [2, 3]]
    P2 = [2.0 1.0 0.0; 1.0 2.0 1.0; 0.0 1.0 2.0]
    A2 = chordal_projection(P2, cl; max_iters=5000)
    @test norm(A2 - P2) < 1e-8
    # An indefinite tridiagonal is projected to the nearest tridiagonal PSD
    # matrix (distance > 0, projection PSD and pattern-preserving).
    P1 = [1.0 1.0 0.0; 1.0 1.0 1.0; 0.0 1.0 1.0]
    A1 = chordal_projection(P1, cl; max_iters=5000)
    d = norm(A1 - P1)
    @test d > 0.1
    @test minimum(eigvals(Symmetric(A1))) > -1e-8
    @test abs(A1[1, 3]) < 1e-10

    # The implicit adjoint matches finite differences through P.
    _, pb = ChainRulesCore.rrule(chordal_projection, P1, cl; max_iters=5000)
    ΔA = [1.0 2.0 0.0; 2.0 3.0 4.0; 0.0 4.0 5.0]
    ΔP = pb(ΔA)[2]
    h = 1e-6
    for (i, j) in ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3))
        E = zeros(3, 3)
        E[i, j] = h
        E[j, i] = h
        fd =
            (
                sum((chordal_projection(P1 .+ E, cl; max_iters=5000) .- P1) .* ΔA) -
                sum((chordal_projection(P1 .- E, cl; max_iters=5000) .- P1) .* ΔA)
            ) / (2h)
        ad = (i == j ? 1.0 : 2.0) * ΔP[i, j]
        @test isapprox(ad, fd; rtol=1e-4, atol=1e-5)
    end

    # A larger chordal pattern (grid cliques), clearly non-decomposable.
    cl9 = [[2, 3, 6], [6, 8, 9], [1, 2, 4], [4, 7, 8], [2, 4, 5, 8], [2, 5, 6, 8]]
    P9 = zeros(9, 9)
    for C in cl9, i in C, j in C
        P9[i, j] += 0.3
    end
    P9[1, 1] -= 0.8
    P9[9, 9] += 1.0
    P9 = (P9 + P9') / 2
    A9 = chordal_projection(P9, cl9; max_iters=5000)
    @test norm(A9 - P9) > 0.5
    _, pb9 = ChainRulesCore.rrule(chordal_projection, P9, cl9; max_iters=5000)
    ΔP9 = pb9(ones(9, 9))[2]
    for (i, j) in ((1, 1), (2, 2), (2, 5), (4, 8), (9, 9))
        E = zeros(9, 9)
        E[i, j] = h
        E[j, i] = h
        fd =
            (
                sum(
                    (chordal_projection(P9 .+ E, cl9; max_iters=5000) .- P9) .* ones(9, 9)
                ) -
                sum((chordal_projection(P9 .- E, cl9; max_iters=5000) .- P9) .* ones(9, 9))
            ) / (2h)
        ad = (i == j ? 1.0 : 2.0) * ΔP9[i, j]
        @test isapprox(ad, fd; rtol=1e-4, atol=1e-5)
    end
end
