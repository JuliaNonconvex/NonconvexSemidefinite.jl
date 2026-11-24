@testitem "ks_aggregate" begin
    using NonconvexSemidefinite, LinearAlgebra, Random, Zygote, SparseArrays
    Random.seed!(1)
    n = 6
    A0 = Matrix(Symmetric(-(randn(n, n)' * randn(n, n))))  # negative definite
    B0 = Matrix(Diagonal(rand(n) .+ 1.0))

    # deterministic dense eigensolver (B-normalized vectors), for FD
    dense_eigs(A, B, k) = begin
        F = eigen(Symmetric(Matrix(A)), Symmetric(Matrix(B)))
        perm = sortperm(F.values)[1:min(k, length(F.values))]
        λs = F.values[perm]
        Xs = [
            begin
                v = vec(F.vectors[:, j])
                v ./ sqrt(v' * Matrix(B) * v)
            end for j in perm
        ]
        return λs, Xs
    end

    # hand-rolled central differences with SINGLE-ENTRY perturbations: the
    # forward symmetrizes A explicitly, so df/dA[i,j] = Σᵢwᵢ·vᵢ[i]·vᵢ[j]
    function fd_grad(f, A; h=1e-6)
        G = similar(A)
        for i in axes(A, 1), j in axes(A, 2)
            Ap = copy(A); Ap[i, j] += h
            Am = copy(A); Am[i, j] -= h
            G[i, j] = (f(Ap) - f(Am)) / (2h)
        end
        return G
    end

    for step in (0.0, 0.25, 0.5, 1.0)
        f(A) = ks_aggregate(A, B0, 2, 10.0; step=step, eigsolver=dense_eigs)
        g = Zygote.gradient(f, A0)[1]
        g_fd = fd_grad(f, A0)
        @test norm(g - g_fd) / max(norm(g_fd), 1) < 1e-5
    end
    # the ratio mode
    f(A) = ks_aggregate(A, B0, 2, 10.0; ratio=true, eigsolver=dense_eigs)
    g = Zygote.gradient(f, A0)[1]
    g_fd = fd_grad(f, A0)
    @test norm(g - g_fd) / max(norm(g_fd), 1) < 1e-5

    # one-sided bounds on the minimum (the eigenvalues are negative here):
    # lower ≤ min(λ) ≤ upper, and the step walks between them
    λs_dense = dense_eigs(A0, B0, 2)[1]
    v0 = ks_aggregate(A0, B0, 2, 10.0; step=0.0, eigsolver=dense_eigs)
    v1 = ks_aggregate(A0, B0, 2, 10.0; step=1.0, eigsolver=dense_eigs)
    vh = ks_aggregate(A0, B0, 2, 10.0; step=0.5, eigsolver=dense_eigs)
    @test v0 <= minimum(λs_dense) <= v1
    @test v0 <= vh <= v1

    # default LOBPCG path: value agrees with the dense reference, gradient
    # is finite, and the sparse-cotangent path builds for sparse A
    nbig = 40
    A1 = Matrix(Symmetric(-(randn(nbig, nbig)' * randn(nbig, nbig))))
    B1 = Matrix(Diagonal(rand(nbig) .+ 1.0))
    for (step, ratio) in ((0.0, false), (0.5, false), (1.0, false), (0.0, true))
        v_lob = ks_aggregate(A1, B1, 2, 10.0; step=step, ratio=ratio)
        v_dense = ks_aggregate(A1, B1, 2, 10.0; step=step, ratio=ratio,
                               eigsolver=dense_eigs)
        @test abs(v_lob - v_dense) < 1e-3
    end
    g_lob = Zygote.gradient(
        A -> ks_aggregate(A, B1, 2, 10.0; step=0.5), A1
    )[1]
    @test all(isfinite, g_lob)

    g_sp = Zygote.gradient(
        A -> ks_aggregate(sparse(A), B1, 2, 10.0; step=0.5), A1
    )[1]
    @test all(isfinite, g_sp)
end
