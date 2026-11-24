@testitem "ks_aggregate" begin
    using NonconvexSemidefinite, LinearAlgebra, Random, Zygote, SparseArrays
    Random.seed!(1)
    n = 6
    A0 = Matrix(Symmetric(randn(n, n)' * randn(n, n)))
    B0 = Matrix(Diagonal(rand(n) .+ 1.0))

    # deterministic dense eigensolver (B-normalized vectors), for FD
    dense_eigs(A, B, k, which) = begin
        F = eigen(Symmetric(Matrix(A)), Symmetric(Matrix(B)))
        perm = sortperm(F.values; rev=(which === :upper))
        idx = perm[1:min(k, length(perm))]
        λs = F.values[idx]
        Xs = [
            begin
                v = vec(F.vectors[:, j])
                v ./ sqrt(v' * Matrix(B) * v)
            end for j in idx
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

    for which in (:lower, :upper, :ratio)
        f(A) = ks_aggregate(A, B0, 2, 10.0; which=which, eigsolver=dense_eigs)
        g = Zygote.gradient(f, A0)[1]
        g_fd = fd_grad(f, A0)
        @test norm(g - g_fd) / max(norm(g_fd), 1) < 1e-5
    end

    # sanity of the aggregates themselves
    @test ks_aggregate(A0, B0, 2, 10.0; which=:lower, eigsolver=dense_eigs) <=
          ks_aggregate(A0, B0, 2, 10.0; which=:upper, eigsolver=dense_eigs)

    # default LOBPCG path: value agrees with the dense reference, gradient
    # is finite, and the sparse-cotangent path builds for sparse A
    nbig = 40
    A1 = Matrix(Symmetric(randn(nbig, nbig)' * randn(nbig, nbig)))
    B1 = Matrix(Diagonal(rand(nbig) .+ 1.0))
    v_lob = ks_aggregate(A1, B1, 2, 10.0; which=:lower)
    v_dense = ks_aggregate(A1, B1, 2, 10.0; which=:lower, eigsolver=dense_eigs)
    @test abs(v_lob - v_dense) < 1e-3
    g_lob = Zygote.gradient(A -> ks_aggregate(A, B1, 2, 10.0; which=:lower), A1)[1]
    @test all(isfinite, g_lob)

    As = sparse(A1)
    g_sp = Zygote.gradient(
        A -> ks_aggregate(sparse(A), B1, 2, 10.0; which=:upper), A1
    )[1]
    @test all(isfinite, g_sp)
end
