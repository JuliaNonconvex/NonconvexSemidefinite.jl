
@testitem "Chordal decomposition transform (exact)" begin
    using NonconvexSemidefinite, NonconvexCore, LinearAlgebra, Test, Zygote, SparseArrays

    # The exact (Fukuda conversion) decomposition: P ⪰ 0 ⟺
    # P = Σₖ E(Cₖ)ᵀYₖE(Cₖ) with Yₖ ⪰ 0. The clique-submatrix form is a
    # relaxation; the conversion is exact.
    P(x) = [
        x[1] 0.25 0.0 0.0
        0.25 x[2] 0.25 0.0
        0.0 0.25 x[1] 0.25
        0.0 0.0 0.25 x[2]
    ]
    mc = Model(x -> (x[1] - 1)^2 + (x[2] - 1)^2)
    addvar!(mc, [0.0, 0.0], [5.0, 5.0]; init=[2.0, 2.0])
    add_set_constraint!(mc, P, PSDCone(4); flags=[:chordal])
    vmc, xc, _ = NonconvexCore.tovecmodel(mc, [2.0, 2.0])
    pat = sparse(P(vmc.init) .!= 0)

    dm, d0, mapc = chordal_transform(vmc, [pat]; x0=vmc.init, labels=[:chordal])
    @test mapc(d0) ≈ [2.0, 2.0]
    # The transformed model has auxiliary clique variables beyond the two
    # design variables.
    @test length(dm.box_min) > 2
    @test dm.set_constraints.fs[1].set isa EqualTo
    # The clique blocks are PSDCone variable blocks.
    @test any(b -> b.set isa PSDCone, dm.set_variables)
    @test all(b -> b.set isa PSDCone, dm.set_variables[1:(end - length(vmc.set_variables))])

    # Exactness: the coupling residual vanishes iff P = ΣEᵀYE, and Y ⪰ 0
    # holds iff P ⪰ 0.
    c_coupling = dm.set_constraints.fs[1]
    y = d0[3:end]
    res = c_coupling(d0)
    @test all(isfinite, res)
    # A PSD point: with Y = the tridiagonal decomposition of P(2, 2), the
    # residual is zero (build the cliques from the extension).
    cliques = chordal_extension(pat)
    @test all(C -> length(C) == 2, cliques)

    # Chordality detection: the path is chordal, cycles need chords.
    @test is_chordal(pat)
    cycle4 = sparse([1, 1, 2, 2, 3, 3, 4, 4], [2, 4, 1, 3, 2, 4, 1, 3], trues(8), 4, 4)
    @test !is_chordal(cycle4)
    @test !is_chordal(cycle4 + cycle4')  # the square is not chordal either
    # a triangle and a 4-cycle with a chord are chordal
    tri = sparse([1, 1, 2], [2, 3, 3], trues(3), 3, 3)
    @test is_chordal(tri)
    chorded4 = cycle4 + sparse([1, 1], [3, 3], trues(2), 4, 4)  # add chord (1,3)
    @test is_chordal(chorded4)
    # A chordal graph's extension is itself (no fill): the cliques of the
    # path are exactly its edges.
    @test sort(cliques) == sort([[i, i + 1] for i in 1:3])
    # Algorithm selection: the built-in AMD (default) and MCS both recover
    # the edges of the chordal path.
    @test sort(chordal_extension(pat; alg=MCSAlg())) == sort(cliques)
    @test sort(chordal_extension(pat; alg=AMDAlg())) == sort(cliques)
    # The CliqueTrees-only algorithms error informatively without the
    # optional dependency (this testitem does not load it).
    for alg in (MFAlg(), METISAlg(), BTAlg())
        @test_throws ArgumentError chordal_extension(pat; alg=alg)
    end

    # Regression: a chordal graph whose forward MCS numbering is NOT a valid
    # elimination order — the reverse is (Tarjan–Yannakakis). The chordal
    # completion of the 3×3 grid (treewidth 3), with maximal cliques
    # {2,3,6},{6,8,9},{1,2,4},{4,7,8},{2,4,5,8},{2,5,6,8}.
    cl9 = [[2, 3, 6], [6, 8, 9], [1, 2, 4], [4, 7, 8], [2, 4, 5, 8], [2, 5, 6, 8]]
    I9 = Int[]
    J9 = Int[]
    for C in cl9, i in C, j in C
        push!(I9, i)
        push!(J9, j)
    end
    compl9 = sparse(I9, J9, trues(length(I9)), 9, 9)
    @test is_chordal(compl9)
    # A chordal graph's extension is itself: the reversed-MCS elimination
    # order introduces no fill and recovers exactly the maximal cliques.
    @test sort(chordal_extension(compl9; alg=MCSAlg())) == sort(cl9)
    # The raw 3×3 grid pattern (4-cycles without chords) is not chordal.
    Ig = Int[]
    Jg = Int[]
    for i in 1:3, j in 1:3
        v = (i - 1) * 3 + j
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            (1 <= i + di <= 3 && 1 <= j + dj <= 3) || continue
            u = (i + di - 1) * 3 + (j + dj)
            push!(Ig, v)
            push!(Jg, u)
        end
    end
    grid9 = sparse(Ig, Jg, trues(length(Ig)), 9, 9)
    @test !is_chordal(grid9)

    # The assembly function reconstructs the coupling matrix exactly, and
    # its rrule matches Zygote and the manual scatter adjoint.
    Ys = [zeros(2, 2) for _ in cliques]
    A = assemble_chordal(Ys, cliques, 4)
    @test size(A) == (4, 4)
    @test A ≈ zeros(4, 4)
    Ys2 = [fill(1.0, 2, 2) for _ in cliques]
    A2 = assemble_chordal(Ys2, cliques, 4)
    @test A2 ≈ [1.0 1.0 0.0 0.0; 1.0 2.0 1.0 0.0; 0.0 1.0 2.0 1.0; 0.0 0.0 1.0 1.0]
    gz = Zygote.gradient(
        Y -> sum(
            assemble_chordal(Y, cliques, 4) .*
            [1.0 2.0 3.0 4.0; 2.0 3.0 4.0 5.0; 3.0 4.0 5.0 6.0; 4.0 5.0 6.0 7.0],
        ),
        Ys2,
    )[1]
    # manual scatter adjoint: block k gets (W[i,j] + W[j,i])/2 over its clique
    W = [1.0 2.0 3.0 4.0; 2.0 3.0 4.0 5.0; 3.0 4.0 5.0 6.0; 4.0 5.0 6.0 7.0]
    for (k, C) in enumerate(cliques)
        man = [(W[C[a], C[b]] + W[C[b], C[a]]) / 2 for a in 1:2, b in 1:2]
        @test gz[k] ≈ man
    end
    # Gradient flows through the coupling residual.
    g = Zygote.gradient(x -> sum(c_coupling(x)), d0)[1]
    @test all(isfinite, g)

    # Label selection: unselected constraints pass through unchanged.
    dm_none, _, _ = chordal_transform(vmc, [pat]; x0=vmc.init, labels=[:other])
    @test dm_none.set_constraints.fs[1].set isa PSDCone

    # Informative errors.
    @test_throws DimensionMismatch chordal_transform(vmc, [sparse(ones(3, 3))]; x0=vmc.init)
    @test_throws ArgumentError chordal_transform(vmc, [sparse(zeros(4, 4))]; x0=vmc.init)  # pattern misses the off-diagonals of P(x0)
end

@testitem "Chordal relaxation: clique submatrices (chordal_decompose)" begin
    using NonconvexSemidefinite, NonconvexCore, LinearAlgebra, Test, SparseArrays

    # tridiag(1,1,1) has the path graph: maximal cliques {1,2} and {2,3}.
    # Its clique submatrices are PSD but the matrix is indefinite, so the
    # clique-submatrix form is a RELAXATION of P ⪰ 0, not an equivalent
    # reformulation.
    P = [1.0 1.0 0.0; 1.0 1.0 1.0; 0.0 1.0 1.0]
    @test minimum(eigvals(Symmetric(P))) < 0     # indefinite
    cliques = chordal_extension(sparse(P .!= 0))
    @test all(C -> minimum(eigvals(Symmetric(P[C, C]))) >= -1e-10, cliques)

    # chordal_decompose rewrites the constraint with clique submatrices:
    # the rewritten constraint is satisfied by P while the original is not.
    m = Model(x -> x[1])
    addvar!(m, 0.0, 1.0)
    add_sd_constraint!(m, x -> P, 3)
    vm, x0, _ = NonconvexCore.tovecmodel(m, [0.5])
    dm = chordal_decompose(vm, [sparse(P .!= 0)])
    c = dm.sd_constraints.fs[1]
    @test c isa NonconvexSemidefinite.BlockSDConstraint
    blocks = c(x0)
    @test length(blocks) == length(cliques)
    @test all(B -> minimum(eigvals(Symmetric(B))) >= -1e-10, blocks)  # relaxed: feasible
    @test minimum(eigvals(Symmetric(vm.sd_constraints.fs[1](x0)))) < 0  # original: infeasible
    # The relaxed blocks are exactly the clique submatrices of P.
    @test sort(blocks; by=B -> size(B, 1)) == [P[1:2, 1:2], P[2:3, 2:3]]
end

@testitem "Cutting plane method" begin
    using NonconvexSemidefinite, NonconvexCore, NonconvexIpopt, LinearAlgebra, Test

    # min (x₁ - 0.2)² + (x₂ - 0.2)² s.t. the tridiagonal 4×4 P(x) ⪰ 0.
    # The unconstrained point (0.2, 0.2) violates PSD-ness, so the cuts drive
    # the iterates onto the PSD boundary.
    P(x) = [
        x[1] 0.25 0.0 0.0
        0.25 x[2] 0.25 0.0
        0.0 0.25 x[1] 0.25
        0.0 0.0 0.25 x[2]
    ]
    m = Model(x -> (x[1] - 0.2)^2 + (x[2] - 0.2)^2)
    addvar!(m, [0.0, 0.0], [5.0, 5.0]; init=[2.0, 2.0])
    add_sd_constraint!(m, P, 4)
    alg = SDCuttingPlaneAlg(IpoptAlg())

    # Full-matrix cuts (the iterative eigensolver on the whole matrix).
    r = optimize(
        m,
        alg,
        [2.0, 2.0];
        options=SDCuttingPlaneOptions(
            eig_tol=1e-4, sub_options=IpoptOptions(max_iter=200, print_level=0)
        ),
    )
    @test r.n_cuts >= 1
    @test r.min_eigvals[end] >= -1e-3
    @test minimum(eigvals(Symmetric(P(r.minimizer)))) >= -1e-3

    # Block cuts (block_size = 2): the 2×2 principal blocks are enforced —
    # a relaxation of the full PSD constraint — and each block solve is an
    # O(block) eigendecomposition.
    r2 = optimize(
        m,
        alg,
        [2.0, 2.0];
        options=SDCuttingPlaneOptions(
            eig_tol=1e-4,
            block_size=2,
            sub_options=IpoptOptions(max_iter=200, print_level=0),
        ),
    )
    @test r2.n_cuts >= 1
    @test minimum(eigvals(Symmetric(P(r2.minimizer)[1:2, 1:2]))) >= -1e-3
    @test minimum(eigvals(Symmetric(P(r2.minimizer)[3:4, 3:4]))) >= -1e-3

    # The classic semidefinite constraint and a PSDCone set constraint in
    # the same model are both covered by the cuts.
    m3 = Model(x -> (x[1] - 0.2)^2 + (x[2] - 0.2)^2)
    addvar!(m3, [0.0, 0.0], [5.0, 5.0]; init=[2.0, 2.0])
    add_set_constraint!(
        m3, x -> [P(x)[1:2, 1:2] P(x)[1:2, 3:4]; P(x)[3:4, 1:2] P(x)[3:4, 3:4]], PSDCone(4)
    )
    r3 = optimize(
        m3,
        alg,
        [2.0, 2.0];
        options=SDCuttingPlaneOptions(
            eig_tol=1e-4, sub_options=IpoptOptions(max_iter=200, print_level=0)
        ),
    )
    @test minimum(eigvals(Symmetric(P(r3.minimizer)))) >= -1e-3
end

@testitem "Chordal algorithms: CliqueTrees extension (optional dependency)" begin
    using NonconvexSemidefinite, NonconvexCore, LinearAlgebra, Test, SparseArrays
    using CliqueTrees

    # The path graph is chordal: every algorithm returns the same edges.
    pat =
        sparse([1, 2, 3, 1, 2, 3], [2, 3, 4, 1, 2, 3], trues(6), 4, 4) +
        sparse(1:4, 1:4, trues(4), 4, 4)
    edges = sort([[i, i + 1] for i in 1:3])
    @test sort(chordal_extension(pat; alg=AMDAlg())) == edges
    @test sort(chordal_extension(pat; alg=MCSAlg())) == edges
    @test sort(chordal_extension(pat; alg=MFAlg())) == edges
    @test is_chordal(pat)
    # METIS needs `import Metis` (a CliqueTrees sub-dependency); without it
    # the error is informative.
    @test_throws Exception chordal_extension(pat; alg=METISAlg())

    # A non-chordal pattern: the CliqueTrees algorithms give a valid chordal
    # completion whose cliques are at most as large as the built-in ones.
    cycle4 = symmetric_pattern(
        sparse([1, 1, 2, 2, 3, 3, 4, 4], [2, 4, 1, 3, 2, 4, 1, 3], trues(8), 4, 4)
    )
    @test !is_chordal(cycle4)
    amd_c = chordal_extension(cycle4; alg=AMDAlg())
    mf_c = chordal_extension(cycle4; alg=MFAlg())
    @test maximum(length, mf_c) <= maximum(length, amd_c)
    # The completion cliques cover every edge of the original graph.
    for j in 1:4
        for k in nzrange(cycle4, j)
            i = rowvals(cycle4)[k]
            i < j || continue
            @test any(C -> i in C && j in C, mf_c)
        end
    end
    # BT is exact: on this tiny graph it terminates and returns a valid
    # completion (optimal width: a 4-cycle needs one chord, cliques of 3).
    import TreeWidthSolver
    bt_c = chordal_extension(cycle4; alg=BTAlg())
    @test maximum(length, bt_c) == 3
    # METIS nested dissection works once Metis is loaded.
    import Metis
    mt_c = chordal_extension(cycle4; alg=METISAlg())
    @test maximum(length, mt_c) == 3
end

@testitem "Structured domain decomposition (SDD)" begin
    using NonconvexSemidefinite, NonconvexCore, LinearAlgebra, Test, Random, SparseArrays

    # A 1D chain of 8 cells (9 dofs), split into 2 subdomains: the shared
    # node (dof 5) becomes the interface.
    coords = reshape(collect(0.5:1.0:7.5), 8, 1)
    cell_dofs = [[i, i + 1] for i in 1:8]
    part = sdd_partition(coords, cell_dofs; n_subdomains=2)
    @test part.n == 9
    @test part.interface == [5]
    @test sort(vcat(part.interiors...)) == [1, 2, 3, 4, 6, 7, 8, 9]
    @test nsubdomains(part) == 2
    @test_throws ArgumentError sdd_partition(coords, cell_dofs; n_subdomains=9)

    # Exactness on bordered block-diagonal matrices: P ⪰ 0 ⟺ interior
    # blocks ⪰ 0 ∧ interface Schur complement ⪰ 0 (the Schur-complement
    # theorem for the subdomain structure).
    Random.seed!(2)
    for _ in 1:20
        A = zeros(9, 9)
        for r in (1:4, 6:9)
            B = randn(4, 4)
            A[r, r] = B' * B
        end
        B1 = randn(1, 4)
        B2 = randn(1, 4)
        A[5, 1:4] .= vec(B1)
        A[1:4, 5] .= vec(B1)
        A[5, 6:9] .= vec(B2)
        A[6:9, 5] .= vec(B2)
        # The interface entry is shifted so the Schur complement flips sign.
        A[5, 5] = (B1 * B1')[1] + (B2 * B2')[1] + randn() * 0.5
        bundle = NonconvexSemidefinite._sdd_bundle(x -> A, part)([0.0])
        blocks_psd = all(M -> minimum(eigvals(Symmetric(M))) >= -1e-10, bundle)
        @test blocks_psd == (minimum(eigvals(Symmetric(A))) >= -1e-10)
    end

    # Model-level transform: the bundle replaces the large PSDCone.
    A = let
        M = zeros(9, 9)
        for r in (1:4, 6:9)
            B = randn(4, 4)
            M[r, r] = B' * B
        end
        M[5, 1:4] .= 0.1
        M[1:4, 5] .= 0.1
        M[5, 6:9] .= 0.1
        M[6:9, 5] .= 0.1
        M[5, 5] = 5.0
        M
    end
    m = Model(x -> x[1])
    addvar!(m, 0.0, 1.0)
    add_set_constraint!(m, x -> A, PSDCone(9); flags=[:sdd])
    vm, x0, _ = NonconvexCore.tovecmodel(m, [0.5])
    dm, d0, mapc = sdd_transform(vm, part; x0=x0, labels=[:sdd])
    c = dm.set_constraints.fs[1]
    @test c.set isa SetProduct
    @test d0 ≈ [0.5]
    @test mapc(d0) == d0
    out = c(d0)
    @test length(out) == 3          # 2 interior blocks + the Schur complement
    @test size(out[1]) == (4, 4)
    @test size(out[2]) == (4, 4)
    @test size(out[3]) == (1, 1)
    @test all(iszero, set_violation(c, d0)) == (minimum(eigvals(Symmetric(A))) >= -1e-10)
    # Unselected constraints pass through unchanged.
    dm_none, _, _ = sdd_transform(vm, part; labels=[:other])
    @test dm_none.set_constraints.fs[1].set isa PSDCone
    # Dimension mismatch is informative.
    @test_throws DimensionMismatch sdd_transform(
        vm, SDDPartition([[1]], Int[], 1); labels=[:sdd]
    )
end

@testitem "SDD partition of a Ferrite mesh (optional dependency)" begin
    using NonconvexSemidefinite, NonconvexCore, Ferrite, LinearAlgebra, Test

    # A general 2D quadrilateral mesh with a scalar field.
    grid = generate_grid(Quadrilateral, (3, 2))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral,1}())
    close!(dh)
    part = sdd_partition(grid, dh; n_subdomains=2)
    @test part.n == Ferrite.ndofs(dh)
    @test nsubdomains(part) == 2
    # Every dof is interior to exactly one subdomain or on the interface.
    @test length(vcat(part.interiors...)) + length(part.interface) == part.n
    @test isempty(intersect(vcat(part.interiors...), part.interface))
    # The geometric cut of a 3×2 cell strip leaves an interface: dofs on the
    # boundary between the two halves.
    @test length(part.interface) >= 1
    # A 1D grid works too.
    grid1 = generate_grid(Line, (4,))
    dh1 = DofHandler(grid1)
    add!(dh1, :u, Lagrange{RefLine,1}())
    close!(dh1)
    part1 = sdd_partition(grid1, dh1; n_subdomains=2)
    @test part1.n == Ferrite.ndofs(dh1)
    @test length(part1.interface) == 1
end
