@testitem "Block relaxation of PSD set constraints" begin
    using NonconvexSemidefinite, NonconvexCore, LinearAlgebra, Test, SparseArrays

    # tridiag(1, 1, 1) is indefinite, but every 2×2 principal block of the
    # path pattern is PSD: the block relaxation accepts a point the full
    # PSD cone rejects.
    P(x) = [x[1] 1.0 0.0; 1.0 x[2] 1.0; 0.0 1.0 x[3]]
    mc = Model(x -> sum(x .^ 2))
    addvar!(mc, zeros(3), fill(5.0, 3); init=fill(1.0, 3))
    ncalls = Ref(0)
    bundle = x -> begin
        ncalls[] += 1
        return (sum(x) - 3.0, P(x))
    end
    add_set_constraint!(mc, bundle, SetProduct(LessThan(0.0), PSDCone(3)); flags=[:buck])
    vmc, x0, _ = NonconvexCore.tovecmodel(mc, fill(1.0, 3))
    @test any(>(0.0), NonconvexCore.set_violation(vmc.set_constraints.fs[1], x0))

    cliques = chordal_extension(sparse(P(x0) .!= 0))
    @test sort(cliques) == sort([[1, 2], [2, 3]])
    ncalls[] = 0
    dm = block_relaxation(vmc, cliques)
    @test length(dm.set_constraints.fs) == 1
    c = dm.set_constraints.fs[1]
    # One SetProduct: the kept volume component plus one PSDCone per block.
    @test c.set isa SetProduct
    comps = NonconvexCore.components(c.set)
    @test length(comps) == 3
    @test comps[1] isa LessThan
    @test all(s -> s isa PSDCone && s.dim == 2, comps[2:3])
    # A single bundle evaluation produces all three component values.
    v = NonconvexCore.set_value(c, x0)
    @test ncalls[] == 1
    @test length(v) == 3
    @test v[1] ≈ 0.0
    @test v[2] ≈ [1.0 1.0; 1.0 1.0]
    @test v[3] ≈ [1.0 1.0; 1.0 1.0]
    # Feasible in the relaxation at x0 (2×2 blocks PSD, volume ok) but
    # infeasible in the original model (P indefinite).
    @test all(iszero, NonconvexCore.set_violation(c, x0))
    @test any(>(0.0), NonconvexCore.set_violation(vmc.set_constraints.fs[1], x0))
    # The relaxation rejects nothing the original constraint accepts on the
    # PSD component: P(2, 2, 2) ⪰ 0, so its blocks are PSD.
    vp = NonconvexCore.set_value(c, [2.0, 2.0, 2.0])
    @test all(B -> isposdef(Symmetric(B)), vp[2:3])
    # The rewritten model keeps the same variables (no auxiliaries).
    @test dm.box_min == vmc.box_min
    @test dm.box_max == vmc.box_max
    @test dm.set_variables == vmc.set_variables

    # Labels: unselected constraints pass through unchanged.
    dm_none = block_relaxation(vmc, cliques; labels=[:other])
    comps_none = NonconvexCore.components(dm_none.set_constraints.fs[1].set)
    @test any(s -> s isa PSDCone && s.dim == 3, comps_none)

    # Overlapping blocks are allowed (FEM element blocks overlap on shared
    # nodes).
    dm_ov = block_relaxation(vmc, [[1, 2], [2, 3], [1, 3]])
    comps_ov = NonconvexCore.components(dm_ov.set_constraints.fs[1].set)
    @test length(comps_ov) == 4
    @test all(s -> s isa PSDCone && s.dim == 2, comps_ov[2:end])

    # Informative errors: empty collection, out-of-range indices, wrong
    # number of collections.
    @test_throws ArgumentError block_relaxation(vmc, Vector{Vector{Int}}[])
    @test_throws ArgumentError block_relaxation(vmc, [[1, 4]])
    @test_throws ArgumentError block_relaxation(vmc, [[[1, 2]], [[2, 3]]])

    # The relaxation works with the chordal cliques of a larger pattern: the
    # 4-node path decomposes into {1,2}, {2,3}, {3,4}.
    P4(x) = [
        x[1] 0.5 0.0 0.0
        0.5 x[2] 0.5 0.0
        0.0 0.5 x[3] 0.5
        0.0 0.0 0.5 x[4]
    ]
    mc4 = Model(x -> sum(x))
    addvar!(mc4, zeros(4), fill(5.0, 4); init=fill(1.0, 4))
    add_set_constraint!(mc4, P4, PSDCone(4))
    vmc4, x04, _ = NonconvexCore.tovecmodel(mc4, fill(1.0, 4))
    cl4 = chordal_extension(sparse(P4(x04) .!= 0))
    @test sort(cl4) == sort([[1, 2], [2, 3], [3, 4]])
    dm4 = block_relaxation(vmc4, cl4)
    c4 = dm4.set_constraints.fs[1]
    @test all(s -> s isa PSDCone && s.dim == 2, NonconvexCore.components(c4.set))
    @test all(iszero, NonconvexCore.set_violation(c4, x04))
end

@testitem "Ferrite element blocks for the element-based relaxation" begin
    using Ferrite, NonconvexSemidefinite, NonconvexCore, Test, LinearAlgebra

    grid = generate_grid(Quadrilateral, (3, 2), Vec((0.0, 0.0)), Vec((3.0, 2.0)))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral,1}()^2)
    close!(dh)
    blocks = NonconvexSemidefinite.element_blocks(dh)
    @test blocks == NonconvexSemidefinite.element_blocks(grid, dh)
    @test length(blocks) == Ferrite.getncells(grid)
    @test blocks[1] == collect(Ferrite.celldofs(dh, 1))
    @test all(b -> length(b) == 8, blocks)
    @test all(b -> all(d -> 1 <= d <= ndofs(dh), b), blocks)
    # Neighbouring cells share dofs (the blocks overlap).
    @test !isdisjoint(blocks[1], blocks[2])

    # End-to-end: the element-based relaxation of an assembled PSD bundle on
    # a two-cell mesh. P(x) = tridiag(x; 0.4) is PSD at x = 1, so its
    # per-element principal blocks are PSD as well.
    g2 = generate_grid(Quadrilateral, (2, 1), Vec((0.0, 0.0)), Vec((2.0, 1.0)))
    dh2 = DofHandler(g2)
    add!(dh2, :u, Lagrange{RefQuadrilateral,1}()^2)
    close!(dh2)
    n = ndofs(dh2)
    Pfun = x -> begin
        P = zeros(n, n)
        for i in 1:n
            P[i, i] = x[i]
        end
        for i in 1:(n - 1)
            P[i, i + 1] = 0.4
            P[i + 1, i] = 0.4
        end
        return P
    end
    mc = Model(x -> sum(x))
    addvar!(mc, zeros(n), fill(5.0, n); init=fill(1.0, n))
    add_set_constraint!(mc, Pfun, PSDCone(n))
    vmc, x0, _ = NonconvexCore.tovecmodel(mc, fill(1.0, n))
    @test all(iszero, NonconvexCore.set_violation(vmc.set_constraints.fs[1], x0))

    dm = NonconvexSemidefinite.block_relaxation(
        vmc, NonconvexSemidefinite.element_blocks(dh2)
    )
    c = dm.set_constraints.fs[1]
    comps = NonconvexCore.components(c.set)
    # One PSDCone per cell (8 dofs each).
    @test length(comps) == 2
    @test all(s -> s isa PSDCone && s.dim == 8, comps)
    v = NonconvexCore.set_value(c, x0)
    # The evaluated blocks are exactly the principal submatrices of P on the
    # cell dofs, and they are PSD at x0.
    blks2 = NonconvexSemidefinite.element_blocks(dh2)
    for (B, C) in zip(v, blks2)
        @test B ≈ Pfun(x0)[C, C]
        @test isposdef(Symmetric(B))
    end
    @test all(iszero, NonconvexCore.set_violation(c, x0))
end
