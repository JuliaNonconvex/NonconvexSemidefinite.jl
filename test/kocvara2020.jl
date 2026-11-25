# Reproduction of Kočvara's "Decomposition of arrow type positive
# semidefinite matrices with application to topology optimization"
# (Math. Program. 190:105–134, 2021; arXiv:1911.09412): Example 1 and
# Theorem 2.3.
#
# Example 1: a 4×4 mesh of Q1 quadrilaterals (16 elements, 25 nodes), the
# left column of nodes fixed (eliminated), 40 free degrees of freedom
# numbered column-wise (bottom to top within each column, 2 dofs per node),
# elements numbered column-wise. The four subdomains are the 2×2 blocks of
# elements {1,2,5,6} (bottom-left), {3,4,7,8} (top-left), {9,10,13,14}
# (bottom-right), {11,12,15,16} (top-right).
#
# Note: the paper's Example 1 lists I₂ and I₃ swapped relative to its own
# element sets; the reading used here (subdomain 2 = top-left, 3 =
# bottom-right) is the one consistent with all six intersection sets
# I_{k,ℓ}, the element sets and the "two constraints of order 13, two of
# order 19" statement. With it the subdomain dof sets are
#
#     I₁ = {1..6, 11..16},             I₂ = {5..10, 15..20},
#     I₃ = {11..16, 21..26, 31..36},   I₄ = {15..20, 25..30, 35..40},
#
# and the interfaces exactly the paper's
#
#     I₁,₂ = {5,6,15,16}, I₁,₃ = {11..16}, I₁,₄ = {15,16},
#     I₂,₃ = {15,16},     I₂,₄ = {15..20}, I₃,₄ = {15,16,25,26,35,36}.

@testitem "Kočvara 2020: Example 1 reproduction (arrow decomposition)" begin
    using NonconvexSemidefinite, NonconvexCore, NonconvexIpopt
    using LinearAlgebra, Test, SparseArrays, Random

    # -- The mesh of Example 1 -------------------------------------------
    # Nodes (i, j), i = 1..5 columns (left→right), j = 1..5 rows
    # (bottom→top). The left column i = 1 is fixed. Free dofs are numbered
    # column-wise: column 2 → dofs 1..10, column 3 → 11..20, column 4 →
    # 21..30, column 5 → 31..40 (2 dofs per node, bottom to top).
    nodeid(i, j) = (j - 1) * 5 + i
    freenodes = Int[]
    for ii in 2:5, jj in 1:5
        push!(freenodes, nodeid(ii, jj))
    end
    fdof = Dict(node => [2t - 1, 2t] for (t, node) in enumerate(freenodes))
    # elements numbered column-wise: elem(i, j) = (i - 1) * 4 + j
    elem(i, j) = (i - 1) * 4 + j
    # the dofs of element e = elem(i, j), spanning nodes (i..i+1, j..j+1)
    # (fixed nodes contribute nothing: they are eliminated)
    function celldofs(e)
        i = (e - 1) ÷ 4 + 1
        j = (e - 1) % 4 + 1
        ds = Int[]
        for jj in j:(j + 1), ii in i:(i + 1)
            node = (jj - 1) * 5 + ii
            haskey(fdof, node) && append!(ds, fdof[node])
        end
        return ds
    end

    # Q1 plane-stress element stiffness matrix (E = 1, ν = 0.3, unit square,
    # 2×2 Gauss integration), 8×8 in the node order
    # (1,1),(2,1),(1,2),(2,2) × (x, y).
    function q1_ps_Ke(E, ν)
        D = E / (1 - ν^2) * [1.0 ν 0.0; ν 1.0 0.0; 0.0 0.0 (1 - ν) / 2]
        Ke = zeros(8, 8)
        g = 1 / sqrt(3)
        for (ξ, w1) in ((-g, 1.0), (g, 1.0)), (η, w2) in ((-g, 1.0), (g, 1.0))
            dN = [
                -(1 - η) / 4 (1 - η) / 4 -(1 + η) / 4 (1 + η) / 4;
                -(1 - ξ) / 4 -(1 + ξ) / 4 (1 - ξ) / 4 (1 + ξ) / 4
            ]
            B = zeros(3, 8)
            for a in 1:4
                B[1, 2a - 1] = dN[1, a]
                B[2, 2a] = dN[2, a]
                B[3, 2a - 1] = dN[2, a]
                B[3, 2a] = dN[1, a]
            end
            Ke .+= w1 * w2 .* (B' * D * B)
        end
        return Ke
    end
    K0 = q1_ps_Ke(1.0, 0.3)

    # Global assembly: K(x) = Σ_e x_e K_e (the fixed dofs are eliminated).
    # Mutation-free (comprehensions + sums) so Zygote can differentiate.
    # `l2g` does not depend on the design: precompute it once, so the
    # differentiated assemblers never trace its splatted `vcat` (Zygote
    # rejects the push!-based lowering of `vcat(args...)`).
    function l2g(e)
        i = (e - 1) ÷ 4 + 1
        j = (e - 1) % 4 + 1
        return vcat(
            [
                haskey(fdof, nodeid(ii, jj)) ? fdof[nodeid(ii, jj)] : [0, 0] for
                jj in j:(j + 1) for ii in i:(i + 1)
            ]...,
        )
    end
    L2G = [l2g(e) for e in 1:16]
    function assemble_K(x)
        parts = [
            [
                sum(
                    x[e] * K0[a, b] * (L2G[e][a] == r && L2G[e][b] == c) for
                    a in 1:8, b in 1:8
                ) for r in 1:40, c in 1:40
            ] for e in 1:16
        ]
        return sum(parts)
    end

    # -- The subdomains of Example 1 -------------------------------------
    # (subdomain 2 = top-left, 3 = bottom-right — see the header note).
    D1 = [elem(i, j) for i in 1:2, j in 1:2][:]
    D2 = [elem(i, j) for i in 1:2, j in 3:4][:]
    D3 = [elem(i, j) for i in 3:4, j in 1:2][:]
    D4 = [elem(i, j) for i in 3:4, j in 3:4][:]
    @test sort(D1) == [1, 2, 5, 6]
    @test sort(D2) == [3, 4, 7, 8]
    @test sort(D3) == [9, 10, 13, 14]
    @test sort(D4) == [11, 12, 15, 16]
    I1 = vcat(1:6, 11:16)
    I2 = vcat(5:10, 15:20)
    I3 = vcat(11:16, 21:26, 31:36)
    I4 = vcat(15:20, 25:30, 35:40)
    I = [I1, I2, I3, I4]
    # the listed Iₖ are exactly the free dofs of the subdomain element sets
    for (Dk, Ik) in zip((D1, D2, D3, D4), I)
        @test sort(Ik) == sort(unique!(vcat((celldofs(e) for e in Dk)...)))
    end

    # The interfaces, exactly as in the paper.
    it12 = ArrowIntersection(1, 2, [5, 6, 15, 16])
    it13 = ArrowIntersection(1, 3, collect(11:16))
    it14 = ArrowIntersection(1, 4, [15, 16])
    it23 = ArrowIntersection(2, 3, [15, 16])
    it24 = ArrowIntersection(2, 4, collect(15:20))
    it34 = ArrowIntersection(3, 4, [15, 16, 25, 26, 35, 36])
    its = [it12, it13, it14, it23, it24, it34]
    for it in its
        @test sort(it.dofs) == sort(intersect(I[it.k], I[it.l]))
    end
    # The paper: "g1,4, g2,3 ∈ ℝ², g1,2 ∈ ℝ⁴, g1,3, g2,4, g3,4 ∈ ℝ⁶".
    @test sort(length.([it.dofs for it in its])) == sort([4, 6, 2, 2, 6, 6])
    @test sum(length, (it.dofs for it in its)) == 26

    # Subdomain stiffness assemblers (the elements of the subdomain only)
    # and loads: a unit downward load at the bottom-right corner node
    # (column 5, row 1 → dof 32; it lies only in subdomain 3, so the load
    # split is trivial).
    function assemble_subK(x, Dk, Ik)
        parts = [
            [
                sum(
                    x[e] * K0[a, b] * (L2G[e][a] == Ik[p] && L2G[e][b] == Ik[q]) for
                    a in 1:8, b in 1:8
                ) for p in 1:length(Ik), q in 1:length(Ik)
            ] for e in Dk
        ]
        return sum(parts)
    end
    D = [D1, D2, D3, D4]
    Kfuns = [x -> assemble_subK(x, D[k], I[k]) for k in 1:4]
    f = zeros(40)
    f[32] = -1.0
    fs = [f[Ik] for Ik in I]
    subdomains = [ArrowSubdomain(Kfuns[k], fs[k], I[k]) for k in 1:4]

    # -- The SDP (11) of the paper ---------------------------------------
    # min γ s.t. Z(x, γ) = [K(x) f; fᵀ γ] ⪰ 0, Σxᵢ ≤ V, x̄ ≤ x ≤ x̄.
    V, xlb, xub = 8.0, 1e-4, 1.0
    m = Model(x -> x[17])
    addvar!(m, fill(xlb, 16), fill(xub, 16); init=fill(0.5, 16))
    addvar!(m, 0.0, 1e3; init=2.0)
    add_ineq_constraint!(m, x -> sum(x[1:16]) - V)
    add_set_constraint!(m, x -> begin
        K = assemble_K(x[1:16])
        return [K f; f' x[17]]
    end, PSDCone(41); flags=[:arrow])
    vm, x0, _ = NonconvexCore.tovecmodel(m, m.init)

    # -- The arrow decomposition (19) of the paper ------------------------
    dm, d0, mapc = arrow_decompose(vm, subdomains, its; x0=x0, labels=[:arrow])
    @test length(dm.box_min) == 17 + 4 + 26
    @test length(dm.set_constraints.fs) == 4
    # "four matrix constraints, two of order 13 and two of order 19".
    orders = sort([c.set.dim for c in dm.set_constraints.fs])
    @test orders == [13, 13, 19, 19]
    @test mapc(d0) ≈ x0

    # The exact matrix-sum identity Σₖ E(Iₖ∪{γ})ᵀZ⁽ᵏ⁾_BE(Iₖ∪{γ}) = Z(x, Σₖγₖ):
    # the decomposed blocks live on their own index sets and rebuild the
    # original matrix when embedded.
    function embed(Z, Ik)
        Zfull = zeros(41, 41)
        idx = [Ik; 41]
        Zfull[idx, idx] .= Z
        return Zfull
    end
    Random.seed!(11)
    xr = vcat(fill(0.5, 16), 2.0)
    v = vcat(xr, fill(2.0 / 4, 4), randn(26))
    Zsum = sum(embed(c(v), I[k]) for (k, c) in enumerate(dm.set_constraints.fs))
    Zorig = [assemble_K(xr[1:16]) f; f' sum(v[18:21])]
    @test Zsum ≈ Zorig atol = 1e-12

    # -- Theorem 2.3 exactness -------------------------------------------
    # (⇐) PSD of every decomposed block implies PSD of Z (a sum of PSD
    # matrices). (⇒) given Z(x, γ) ⪰ 0 with K(x) ≻ 0, the paper's
    # construction (X = K⁻¹f, the fictitious loads from K⁽ᵏ⁾u − f⁽ᵏ⁾,
    # γₖ = uᵀK⁽ᵏ⁾u) yields a feasible decomposition.
    posin(Ik, ds) = [findfirst(==(d), Ik) for d in ds]
    for _ in 1:10
        xt = vcat(0.2 .+ 0.8 .* rand(16), 0.0)
        K = assemble_K(xt[1:16])
        isposdef(Symmetric(K)) || continue
        u = K \ f
        γ = f'u + rand() * 5.0            # some feasible γ ≥ fᵀu
        # construct the fictitious loads sequentially (k = 1, 2, 3)
        gvals = Dict{Tuple{Int,Int},Vector{Float64}}()
        for it in its
            gvals[(it.k, it.l)] = zeros(length(it.dofs))
        end
        for k in 1:3
            r = Kfuns[k](xt[1:16]) * u[I[k]] - fs[k]   # K⁽ᵏ⁾u − f⁽ᵏ⁾
            for it in its
                # r⁽ᵏ⁾ + Σ_{ℓ<k} g_{ℓ,k} = Σ_{ℓ>k} g_{k,ℓ} (the paper's
                # eq. (17) with the incoming loads moved to the left)
                it.l == k && (r[posin(I[k], it.dofs)] .+= gvals[(it.k, it.l)])
            end
            # assign each interface dof of Iₖ to its first outgoing pair
            assigned = falses(length(I[k]))
            for it in its
                it.k == k || continue
                for (a, d) in enumerate(it.dofs)
                    q = findfirst(==(d), I[k])
                    assigned[q] && continue
                    gvals[(it.k, it.l)][a] = r[q]
                    assigned[q] = true
                end
            end
            # the interior residual vanishes (eq. (4) of the paper)
            @test all(abs(r[q]) < 1e-10 for q in 1:length(I[k]) if !assigned[q])
        end
        γk = [u[I[k]]' * Kfuns[k](xt[1:16]) * u[I[k]] for k in 1:3]
        push!(γk, γ - sum(γk))
        vv = vcat(xt, γk, reduce(vcat, [gvals[(it.k, it.l)] for it in its]))
        @test sum(γk) ≈ γ
        for (k, c) in enumerate(dm.set_constraints.fs)
            @test minimum(eigvals(Symmetric(c(vv)))) >= -1e-8
        end
        Zb = sum(embed(c(vv), I[k]) for (k, c) in enumerate(dm.set_constraints.fs))
        @test minimum(eigvals(Symmetric(Zb))) >= -1e-8
        @test Zb ≈ [assemble_K(xt[1:16]) f; f' sum(γk)] atol = 1e-10
    end

    # -- The decomposed problem has the same optimum as the original ------
    # (the equivalence the paper's numerical tables rely on; both sides
    # solved by the same cutting-plane algorithm).
    alg = SDCuttingPlaneAlg(IpoptAlg())
    ropt = NonconvexCore.optimize(
        vm,
        alg,
        x0;
        options=SDCuttingPlaneOptions(
            eig_tol=1e-4, sub_options=IpoptOptions(max_iter=200, print_level=0)
        ),
    )
    dopt = NonconvexCore.optimize(
        dm,
        alg,
        d0;
        options=SDCuttingPlaneOptions(
            eig_tol=1e-4, sub_options=IpoptOptions(max_iter=200, print_level=0)
        ),
    )
    @test dopt.minimum ≈ ropt.minimum rtol = 1e-2 atol = 1e-3
    xd = mapc(dopt.minimizer)
    @test minimum(eigvals(Symmetric([assemble_K(xd[1:16]) f; f' xd[17]]))) >= -1e-4
end
