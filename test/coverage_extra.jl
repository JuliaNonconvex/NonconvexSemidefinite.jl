# Additional coverage tests for NonconvexSemidefinite.
#
# Focus: corner-case / error paths of the projection rrule, chordal pattern
# validation, barrier option/selection validation, the adaptive barrier
# schedule and duality-gap termination, SDPQuadAlg layout validation and
# low-rank re-parameterization, PSDMatrixView / add_psd_variable! error paths,
# and the `_optimize_precheck` no-ops of the meta-algorithms.
#
# Small numeric checks only; no heavy optimization. A trivial identity
# sub-solver (`DummyAlg`) drives the meta-algorithm loops.

@testitem "project_psd: corner cases and rrule branches" begin
    using NonconvexSemidefinite, Test, LinearAlgebra, ChainRulesCore

    # Non-symmetric input is symmetrized before projection.
    B = [0.0 2.0; 0.0 0.0]
    ZB = project_psd(B)
    @test ZB ≈ ZB'
    @test ZB ≈ fill(0.5, 2, 2) atol = 1e-8  # project of [[0,1],[1,0]] is all-0.5

    # Idempotence on a PSD matrix, Moreau decomposition and infeasibility
    # measure with known eigenvalues.
    A = [2.0 0.5; 0.5 1.0]
    @test project_psd(A) ≈ A
    @test project_psd_neg(A) ≈ zeros(2, 2) atol = 1e-8
    @test psd_infeasibility(A) ≈ 0 atol = 1e-8
    M = diagm([-2.0, -1.0, 0.5])
    @test psd_infeasibility(M) ≈ sqrt(5.0) atol = 1e-8

    # --- rrule corner cases (Lewis spectral differential) ---

    # Repeated positive eigenvalues: the function is smooth (identity) there
    # and G is all ones → pullback is the symmetrized cotangent itself.
    _, pb = ChainRulesCore.rrule(project_psd, diagm([1.0, 1.0]))
    @test pb(ones(2, 2))[2] ≈ ones(2, 2)

    # Repeated negative eigenvalues: fully inactive, zero pullback.
    _, pb2 = ChainRulesCore.rrule(project_psd, diagm([-1.0, -1.0]))
    @test pb2(ones(2, 2))[2] ≈ zeros(2, 2)

    # Repeated zero eigenvalues: the subgradient choice is zero.
    _, pb3 = ChainRulesCore.rrule(project_psd, diagm([0.0, 0.0]))
    @test pb3(ones(2, 2))[2] ≈ zeros(2, 2)

    # Mixed zero / positive eigenvalue: G = [0 1; 1 1].
    _, pb4 = ChainRulesCore.rrule(project_psd, diagm([0.0, 1.0]))
    @test pb4(ones(2, 2))[2] ≈ [0.0 1.0; 1.0 1.0]

    # Mixed negative / positive eigenvalue: smooth off-diagonal quotient
    # (θ1 - θ2) / (λ1 - λ2) = (0 - 2) / (-1 - 2) = 2/3.
    _, pb5 = ChainRulesCore.rrule(project_psd, diagm([-1.0, 2.0]))
    @test pb5(ones(2, 2))[2] ≈ [0.0 2 / 3; 2 / 3 1.0]

    # The cotangent is symmetrized before the G-product.
    _, pb6 = ChainRulesCore.rrule(project_psd, diagm([0.0, 1.0]))
    @test pb6([0.0 2.0; 0.0 0.0])[2] ≈ [0.0 1.0; 1.0 0.0]

    # Consistency with finite differences where the function is smooth
    # (repeated positive eigenvalues).
    E = [0.3 0.1; 0.1 -0.2]
    _, pbfd = ChainRulesCore.rrule(project_psd, diagm([1.0, 1.0]))
    Δfd = pbfd(ones(2, 2))[2]
    h = 1e-6
    fd =
        (project_psd(diagm([1.0, 1.0]) + h * E) - project_psd(diagm([1.0, 1.0]) - h * E)) /
        (2h)
    @test sum(Δfd .* E) ≈ sum(fd .* ones(2, 2)) rtol = 1e-4

    # Zero cotangent does not throw (AbstractZero short-circuit).
    _, pbz = ChainRulesCore.rrule(project_psd, A)
    cotz = pbz(ChainRulesCore.ZeroTangent())
    @test cotz[1] isa ChainRulesCore.AbstractZero
    @test cotz[2] isa ChainRulesCore.AbstractZero
end

@testitem "decompress_symmetric and rearrange_x rrule corners" begin
    using NonconvexSemidefinite, Test, LinearAlgebra, ChainRulesCore

    xL = [1.0, 2.0, 3.0]
    xD = [4.0, 5.0, 6.0]
    L, D = NonconvexSemidefinite.rearrange_x(xL, xD)
    @test L ≈ [0.0 0.0 0.0; 1.0 0.0 0.0; 2.0 3.0 0.0]
    @test D ≈ diagm([4.0, 5.0, 6.0])
    @test NonconvexSemidefinite.lowertriangind(zeros(3, 3)) == [2, 3, 6]

    @test decompress_symmetric(xL, xD) ≈ [4.0 1.0 2.0; 1.0 5.0 3.0; 2.0 3.0 6.0]
    @test NonconvexSemidefinite.decompress_symmetric(L, D) ≈ decompress_symmetric(xL, xD)
    @test decompress_symmetric([1.0], [4.0, 5.0]) ≈ [4.0 1.0; 1.0 5.0]
    @test decompress_symmetric([1, 2, 3], [4, 5, 6]) isa Symmetric

    # rrule with full (matrix, matrix) cotangents.
    _, pb = ChainRulesCore.rrule(NonconvexSemidefinite.rearrange_x, xL, xD)
    cot = pb((ones(3, 3), ones(3, 3)))
    @test cot[1] isa ChainRulesCore.AbstractZero  # NoTangent for the function
    @test cot[2] == ones(3)                       # strict lower entries of ΔL
    @test cot[3] == ones(3)                       # diagonal of ΔD

    # Nonzero ΔL with ΔD = zeros: only the lower part contributes.
    cotL = pb((ones(3, 3), zeros(3, 3)))
    @test cotL[2] == ones(3)
    @test cotL[3] == zeros(3)

    # Mixed AbstractZero cotangents: documented behaviour returns zero for
    # both tangent components as soon as one of them is an AbstractZero.
    cot2 = pb((ones(3, 3), ChainRulesCore.ZeroTangent()))
    @test cot2[2] isa ChainRulesCore.AbstractZero
    @test cot2[3] isa ChainRulesCore.AbstractZero
    cot3 = pb((ChainRulesCore.ZeroTangent(), ones(3, 3)))
    @test cot3[2] isa ChainRulesCore.AbstractZero
    @test cot3[3] isa ChainRulesCore.AbstractZero
end

@testitem "chordal: pattern validation errors" begin
    using NonconvexSemidefinite, NonconvexCore, Test, LinearAlgebra, SparseArrays

    # symmetric_pattern drops stored zeros but always keeps the diagonal.
    Pz = sparse([1, 2], [2, 2], [0.0, 1.0], 2, 2)
    @test Matrix(symmetric_pattern(Pz)) == Matrix{Bool}(I, 2, 2)

    # A diagonal pattern is already chordal: n singleton cliques.
    cl = chordal_extension(sparse(1.0 * I, 4, 4))
    @test length(cl) == 4
    @test all(length(c) == 1 for c in cl)
    @test sort(vcat(cl...)) == [1, 2, 3, 4]

    # A dense pattern has a single clique covering every vertex.
    cld = chordal_extension(sparse(trues(4, 4)))
    @test length(cld) == 1 && cld[1] == [1, 2, 3, 4]

    # Non-square patterns are rejected by checksquare.
    @test_throws DimensionMismatch chordal_extension(sparse(ones(2, 3)))

    # _maximal_cliques: subsumed and duplicated cliques are dropped.
    @test NonconvexSemidefinite._maximal_cliques([[1], [1, 2], [1]]) == [[1, 2]]
    @test NonconvexSemidefinite._maximal_cliques([[1, 2], [1, 2]]) == [[1, 2]]

    # aggregate_sparsity: wrong output length → ArgumentError.
    @test_throws ArgumentError aggregate_sparsity(x -> zeros(2, 2), ones(3), 3)
    @test_throws ArgumentError aggregate_sparsity(x -> x[1:2], ones(3), 3)

    # Full dependence on every entry → dense pattern.
    @test all(Matrix(aggregate_sparsity(x -> [x[1] x[2]; x[2] x[3]], ones(3), 2)))
    # Constant matrix output → no dependence, only the diagonal remains.
    @test Matrix(aggregate_sparsity(x -> ones(2, 2), ones(3), 2)) == Matrix{Bool}(I, 2, 2)

    # Model with one 4×4 tridiagonal sd constraint.
    P = [2.0 0.5 0.0 0.0; 0.5 2.0 0.5 0.0; 0.0 0.5 2.0 0.5; 0.0 0.0 0.5 2.0]
    function sd_vm(Pmat)
        sd = NonconvexCore.SDConstraint(x -> Pmat, 4)
        return NonconvexCore.VecModel(
            NonconvexCore.Objective(x -> 0.0),
            NonconvexCore.VectorOfFunctions(()),
            NonconvexCore.VectorOfFunctions(()),
            NonconvexCore.VectorOfFunctions([sd]),
            [-1.0],
            [1.0],
            [0.0],
            falses(1),
        )
    end
    vm = sd_vm(P)

    # Single pattern (not wrapped in a vector) with one constraint. The
    # decomposition produces one BlockSDConstraint whose function returns
    # every clique block from a single evaluation.
    dm = chordal_decompose(vm, sparse(P .!= 0))
    @test length(dm.sd_constraints.fs) == 1
    @test dm.sd_constraints.fs[1] isa NonconvexSemidefinite.BlockSDConstraint
    @test NonconvexSemidefinite.getdims(dm.sd_constraints.fs[1]) == [2, 2, 2]

    # Pattern-count mismatch.
    err = try
        chordal_decompose(vm, [sparse(P .!= 0), sparse(P .!= 0)])
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("1 semidefinite constraint", sprint(showerror, err))

    # Wrong-size (square) pattern.
    err2 = try
        chordal_decompose(vm, sparse(ones(2, 2)))
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("size 2", sprint(showerror, err2))

    # Wrong-shape (non-square) pattern.
    @test_throws DimensionMismatch chordal_decompose(vm, sparse(ones(2, 3)))

    # Probing (nothing pattern) without an initial solution.
    @test_throws ArgumentError chordal_decompose(vm, [nothing])

    # A pattern that misses a nonzero entry of P(x0) is rejected.
    Pbad = copy(P)
    Pbad[3, 1] = 0.3
    Pbad[1, 3] = 0.3
    vmbad = sd_vm(Pbad)
    err3 = try
        chordal_decompose(vmbad, sparse(P .!= 0); x0=[0.0])
    catch e
        e
    end
    @test err3 isa ArgumentError
    @test occursin("outside the supplied sparsity pattern", sprint(showerror, err3))

    # The same pattern is accepted when it covers every entry of P(x0).
    dm_ok = chordal_decompose(vm, sparse(P .!= 0); x0=[0.0])
    @test length(dm_ok.sd_constraints.fs) == 1

    # The matrix-valued function is evaluated exactly once per call.
    calls = Ref(0)
    Pcounted(x) = (calls[] += 1; copy(P))
    vmc = NonconvexCore.VecModel(
        NonconvexCore.Objective(x -> 0.0),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions([NonconvexCore.SDConstraint(Pcounted, 4)]),
        [-1.0],
        [1.0],
        [0.0],
        falses(1),
    )
    dmc = chordal_decompose(vmc, sparse(P .!= 0))
    calls[] = 0
    blocks = dmc.sd_constraints.fs[1]([0.5])
    @test calls[] == 1
    @test blocks isa AbstractVector
    @test length(blocks) == 3
    @test all(B -> size(B) == (2, 2), blocks)
    # The blocks are exactly the clique slices of the single evaluation.
    P0 = copy(P)
    expected = [P0[1:2, 1:2], P0[2:3, 2:3], P0[3:4, 3:4]]
    @test Set(vec.(blocks)) == Set(vec.(expected))
end

@testitem "modeling: PSDMatrixView and add_psd_variable! error paths" begin
    using NonconvexSemidefinite, NonconvexCore, Test, LinearAlgebra

    # PSDMatrixView basics.
    M = PSDMatrixView(1, 2)  # nvars = 3, full model length = 4
    @test size(M) == (2, 2)
    @test size(M, 1) == 2 && size(M, 2) == 2 && size(M, 3) == 1
    @test M([0.0, 1.0, 1.0]) ≈ Matrix{Float64}(I, 2, 2)      # chunk alone
    @test M([9.0, 0.0, 1.0, 1.0]) ≈ Matrix{Float64}(I, 2, 2)  # offset access

    # Wrong vector length: neither the chunk length nor the full model length.
    err = try
        M(zeros(2))
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("length 2", sprint(showerror, err))

    # 3×3 view decompression.
    M0 = PSDMatrixView(0, 3)
    @test M0([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) ≈ [3.0 0.0 1.0; 0.0 4.0 2.0; 1.0 2.0 5.0]

    # _psd_chunk rejects wrong-sized init matrices for every model type.
    m = Model()
    @test_throws ArgumentError add_psd_variable!(m, 3; init=zeros(2, 2))
    vm = NonconvexCore.VecModel(
        NonconvexCore.Objective(x -> 0.0),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions(()),
        Float64[],
        Float64[],
        Float64[],
        falses(0),
    )
    @test_throws ArgumentError add_psd_variable!(vm, 3; init=zeros(2, 2))
    dm = DictModel()
    @test_throws ArgumentError add_psd_variable!(dm, :P, 3; init=zeros(2, 2))
    @test_throws ArgumentError add_psd_variable!(dm, "P", 3; init=zeros(2, 2))

    # VecModel good path: chunk appended with non-negative diagonal bounds.
    Mv = add_psd_variable!(vm, 2)
    @test Mv(vm.init) ≈ Matrix{Float64}(I, 2, 2)
    @test vm.box_min == [-Inf, 0.0, 0.0]
    @test all(isinf, vm.box_max)

    # Model good paths: default init and offset behind a plain variable.
    m2 = Model()
    M2 = add_psd_variable!(m2, 2)
    @test M2([0.0, 1.0, 1.0]) ≈ Matrix{Float64}(I, 2, 2)
    m3 = Model()
    addvar!(m3, -1.0, 1.0)
    M3 = add_psd_variable!(m3, 2)
    @test M3([0.5, 0.0, 1.0, 1.0]) ≈ Matrix{Float64}(I, 2, 2)

    # DictModel paths: Symbol and String keys, offset behind another variable.
    dm2 = DictModel()
    addvar!(dm2, :a, [0.0], [1.0])
    Md = add_psd_variable!(dm2, :P, 2)
    @test Md([0.5, 0.0, 1.0, 1.0]) ≈ Matrix{Float64}(I, 2, 2)
    @test Md([0.0, 1.0, 1.0]) ≈ Matrix{Float64}(I, 2, 2)
    Ms = add_psd_variable!(dm2, "Q", 2)
    @test Ms([0.5, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]) ≈ Matrix{Float64}(I, 2, 2)
end

@testitem "barrier: option, selection and strict-feasibility validation" begin
    using NonconvexSemidefinite, NonconvexCore, Test, LinearAlgebra

    struct DummyAlg <: NonconvexCore.AbstractOptimizer end

    alg = BarrierAlg(sub_alg=DummyAlg())

    # Invalid selection entry types: rejected at construction ...
    @test_throws Union{TypeError,MethodError} BarrierSelection(ineq=1.5)
    @test_throws Union{TypeError,MethodError} BarrierSelection(sd=:all)
    # ... and by the internal _selected fallback for other element types.
    @test_throws ArgumentError NonconvexSemidefinite._selected(1.5, 3)
    @test_throws ArgumentError NonconvexSemidefinite._selected("all", 3)
    @test NonconvexSemidefinite._selected(true, 3) == [1, 2, 3]
    @test NonconvexSemidefinite._selected(false, 3) == Int[]
    @test NonconvexSemidefinite._selected([2], 3) == [2]

    # BarrierOptions assertion errors for bad c_decr / c_init.
    @test_throws AssertionError BarrierOptions(sub_options=nothing, c_decr=1.5)
    @test_throws AssertionError BarrierOptions(sub_options=nothing, c_decr=0.0)
    @test_throws AssertionError BarrierOptions(sub_options=nothing, c_decr=[0.5, 1.1])
    @test_throws AssertionError BarrierOptions(sub_options=nothing, c_init=0.0)
    @test_throws AssertionError BarrierOptions(sub_options=nothing, c_init=-2.0)
    @test_throws AssertionError BarrierOptions(0.0, 0.1, 3; sub_options=nothing)

    # Legacy positional constructor stores the default sd-only selection.
    opts_pos = BarrierOptions(1.0, 0.1, 3; sub_options=nothing)
    @test opts_pos.c_init == 1.0 && opts_pos.c_decr == 0.1 && opts_pos.n_iter == 3
    @test opts_pos.barrier.sd === true
    @test opts_pos.barrier.ineq === false && opts_pos.barrier.box === false
    @test opts_pos.adaptive === false && opts_pos.tol === nothing
    @test opts_pos.keep_all === false

    # --- strict-feasibility validation inside Workspace ---

    # Two inequalities; at x = [1.0] constraint 1 is satisfied (-1) and
    # constraint 2 is violated (1).
    vm_ineq = NonconvexCore.VecModel(
        NonconvexCore.Objective(x -> x[1]^2),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions([
            NonconvexCore.IneqConstraint(x -> x[1] - 2.0, 0.0),
            NonconvexCore.IneqConstraint(x -> 2.0 - x[1], 0.0),
        ]),
        NonconvexCore.VectorOfFunctions(()),
        [-5.0],
        [5.0],
        [1.0],
        falses(1),
    )
    opts_all_ineq = BarrierOptions(
        sub_options=nothing, barrier=BarrierSelection(ineq=true, box=false, sd=false)
    )
    @test_throws ArgumentError NonconvexCore.Workspace(
        vm_ineq, alg, [1.0]; options=opts_all_ineq
    )

    # Per-index selection: constraint 1 only → accepted ...
    opts_sel1 = BarrierOptions(
        sub_options=nothing, barrier=BarrierSelection(ineq=[1], box=false, sd=false)
    )
    ws1 = NonconvexCore.Workspace(vm_ineq, alg, [1.0]; options=opts_sel1)
    @test ws1 isa NonconvexSemidefinite.BarrierWorkspace
    # ... constraint 2 only → still rejected, naming the right index.
    opts_sel2 = BarrierOptions(
        sub_options=nothing, barrier=BarrierSelection(ineq=[2], box=false, sd=false)
    )
    err_sel2 = try
        NonconvexCore.Workspace(vm_ineq, alg, [1.0]; options=opts_sel2)
    catch e
        e
    end
    @test err_sel2 isa ArgumentError
    @test occursin("constraint 2", sprint(showerror, err_sel2))

    # Box strict feasibility (lower and upper).
    function vm_box(lo, hi)
        return NonconvexCore.VecModel(
            NonconvexCore.Objective(x -> x[1]^2),
            NonconvexCore.VectorOfFunctions(()),
            NonconvexCore.VectorOfFunctions(()),
            NonconvexCore.VectorOfFunctions(()),
            [lo],
            [hi],
            [0.0],
            falses(1),
        )
    end
    opts_box = BarrierOptions(
        sub_options=nothing, barrier=BarrierSelection(ineq=false, box=true, sd=false)
    )
    err_lo = try
        NonconvexCore.Workspace(vm_box(0.0, Inf), alg, [0.0]; options=opts_box)
    catch e
        e
    end
    @test err_lo isa ArgumentError
    @test occursin("lower bound", sprint(showerror, err_lo))
    @test occursin("variable 1", sprint(showerror, err_lo))
    err_hi = try
        NonconvexCore.Workspace(vm_box(-Inf, 1.0), alg, [1.0]; options=opts_box)
    catch e
        e
    end
    @test err_hi isa ArgumentError
    @test occursin("upper bound", sprint(showerror, err_hi))
    ws_box = NonconvexCore.Workspace(vm_box(0.0, Inf), alg, [0.5]; options=opts_box)
    @test ws_box isa NonconvexSemidefinite.BarrierWorkspace

    # Semidefinite strict interiority.
    vm_sd = NonconvexCore.VecModel(
        NonconvexCore.Objective(x -> x[1]^2),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions([NonconvexCore.SDConstraint(x -> [x[1];;], 1)]),
        [-Inf],
        [Inf],
        [0.0],
        falses(1),
    )
    opts_sd = BarrierOptions(sub_options=nothing)
    err_sd = try
        NonconvexCore.Workspace(vm_sd, alg, [0.0]; options=opts_sd)
    catch e
        e
    end
    @test err_sd isa ArgumentError
    @test occursin("strictly interior", sprint(showerror, err_sd))
    ws_sd = NonconvexCore.Workspace(vm_sd, alg, [1.0]; options=opts_sd)
    @test ws_sd isa NonconvexSemidefinite.BarrierWorkspace

    # Per-index sd selection: an indefinite non-selected constraint passes.
    vm2sd = NonconvexCore.VecModel(
        NonconvexCore.Objective(x -> x[1] + x[2]),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions([
            NonconvexCore.SDConstraint(x -> [x[1] + 1.0;;], 1),
            NonconvexCore.SDConstraint(x -> [x[2] - 1.0;;], 1),
        ]),
        [-Inf, -Inf],
        [Inf, Inf],
        [1.0, -1.0],
        falses(2),
    )
    opts_sd1 = BarrierOptions(
        sub_options=nothing, barrier=BarrierSelection(ineq=false, box=false, sd=[1])
    )
    ws2sd = NonconvexCore.Workspace(vm2sd, alg, [1.0, -1.0]; options=opts_sd1)
    @test ws2sd isa NonconvexSemidefinite.BarrierWorkspace
    opts_sd12 = BarrierOptions(
        sub_options=nothing, barrier=BarrierSelection(ineq=false, box=false, sd=[1, 2])
    )
    @test_throws ArgumentError NonconvexCore.Workspace(
        vm2sd, alg, [1.0, -1.0]; options=opts_sd12
    )

    # c_init / c_decr arrays must have one entry per selected constraint group.
    @test_throws AssertionError NonconvexCore.Workspace(
        vm_sd, alg, [1.0]; options=BarrierOptions(sub_options=nothing, c_init=[1.0, 2.0])
    )
    @test_throws AssertionError NonconvexCore.Workspace(
        vm_sd, alg, [1.0]; options=BarrierOptions(sub_options=nothing, c_decr=[0.5, 0.5])
    )
end

@testitem "barrier: schedules, gap termination, to_barrier and keep_all" begin
    using NonconvexSemidefinite, NonconvexCore, Test, LinearAlgebra

    # A trivial identity sub-solver: returns the incoming point unchanged.
    struct DummyOptions end
    struct DummyResult
        minimizer::Vector{Float64}
        minimum::Float64
    end
    struct DummyAlg <: NonconvexCore.AbstractOptimizer end
    function NonconvexCore.solver_capabilities(::DummyAlg)
        return ConstraintCapabilities(;
            linear_eq=true,
            nonlinear_eq=true,
            linear_ineq=true,
            nonlinear_ineq=true,
            linear_sd=true,
            nonlinear_sd=true,
            box=true,
            continuous=true,
            binary=false,
            integer=false,
            infinite_bounds=true,
        )
    end
    struct DummyWorkspace <: NonconvexCore.Workspace
        model::NonconvexCore.VecModel
        x0::Vector{Float64}
    end
    function NonconvexCore.Workspace(
        model::NonconvexCore.VecModel, ::DummyAlg, x0, args...; options, kwargs...
    )
        return DummyWorkspace(model, copy(x0))
    end
    function NonconvexCore.optimize!(w::DummyWorkspace)
        return DummyResult(copy(w.x0), w.model.objective(w.x0))
    end

    # --- _next_c: fixed and adaptive schedules ---
    @test NonconvexSemidefinite._next_c(
        (1.0, 2.0), (2, 3), 0.0, 5, 20, (0.5, 0.5), false
    ) == (0.5, 1.0)
    @test NonconvexSemidefinite._next_c((1.0,), (2,), 0.5, 1, 2, (0.1,), true) == (0.25,)
    c_ad = NonconvexSemidefinite._next_c((1.0,), (2,), 0.0, 1, 50, (0.1,), true)
    @test c_ad[1] ≈ max(0.1, eps(Float64)^(1 / 49))
    # The c_decr floor dominates when the superlinear ratio would be smaller.
    @test NonconvexSemidefinite._next_c((1.0,), (2,), 1e-3, 1, 2, (0.99,), true) == (0.99,)

    # --- _log_barrier branches ---
    @test NonconvexSemidefinite._log_barrier(-2.0, 1e-6) == -log(2.0)
    @test NonconvexSemidefinite._log_barrier(-1e-6, 1e-6) ≈ -log(1e-6)  # t == -ε
    let ε = 1e-6, t = -1e-7  # smooth quadratic extension
        @test NonconvexSemidefinite._log_barrier(t, ε) ≈
            -log(ε) + (t + ε) / ε + (t + ε)^2 / (2ε^2)
    end

    # --- safe_logdet ---
    @test NonconvexSemidefinite.safe_logdet([2.0 0.5; 0.5 2.0]) ≈ log(3.75)
    @test NonconvexSemidefinite.safe_logdet([-2.0 0.0; 0.0 1.0]) == -Inf

    # --- _barrier_units: ordering ineq, box, sd ---
    vm_mix = NonconvexCore.VecModel(
        NonconvexCore.Objective(x -> x[1]),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions([
            NonconvexCore.IneqConstraint(x -> x[1] - 1.0, 0.0),
            NonconvexCore.IneqConstraint(x -> [x[1] - 1.0, x[1] + x[2] - 1.0], [0.0, 0.0]),
        ]),
        NonconvexCore.VectorOfFunctions([
            NonconvexCore.SDConstraint(x -> [x[1] + 2.0;;], 1)
        ]),
        [-Inf, -Inf],
        [2.0, Inf],
        [0.5, 0.5],
        falses(2),
    )
    ineq_sel, sd_sel, ms = NonconvexSemidefinite._barrier_units(
        vm_mix, BarrierSelection(ineq=true, box=true, sd=true)
    )
    @test ineq_sel == [1, 2] && sd_sel == [1]
    @test ms == [1, 2, 1, 1]  # ineq dims, one finite box bound, sd dim

    # --- to_barrier: selected constraints are removed, others kept ---
    # All selected (empty kept lists): use a single-ineq model so the kept
    # lists have a concrete element type.
    vm_single = NonconvexCore.VecModel(
        NonconvexCore.Objective(x -> x[1]),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions([
            NonconvexCore.IneqConstraint(x -> x[1] - 1.0, 0.0)
        ]),
        NonconvexCore.VectorOfFunctions([
            NonconvexCore.SDConstraint(x -> [x[1] + 2.0;;], 1)
        ]),
        [-Inf],
        [Inf],
        [0.5],
        falses(1),
    )
    bm = NonconvexSemidefinite.to_barrier(
        vm_single, (0.5, 0.5); selection=BarrierSelection(ineq=true, box=false, sd=true)
    )
    @test isempty(bm.ineq_constraints.fs)
    @test isempty(bm.sd_constraints.fs)
    bm_default = NonconvexSemidefinite.to_barrier(vm_mix, (0.5,))  # sd-only by default
    @test length(bm_default.ineq_constraints.fs) == 2
    @test isempty(bm_default.sd_constraints.fs)
    bm_part = NonconvexSemidefinite.to_barrier(
        vm_mix, (0.5, 0.5); selection=BarrierSelection(ineq=[2], box=true, sd=false)
    )
    @test length(bm_part.ineq_constraints.fs) == 1
    @test length(bm_part.sd_constraints.fs) == 1

    # --- barrier objective: value at a strictly interior point, and the
    # smooth-extension/safe-logdet behaviour at an infeasible point ---
    vm_obj = NonconvexCore.VecModel(
        NonconvexCore.Objective(x -> x[1]),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions([
            NonconvexCore.SDConstraint(x -> [x[1] + 2.0;;], 1)
        ]),
        [-Inf],
        [Inf],
        [1.0],
        falses(1),
    )
    bo = NonconvexSemidefinite._barrier_objective(
        vm_obj.objective,
        vm_obj,
        (0.5,),
        BarrierSelection(ineq=false, box=false, sd=true),
        1e-6,
    )
    @test bo([1.0]) ≈ 1.0 - 0.5 * log(3.0)
    @test isinf(bo([-3.0]))

    alg = BarrierAlg(sub_alg=DummyAlg())

    # --- full run with the identity sub-solver: keep_all stores the trajectory ---
    vm_sd = NonconvexCore.VecModel(
        NonconvexCore.Objective(x -> x[1]^2),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions([
            NonconvexCore.SDConstraint(x -> [x[1] + 1.0;;], 1)
        ]),
        [-10.0],
        [10.0],
        [1.0],
        falses(1),
    )
    opts = BarrierOptions(sub_options=DummyOptions(), n_iter=4, keep_all=true)
    r = optimize(vm_sd, alg, [1.0]; options=opts)
    @test r isa BarrierResult
    @test r.minimizer ≈ [1.0]
    @test r.minimum ≈ 1.0
    @test r.results isa Vector && length(r.results) == 4
    @test r.optimal_ind == 1

    # keep_all=false drops the trajectory.
    r2 = optimize(
        vm_sd, alg, [1.0]; options=BarrierOptions(sub_options=DummyOptions(), n_iter=4)
    )
    @test r2.results === nothing
    @test r2.optimal_ind === nothing

    # --- non-adaptive schedule + duality-gap termination: with ms = [2],
    # c_init = 1.0, c_decr = 0.5 and tol = 1e-3, the gap 2*0.5^(k-1) first
    # reaches 1e-3 at k = 12 → exactly 11 sub-problems are solved. ---
    vm2 = NonconvexCore.VecModel(
        NonconvexCore.Objective(x -> sum(abs2, x)),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions([
            NonconvexCore.SDConstraint(
                x -> decompress_symmetric(@view(x[1:1]), @view(x[2:3])), 2
            ),
        ]),
        [-Inf, 0.0, 0.0],
        fill(Inf, 3),
        [0.0, 2.0, 3.0],
        falses(3),
    )
    opts3 = BarrierOptions(
        sub_options=DummyOptions(),
        n_iter=20,
        c_init=1.0,
        c_decr=0.5,
        tol=1e-3,
        keep_all=true,
    )
    r3 = optimize(vm2, alg, [0.0, 2.0, 3.0]; options=opts3)
    @test length(r3.results) == 11

    # --- adaptive schedule without tol: runs all n_iter iterations ---
    r4 = optimize(
        vm2,
        alg,
        [0.0, 2.0, 3.0];
        options=BarrierOptions(
            sub_options=DummyOptions(), n_iter=3, adaptive=true, keep_all=true
        ),
    )
    @test length(r4.results) == 3

    # --- adaptive + tol: with one step remaining the superlinear ratio is
    # target / c exactly, the gap reaches tol at k = 2 and the loop stops
    # after a single sub-problem ---
    r5 = optimize(
        vm2,
        alg,
        [0.0, 2.0, 3.0];
        options=BarrierOptions(
            sub_options=DummyOptions(), n_iter=2, adaptive=true, tol=0.5, keep_all=true
        ),
    )
    @test length(r5.results) == 1

    # --- box-only barrierification through the full loop ---
    vm_box = NonconvexCore.VecModel(
        NonconvexCore.Objective(x -> (x[1] - 3.0)^2),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions(()),
        [-Inf],
        [2.0],
        [0.0],
        falses(1),
    )
    r_box = optimize(
        vm_box,
        alg,
        [0.0];
        options=BarrierOptions(
            sub_options=DummyOptions(),
            n_iter=2,
            barrier=BarrierSelection(ineq=false, box=true, sd=false),
        ),
    )
    @test r_box.minimizer ≈ [0.0]
    @test r_box.minimum ≈ 9.0

    # --- end-to-end through the AbstractModel path (Model + tovecmodel) ---
    m = Model(x -> x[1]^2)
    addvar!(m, -10.0, 10.0)
    add_sd_constraint!(m, x -> [x[1] + 1.0;;])
    rm = optimize(
        m,
        alg,
        [1.0];
        options=BarrierOptions(sub_options=DummyOptions(), n_iter=2, keep_all=true),
    )
    @test rm.minimizer ≈ [1.0]
    @test rm.minimum ≈ 1.0
    @test length(rm.results) == 2
end

@testitem "SDPQuadAlg: internals, layout errors and low-rank factors" begin
    using NonconvexSemidefinite, NonconvexCore, Test, LinearAlgebra

    # Identity sub-solver scaffolding (see the barrier testitem).
    struct DummyOptions end
    struct DummyResult
        minimizer::Vector{Float64}
        minimum::Float64
    end
    struct DummyAlg <: NonconvexCore.AbstractOptimizer end
    function NonconvexCore.solver_capabilities(::DummyAlg)
        return ConstraintCapabilities(;
            linear_eq=true,
            nonlinear_eq=true,
            linear_ineq=true,
            nonlinear_ineq=true,
            linear_sd=true,
            nonlinear_sd=true,
            box=true,
            continuous=true,
            binary=false,
            integer=false,
            infinite_bounds=true,
        )
    end
    struct DummyWorkspace <: NonconvexCore.Workspace
        model::NonconvexCore.VecModel
        x0::Vector{Float64}
    end
    function NonconvexCore.Workspace(
        model::NonconvexCore.VecModel, ::DummyAlg, x0, args...; options, kwargs...
    )
        return DummyWorkspace(model, copy(x0))
    end
    function NonconvexCore.optimize!(w::DummyWorkspace)
        return DummyResult(copy(w.x0), w.model.objective(w.x0))
    end

    # --- options validation ---
    @test_throws ArgumentError SDPQuadOptions(sub_options=DummyOptions(), rank=0)
    @test_throws ArgumentError SDPQuadOptions(sub_options=DummyOptions(), rank=-3)
    @test_throws ArgumentError SDPQuadOptions(sub_options=DummyOptions(), diag_floor=-1.0)
    @test SDPQuadOptions(sub_options=DummyOptions()).rank === nothing
    caps = solver_capabilities(SDPQuadAlg(sub_alg=DummyAlg()))
    @test caps.linear_sd === true
    @test caps.nonlinear_sd === true

    # --- internal helpers ---
    @test NonconvexSemidefinite._cholesky_param([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 3) ≈
        [4.0 0.0 0.0; 1.0 5.0 0.0; 2.0 3.0 6.0]
    @test NonconvexSemidefinite._unpack_lowrank([1.0, 2.0, 3.0, 4.0], 2, 2) ≈
        [1.0 3.0; 2.0 4.0]
    @test NonconvexSemidefinite._factor_chunk_size(2, nothing) == 3
    @test NonconvexSemidefinite._factor_chunk_size(2, 3) == 6
    @test NonconvexSemidefinite._factor_chunk_size(3, 1) == 3
    @test NonconvexSemidefinite._wrap(sum, x -> 2.0 .* x)([1.0, 2.0]) == 6.0

    # --- model layout: one 2×2 sd chunk = 3 variables in (x_L, x_D) layout ---
    function make_quad_vm(; extra=0, init=Float64[])
        nbase = 3 + extra
        sd = NonconvexCore.SDConstraint(
            x -> decompress_symmetric(@view(x[1:1]), @view(x[2:3])), 2
        )
        _init = isempty(init) ? vcat([0.0, 2.0, 3.0], zeros(extra)) : init
        return NonconvexCore.VecModel(
            NonconvexCore.Objective(x -> sum(abs2, x)),
            NonconvexCore.VectorOfFunctions(()),
            NonconvexCore.VectorOfFunctions(()),
            NonconvexCore.VectorOfFunctions([sd]),
            vcat([-Inf, 0.0, 0.0], fill(-Inf, extra)),
            fill(Inf, nbase),
            _init,
            falses(nbase),
        )
    end
    vm = make_quad_vm()
    @test NonconvexSemidefinite._sd_layout(vm) == [(0, 1, 2)]
    @test NonconvexSemidefinite._check_layout(vm) == [(0, 1, 2)]

    # --- factor bounds ---
    lbs, ubs = NonconvexSemidefinite._factor_bounds([(0, 1, 2)], nothing, 1e-6)
    @test lbs == [-Inf, 1e-6, 1e-6]
    @test all(isinf, ubs)
    lbs2, _ = NonconvexSemidefinite._factor_bounds([(0, 1, 2)], 1, 1e-6)
    @test lbs2 == [-Inf, -Inf]

    # --- layout errors: extra variables ---
    vm_extra = make_quad_vm(extra=1)
    err = try
        NonconvexSemidefinite._check_layout(vm_extra)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("3 entries expected, got 4", sprint(showerror, err))

    # --- layout errors: no sd constraints at all ---
    vm_nosd = NonconvexCore.VecModel(
        NonconvexCore.Objective(x -> x[1]),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions(()),
        [-Inf],
        [Inf],
        [0.0],
        falses(1),
    )
    @test_throws ArgumentError NonconvexSemidefinite._check_layout(vm_nosd)

    # --- wrong init size: the layout check passes but building the initial
    # factors reads out of bounds ---
    vm_bad = make_quad_vm(init=[0.0, 2.0])
    @test_throws BoundsError optimize(
        vm_bad,
        SDPQuadAlg(sub_alg=DummyAlg()),
        [0.0, 2.0];
        options=SDPQuadOptions(sub_options=DummyOptions()),
    )

    alg = SDPQuadAlg(sub_alg=DummyAlg())
    x0 = [0.0, 2.0, 3.0]

    # --- full-rank Cholesky re-parameterization (identity sub-solver) ---
    r = optimize(vm, alg, x0; options=SDPQuadOptions(sub_options=DummyOptions()))
    @test r isa SDPQuadResult
    @test length(r.minimizer) == 3
    A = decompress_symmetric(@view(r.minimizer[1:1]), @view(r.minimizer[2:3]))
    @test A ≈ [2.0 0.0; 0.0 3.0] atol = 1e-3  # A0 + 1e-6 I
    @test minimum(eigvals(Symmetric(A))) > 0
    @test r.minimum == vm.objective(r.minimizer)

    # --- low-rank (rank 1) Burer-Monteiro: A = R R', R ∈ R^{2×1} ---
    r_lr = optimize(vm, alg, x0; options=SDPQuadOptions(sub_options=DummyOptions(), rank=1))
    A_lr = decompress_symmetric(@view(r_lr.minimizer[1:1]), @view(r_lr.minimizer[2:3]))
    @test A_lr ≈ [2.0 0.0; 0.0 0.0] atol = 1e-8

    # --- rank 2 recovers the diagonal matrix exactly ---
    r_2 = optimize(vm, alg, x0; options=SDPQuadOptions(sub_options=DummyOptions(), rank=2))
    A_2 = decompress_symmetric(@view(r_2.minimizer[1:1]), @view(r_2.minimizer[2:3]))
    @test A_2 ≈ [2.0 0.0; 0.0 3.0] atol = 1e-8
end

@testitem "_optimize_precheck no-ops and capability merging" begin
    using NonconvexSemidefinite, NonconvexCore, Test

    struct DummyOptions end
    struct DummyAlg <: NonconvexCore.AbstractOptimizer end
    struct NoCapAlg <: NonconvexCore.AbstractOptimizer end  # declares no capabilities

    vm_inf = NonconvexCore.VecModel(
        NonconvexCore.Objective(x -> Inf),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions(()),
        NonconvexCore.VectorOfFunctions(()),
        [-1.0],
        [1.0],
        [0.0],
        falses(1),
    )

    # The meta-algorithms skip the finite-start check of the generic precheck
    # (their own Workspace constructors validate strict feasibility instead).
    @test NonconvexCore._optimize_precheck(
        vm_inf,
        BarrierAlg(sub_alg=DummyAlg()),
        [0.0];
        options=BarrierOptions(sub_options=DummyOptions()),
    ) === nothing
    @test NonconvexCore._optimize_precheck(
        vm_inf,
        SDPQuadAlg(sub_alg=DummyAlg()),
        [0.0];
        options=SDPQuadOptions(sub_options=DummyOptions()),
    ) === nothing

    # The generic precheck still rejects non-finite starts for other solvers.
    @test_throws ErrorException NonconvexCore._optimize_precheck(vm_inf, DummyAlg(), [0.0])

    # Capability merging: the meta-algorithms add sd (and barrier kinds) on
    # top of the sub-solver's declaration; undeclared fields stay missing.
    caps_barrier = solver_capabilities(BarrierAlg(sub_alg=NoCapAlg()))
    @test caps_barrier.linear_sd === true
    @test caps_barrier.nonlinear_sd === true
    @test caps_barrier.linear_ineq === true
    @test caps_barrier.nonlinear_ineq === true
    @test caps_barrier.box === true
    @test caps_barrier.linear_eq === missing
    caps_quad = solver_capabilities(SDPQuadAlg(sub_alg=NoCapAlg()))
    @test caps_quad.linear_sd === true
    @test caps_quad.nonlinear_sd === true
    @test caps_quad.box === missing
end
