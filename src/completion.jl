# Exact chordal PSD projection onto the decomposable cone
#   C = { Z : Z = Σₖ EₖᵀZₖEₖ, Zₖ ⪰ 0 }   (Theorem 2.2, the Fukuda cone)
# via the ADMM with entry-wise closed-form KKT solves, with an implicit
# (fixed-point) adjoint: the pullback runs the transposed linearized ADMM
# map the same number of iterations, converging at the forward's rate —
# no differentiation through the iterates, no large linear solves.

"""
    chordal_projection(P, cliques; ρ = 1.0, max_iters = 2000, tol = 1e-10)

The projection of the symmetric matrix `P` onto the chordal-PSD cone of
the (chordal) clique structure `cliques`: the closest matrix
`A = Σₖ EₖᵀỸₖEₖ` with `Ỹₖ ⪰ 0`, computed by the ADMM fixed point.

The function carries an implicit derivative: the pullback iterates the
transposed linearized ADMM steps (the adjoint of the fixed-point map), so
gradients match the projection derivative without tracing the iterates.

`P ⪰ 0` with a chordal sparsity pattern iff the projection reproduces `P`
(the distance vanishes) — the exact chordal feasibility certificate.
"""
function chordal_projection(P, cliques; ρ=1.0, max_iters=2000, tol=1e-10)
    A, _ = _chordal_projection_impl(P, cliques, ρ, max_iters, tol)
    return A
end

function ChainRulesCore.rrule(
    ::typeof(chordal_projection), P, cliques; ρ=1.0, max_iters=2000, tol=1e-10
)
    A, state = _chordal_projection_impl(P, cliques, ρ, max_iters, tol)
    niter, Wlast, Z, V, Λ = state
    p = length(cliques)
    sizes = [length(C) for C in cliques]
    n = size(P, 1)
    m = zeros(Int, n, n)
    for C in cliques, i in C, j in C
        m[i, j] += 1
    end
    function chordal_projection_pullback(ΔA)
        ΔA = ChainRulesCore.unthunk(ΔA)
        if ΔA isa ChainRulesCore.AbstractZero
            return (ChainRulesCore.NoTangent(), zeros(size(P)), ChainRulesCore.NoTangent())
        end
        ΔZ = [zeros(s, s) for s in sizes]
        # The output cotangent (assembly adjoint) is injected ONCE, into
        # the innermost reverse iteration; the following iterations carry
        # the reverse-mode flow of the truncated map F^niter.
        ΔV = [zeros(s, s) for s in sizes]
        for (k, C) in enumerate(cliques)
            for (a, i) in enumerate(C), (b, j) in enumerate(C)
                ΔV[k][a, b] += ΔA[i, j]
            end
        end
        ΔΛ = [zeros(s, s) for s in sizes]
        ΔP = zeros(n, n)
        # The adjoint fixed-point iteration runs to its own convergence
        # (the truncated Neumann series needs more terms than the forward
        # for the slowest modes of the ADMM contraction).
        for _ in 1:max(10 * niter, 5000)
            # reverse f3: Λ⁺ = Λ + ρ(Z − V⁺)  →  ΔZ += ρΔΛ⁺, ΔV⁺ -= ρΔΛ⁺
            for k in 1:p
                ΔZ[k] .+= ρ .* ΔΛ[k]
                ΔV[k] .-= ρ .* ΔΛ[k]
            end
            # reverse f2: V⁺ = Z + Λ/ρ + EμEᵀ, μ = (P − S)/(1 + m),
            #            S = ΣEₖᵀ(Zₖ + Λₖ/ρ)Eₖ
            Δμ = zeros(n, n)
            for (k, C) in enumerate(cliques)
                for (a, i) in enumerate(C), (b, j) in enumerate(C)
                    d = ΔV[k][a, b]
                    ΔZ[k][a, b] += d
                    ΔΛ[k][a, b] += d / ρ
                    Δμ[i, j] += d
                end
            end
            ΔS = zeros(n, n)
            for i in 1:n, j in 1:n
                m[i, j] == 0 && continue
                ΔS[i, j] = -Δμ[i, j] / (1 + m[i, j])
                ΔP[i, j] += Δμ[i, j] / (1 + m[i, j])
            end
            for (k, C) in enumerate(cliques)
                for (a, i) in enumerate(C), (b, j) in enumerate(C)
                    ΔZ[k][a, b] += ΔS[i, j]
                    ΔΛ[k][a, b] += ΔS[i, j] / ρ
                end
            end
            # reverse f1: Z = Π₊(W), W = V − Λ/ρ
            ΔW = [ChainRulesCore.rrule(project_psd, Wlast[k])[2](ΔZ[k])[2] for k in 1:p]
            ΔV = ΔW
            for k in 1:p
                ΔΛ[k] .-= ΔW[k] ./ ρ
            end
            ΔZ = [zeros(s, s) for s in sizes]
        end
        return (ChainRulesCore.NoTangent(), ΔP, ChainRulesCore.NoTangent())
    end
    return A, chordal_projection_pullback
end

# The ADMM fixed point: returns (A, (niter, Wlast, Z, V, Λ)).
function _chordal_projection_impl(P, cliques, ρ, max_iters, tol)
    n = size(P, 1)
    p = length(cliques)
    sizes = [length(C) for C in cliques]
    m = zeros(Int, n, n)
    for C in cliques, i in C, j in C
        m[i, j] += 1
    end
    Z = [zeros(s, s) for s in sizes]
    V = [zeros(s, s) for s in sizes]
    Λ = [zeros(s, s) for s in sizes]
    Wlast = [zeros(s, s) for s in sizes]
    niter = max_iters
    for it in 1:max_iters
        Wlast = [V[k] - Λ[k] ./ ρ for k in 1:p]
        Zp = [project_psd(Wlast[k]) for k in 1:p]
        S = zeros(n, n)
        for (k, C) in enumerate(cliques)
            W = Zp[k] .+ Λ[k] ./ ρ
            for (a, i) in enumerate(C), (b, j) in enumerate(C)
                S[i, j] += W[a, b]
            end
        end
        μ = [(P[i, j] - S[i, j]) / (1 + m[i, j]) for i in 1:n, j in 1:n]
        Vp = [
            [
                Zp[k][a, b] + Λ[k][a, b] / ρ + μ[C[a], C[b]] for
                a in 1:sizes[k], b in 1:sizes[k]
            ] for (k, C) in enumerate(cliques)
        ]
        Λp = [Λ[k] .+ ρ .* (Zp[k] .- Vp[k]) for k in 1:p]
        resid = sqrt(
            sum(norm(Zp[k] - Vp[k])^2 for k in 1:p) + sum(norm(Λp[k] - Λ[k])^2 for k in 1:p)
        )
        Z, V, Λ = Zp, Vp, Λp
        if resid < tol
            niter = it
            break
        end
    end
    A = zeros(n, n)
    for (k, C) in enumerate(cliques)
        for (a, i) in enumerate(C), (b, j) in enumerate(C)
            A[i, j] += V[k][a, b]
        end
    end
    return A, (niter, Wlast, Z, V, Λ)
end
