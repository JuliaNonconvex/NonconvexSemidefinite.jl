# Kreisselmeier–Steinhauser aggregation of the algebraically smallest
# generalized eigenvalues of a symmetric pencil A·x = λ·B·x, with an
# analytic reverse rule. The eigenpairs come from LOBPCG by default.

"""
    ks_aggregate(A, B, k, ρ; step=0.0, ratio=false, eigsolver=nothing)

The Kreisselmeier–Steinhauser approximation of the minimum of the `k`
algebraically smallest generalized eigenvalues `λ₁ ≤ … ≤ λₖ` of the
symmetric pencil `A·x = λ·B·x` (`A`, `B` symmetric, `B` positive
definite). Two one-sided logsumexp aggregates bound the minimum — the
eigenvalues may be negative (e.g. the ν's of the buckling pencil):

- the lower bound on min (default, `step = 0`):
  `KS = -(1/ρ)·log(Σᵢ exp(-ρλᵢ))` — KS ≤ λ₁, KS → λ₁ as ρ → ∞;
- the negative upper bound on the negative eigenvalues (`step = 1`):
  `KS = -(1/ρ)·log(Σᵢ exp(ρλᵢ))` — KS ≥ λ₁ whenever the λᵢ are negative;
- `0 < step < 1`: the convex combination
  `KS = (1-step)·lower + step·upper` — a step between the lower and
  upper bounds on the minimum;
- `ratio = true`: `KS = upper − lower = (1/ρ)·log(Σexp(-ρλᵢ)/Σexp(ρλᵢ))`,
  a surrogate for the spread of the smallest eigenvalues.

The reverse rule is the closed-form eigenvalue sensitivity: with the
B-normalized eigenvectors `vᵢ` (`vᵢᵀBvᵢ = 1`), `dλᵢ = vᵢᵀ(dA − λᵢdB)vᵢ`,
weighted per mode by `wᵢ = (1-step)·softmax(-ρλ)ᵢ − step·softmax(ρλ)ᵢ`;
the cotangents are

    ΔA = Σᵢ Δ·wᵢ·vᵢvᵢᵀ,     ΔB = Σᵢ Δ·wᵢ·(−λᵢ)·vᵢvᵢᵀ,

constructed as sparse matrices when `A`/`B` are sparse. The outer-product
cotangents are populated over the full vector space (O(k·n²)), so the
matrix-level rule targets small-to-moderate problems; large sparse
pencils should compose through element-level rules instead.

The eigenpairs come from LOBPCG by default — `lobpcg(A, B, false, X0)`
for the smallest generalized eigenvalues, B-orthonormal vectors — and can
be replaced with any `eigsolver(A, B, k) -> (λs, vectors)`, e.g. a dense
`eigen` for testing.
"""
function ks_aggregate(
    A, B, k::Int, ρ; step::Real=0.0, ratio::Bool=false, eigsolver=nothing
)
    0 <= step <= 1 || throw(ArgumentError("step must be in [0, 1], got $(step)"))
    es = eigsolver === nothing ? _ks_lobpcg : eigsolver
    # the pencil is symmetric by definition: symmetrize explicitly so the
    # cotangent Σwᵢvᵢvᵢᵀ is the true derivative w.r.t. the full-matrix
    # argument (eigen(Symmetric(A)) alone would ignore the lower triangle).
    As = (A + A') / 2
    Bs = (B + B') / 2
    λs, _ = es(As, Bs, k)
    return _ks_value(λs, ρ, step, ratio)
end

# the two one-sided logsumexp aggregates of the minimum
_ks_lower(λs, ρ) =
    isempty(λs) ? -Inf : -(1 / ρ) * log(sum(exp(-ρ * λ) for λ in λs))
_ks_upper(λs, ρ) =
    isempty(λs) ? Inf : -(1 / ρ) * log(sum(exp(ρ * λ) for λ in λs))

function _ks_value(λs, ρ, step, ratio)
    if ratio
        return _ks_upper(λs, ρ) - _ks_lower(λs, ρ)
    end
    return (1 - step) * _ks_lower(λs, ρ) + step * _ks_upper(λs, ρ)
end

# default eigensolver: LOBPCG for the algebraically smallest eigenvalues
function _ks_lobpcg(A, B, k::Int)
    n = size(A, 1)
    X0 = rand(n, min(k + 2, n))
    res = lobpcg(A, B, false, X0; tol=1.0e-6, maxiter=400)
    nk = min(k, length(res.λ))
    return collect(Float64, res.λ[1:nk]), [vec(res.X[:, j]) for j in 1:nk]
end

# the KS softmax weights per mode
function _ks_weights(λs, ρ, step, ratio)
    if ratio
        # ratio = upper − lower; d(upper)/dλᵢ = −softmax(ρλ)ᵢ and
        # d(lower)/dλᵢ = +softmax(−ρλ)ᵢ
        wu = exp.(ρ .* λs); wu ./= sum(wu)
        wl = exp.(-ρ .* λs); wl ./= sum(wl)
        return .-(wl .+ wu)
    end
    wl = exp.(-ρ .* λs); wl ./= sum(wl)
    wu = exp.(ρ .* λs); wu ./= sum(wu)
    return (1 - step) .* wl .- step .* wu
end

# Σᵢ cᵢ·vᵢvᵢᵀ as a cotangent matching P's storage (sparse via COO when P
# is sparse; the outer products span the full space — documented above).
function _outer_cotangent(P, Δ, w, coeffs, Xs)
    if P isa SparseMatrixCSC
        rows = Int[]
        cols = Int[]
        vals = eltype(P)[]
        for (i, v) in enumerate(Xs)
            c = Δ * w[i] * coeffs[i]
            iszero(c) && continue
            for a in eachindex(v), b in eachindex(v)
                iszero(v[a] * v[b]) && continue
                push!(rows, a)
                push!(cols, b)
                push!(vals, c * v[a] * v[b])
            end
        end
        return sparse(rows, cols, vals, size(P, 1), size(P, 2))
    else
        ΔP = zeros(eltype(P), size(P))
        for (i, v) in enumerate(Xs)
            ΔP .+= (Δ * w[i] * coeffs[i]) .* (v * v')
        end
        return ΔP
    end
end

function ChainRulesCore.rrule(
    ::typeof(ks_aggregate), A, B, k::Int, ρ;
    step::Real=0.0, ratio::Bool=false, eigsolver=nothing
)
    0 <= step <= 1 || throw(ArgumentError("step must be in [0, 1], got $(step)"))
    es = eigsolver === nothing ? _ks_lobpcg : eigsolver
    As = (A + A') / 2
    Bs = (B + B') / 2
    λs, Xs = es(As, Bs, k)
    val = _ks_value(λs, ρ, step, ratio)
    function ks_aggregate_pb(Δ)
        Δ = ChainRulesCore.unthunk(Δ)
        if Δ isa ChainRulesCore.AbstractZero || isempty(λs)
            ΔA = A isa SparseMatrixCSC ? spzeros(eltype(A), size(A)...) :
                 zeros(eltype(A), size(A))
            ΔB = B isa SparseMatrixCSC ? spzeros(eltype(B), size(B)...) :
                 zeros(eltype(B), size(B))
        else
            w = _ks_weights(λs, ρ, step, ratio)
            ΔA = _outer_cotangent(A, Δ, w, ones(length(λs)), Xs)
            ΔB = _outer_cotangent(B, Δ, w, -λs, Xs)
        end
        return NoTangent(), ΔA, ΔB, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    return val, ks_aggregate_pb
end
