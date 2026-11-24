# Kreisselmeier–Steinhauser aggregation of the algebraically lowest /
# highest generalized eigenvalues of a symmetric pencil A·x = λ·B·x, with
# an analytic reverse rule. The eigenpairs come from LOBPCG by default.

"""
    ks_aggregate(A, B, k, ρ; which=:lower, eigsolver=nothing)

The Kreisselmeier–Steinhauser aggregation of the `k` algebraically lowest
or highest generalized eigenvalues `λ₁ ≤ … ≤ λₖ` of the symmetric pencil
`A·x = λ·B·x` (`A`, `B` symmetric, `B` positive definite):

- `which = :lower`: `KS = -(1/ρ)·log(Σᵢ exp(-ρλᵢ))` — a smooth
  under-approximation of `λ₁` (KS ≤ λ₁, KS → λ₁ as ρ → ∞); the
  lower-bound aggregate.
- `which = :upper`: `KS = (1/ρ)·log(Σᵢ exp(ρλᵢ))` — the upper-bound
  aggregate, a smooth over-approximation of `λₖ`.
- `which = :ratio`: `KS = upper − lower ≈ log(λₖ/λ₁)` — the log-ratio of
  the two aggregates, a surrogate for the spectral condition number.

The reverse rule is the closed-form eigenvalue sensitivity: with the
B-normalized eigenvectors `vᵢ` (`vᵢᵀBvᵢ = 1`), `dλᵢ = vᵢᵀ(dA − λᵢdB)vᵢ`,
weighted by the softmax `wᵢ` of `±ρλᵢ` (for `:ratio`, `wᵢ = wᵢ⁺ − wᵢ⁻`);
the cotangents are

    ΔA = Σᵢ Δ·wᵢ·vᵢvᵢᵀ,     ΔB = Σᵢ Δ·wᵢ·(−λᵢ)·vᵢvᵢᵀ,

constructed as sparse matrices when `A`/`B` are sparse. The outer-product
cotangents are populated over the full vector space (O(k·n²)), so the
matrix-level rule targets small-to-moderate problems; large sparse
pencils should compose through element-level rules instead.

The eigenpairs come from LOBPCG by default — `lobpcg(A, B, largest, X0)`
with B-orthonormal vectors, `largest` per `which` — and can be replaced
with any `eigsolver(A, B, k, which) -> (λs, vectors)`, e.g. a dense
`eigen` for testing.
"""
function ks_aggregate(
    A, B, k::Int, ρ; which::Symbol=:lower, eigsolver=nothing
)
    which in (:lower, :upper, :ratio) || throw(
        ArgumentError("which must be :lower, :upper or :ratio, got $(which)"),
    )
    es = eigsolver === nothing ? _ks_lobpcg : eigsolver
    # the pencil is symmetric by definition: symmetrize explicitly so the
    # cotangent Σwᵢvᵢvᵢᵀ is the true derivative w.r.t. the full-matrix
    # argument (eigen(Symmetric(A)) alone would ignore the lower triangle).
    As = (A + A') / 2
    Bs = (B + B') / 2
    λs, _ = es(As, Bs, k, which)
    return _ks_value(λs, ρ, which)
end

_ks_value(λs, ρ, ::Val{:lower}) = -(1 / ρ) * log(sum(exp(-ρ * λ) for λ in λs))
_ks_value(λs, ρ, ::Val{:upper}) = (1 / ρ) * log(sum(exp(ρ * λ) for λ in λs))
_ks_value(λs, ρ, ::Val{:ratio}) =
    _ks_value(λs, ρ, Val(:upper)) - _ks_value(λs, ρ, Val(:lower))
_ks_value(λs, ρ, which::Symbol) = _ks_value(λs, ρ, Val(which))

# default eigensolver: LOBPCG on the generalized pencil (A, B)
function _ks_lobpcg(A, B, k::Int, which::Symbol)
    n = size(A, 1)
    largest = which === :upper
    X0 = rand(n, min(k + 2, n))
    res = lobpcg(A, B, largest, X0; tol=1.0e-6, maxiter=400)
    nk = min(k, length(res.λ))
    λs = collect(Float64, res.λ[1:nk])
    Xs = [vec(res.X[:, j]) for j in 1:nk]
    return λs, Xs
end

# the KS softmax weights per mode
function _ks_weights(λs, ρ, which::Symbol)
    if which === :lower
        w = exp.(-ρ .* λs)
        w ./= sum(w)
        return w
    elseif which === :upper
        w = exp.(ρ .* λs)
        w ./= sum(w)
        return w
    else
        wu = exp.(ρ .* λs)
        wu ./= sum(wu)
        wl = exp.(-ρ .* λs)
        wl ./= sum(wl)
        return wu .- wl
    end
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
    which::Symbol=:lower, eigsolver=nothing
)
    es = eigsolver === nothing ? _ks_lobpcg : eigsolver
    As = (A + A') / 2
    Bs = (B + B') / 2
    λs, Xs = es(As, Bs, k, which)
    val = _ks_value(λs, ρ, which)
    function ks_aggregate_pb(Δ)
        Δ = ChainRulesCore.unthunk(Δ)
        if Δ isa ChainRulesCore.AbstractZero || isempty(λs)
            ΔA = A isa SparseMatrixCSC ? spzeros(eltype(A), size(A)...) :
                 zeros(eltype(A), size(A))
            ΔB = B isa SparseMatrixCSC ? spzeros(eltype(B), size(B)...) :
                 zeros(eltype(B), size(B))
        else
            w = _ks_weights(λs, ρ, which)
            ΔA = _outer_cotangent(A, Δ, w, ones(length(λs)), Xs)
            ΔB = _outer_cotangent(B, Δ, w, -λs, Xs)
        end
        return NoTangent(), ΔA, ΔB, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    return val, ks_aggregate_pb
end
