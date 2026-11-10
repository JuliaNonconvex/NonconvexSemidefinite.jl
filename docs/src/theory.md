# Theory: barriers, chordal decomposition and re-parameterization

NonconvexSemidefinite collects the SDP-specific mathematics of the Nonconvex
ecosystem. This page reviews the theory behind each piece.

## The semidefinite cone and its barrier

A symmetric matrix $P \in \mathbb{S}^n$ is positive semidefinite ($P \succeq 0$)
iff $v^\top P v \ge 0$ for all $v$. The function

$$F(P) = -\log\det P, \qquad \operatorname{dom} F = \{P \succ 0\},$$

is a **self-concordant logarithmic barrier** for the cone $\mathbb{S}_+^n$
(Nesterov–Nemirovskii): it is strictly convex on the interior, tends to
$+\infty$ on the boundary, and its barrier parameter is $\nu = n$. The
primal barrier method replaces the constrained problem

$$\min \; f(x) \quad \text{s.t.} \quad P(x) \succeq 0$$

with the sequence of unconstrained subproblems

$$\min_x \; f(x) - c_k \log\det P(x), \qquad c_k \to 0.$$

Each subproblem is solved strictly inside the cone (the barrier is infinite
on the boundary), and the minimizers trace the **central path**
$x^*(c)$, which converges to a solution of the original problem as $c \to 0$.
The duality gap of the barrier subproblem is exactly

$$\text{gap} = c \cdot m,$$

where $m$ is the total number of barrier terms — for the SDP constraint
$m = n$ (each eigenvalue contributes one unit). This identity is what the
`tol` option of [`BarrierOptions`](@ref) exploits for early termination.

### The generalized barrier

[`BarrierAlg`](@ref) applies the same principle to every inequality-like
constraint kind:

| Constraint | Barrier term | Barrier parameter |
|---|---|---|
| inequality $g(x) \le 0$ | $-\log(-g(x))$ | $1$ per scalar entry |
| bound $l \le x_i \le u$ | $-\log(x_i - l) - \log(u - x_i)$ | $1$ per finite bound side |
| semidefinite $P(x) \succeq 0$ | $-\log\det P(x)$ | $n$ |

The log barrier is defined only on the strict interior. So that sub-solver
line searches can probe outside, [`BarrierAlg`](@ref) continues each term by
its C² Taylor extension beyond a threshold $-\varepsilon$:

$$-\log(-t) \;\leadsto\; -\log\varepsilon + \frac{t+\varepsilon}{\varepsilon} + \frac{(t+\varepsilon)^2}{2\varepsilon^2}, \qquad t \ge -\varepsilon.$$

The extension region must **shrink with the barrier coefficient**
($\varepsilon \sim c$): with a fixed $\varepsilon$, points far outside the
interior have a finite barrier value that the vanishing $c$ no longer
penalizes, and the subproblems leak through the constraint.

### Coefficient schedules

The fixed schedule $c_{k+1} = \tau c_k$ ($\tau < 1$) is the textbook choice.
The `adaptive = true` mode instead targets the requested gap within the
remaining budget, $c_{k+1} = c_k \cdot \max(\tau, (\text{gap}_*/c_k m)^{1/(N-k)})$,
which accelerates the final iterations.

## Chordal decomposition

A graph is **chordal** if every cycle of length $\ge 4$ has a chord; its
**maximal cliques** can be listed efficiently once a perfect elimination
ordering is known, and a sparse Cholesky factorization (AMD ordering) of
the sparsity pattern produces exactly such an ordering plus the filled
pattern. The central theorem is due to Grone, Johnson, Sá and Wolkowicz
(1984):

> A matrix $P$ whose sparsity pattern is contained in the edges of a
> chordal graph is positive semidefinite **iff every principal submatrix
> $P_{C,C}$ indexed by a maximal clique $C$ is positive semidefinite.

[`chordal_decompose`](@ref) therefore replaces one large $n \times n$
semidefinite constraint with $k$ clique constraints of sizes
$|C_1|, \ldots, |C_k|$. For a banded/chain pattern the cliques have size
$2$, turning an $O(n^3)$ eigendecomposition into $O(n)$ trivial ones. The
equivalence is exact (no relaxation error), which makes chordal
decomposition the single most effective sparsity optimization for SDP.

Two practical consequences:

- **Barrier side.** For a chordal-sparse $P \succ 0$ the determinant
  factorizes as
  $\det P = \prod_C \det P_{C,C} \,/\, \prod_S \det P_{S,S}$
  (over cliques and their separators), so
  $-\log\det P = -\sum_C \log\det P_{C,C} + \sum_S \log\det P_{S,S}$.
  The clique-sum $\sum_C -\log\det P_{C,C}$ is itself a barrier for the
  chordal cone and drives the same feasible set; the separator correction
  recovers the exact central path.
- **Sparsity probing.** [`aggregate_sparsity`](@ref) detects the structural
  dependence of each matrix entry on the variables with reverse-mode AD
  (union with the numerical support of $P(x_0)$, since AD cannot see
  constant terms). User-supplied patterns are validated against $P(x_0)$:
  an entry outside the pattern would make the decomposition inequivalent.

## Projections onto the PSD cone

The orthogonal projection onto $\mathbb{S}_+^n$ clamps eigenvalues:

$$\Pi_{\mathbb{S}_+^n}(A) = U \max(\Lambda, 0) U^\top, \qquad A = U\Lambda U^\top.$$

By the Moreau decomposition of mutually polar cones,
$A = \Pi_{\mathbb{S}_+^n}(A) + \Pi_{\mathbb{S}_-^n}(A)$, which gives
[`project_psd_neg`](@ref) for free; [`psd_infeasibility`](@ref) is then the
norm of the negative-eigenvalue part, i.e. the distance of $P$ to the cone.
The projection is a spectral function and is differentiable wherever no
eigenvalue is pinned at zero; its differential (Lewis) is

$$D\Pi(A)[E] = U\, [G \circ (U^\top E U)]\, U^\top, \qquad
G_{ij} = \begin{cases}
(\bar\lambda_i - \bar\lambda_j)/(\lambda_i - \lambda_j), & \lambda_i \neq \lambda_j, \\
\mathbf{1}_{\lambda_i > 0}, & \lambda_i = \lambda_j,
\end{cases}$$

with $\bar\lambda_i = \max(\lambda_i, 0)$. [`project_psd`](@ref) ships this
pullback as a ChainRules rrule, so reverse-mode AD differentiates through
the projection (e.g. inside the augmented Lagrangian penalty of
NonconvexAugLagLab) with machine-precision gradients.

## Quadratic re-parameterization

[`SDPQuadAlg`](@ref) eliminates the semidefinite constraint by substituting
a factorization of the matrix:

- **Full rank**: $A = LL^\top$ with lower-triangular $L$ and
  $\operatorname{diag}(L) \ge \varepsilon$. Every PSD matrix admits such a
  factor (Cholesky), so the substitution is *surjective*: no solution is
  lost. The subproblem is a box-constrained NLP over the $n(n+1)/2$ entries
  of $L$.
- **Low rank (Burer–Monteiro)**: $A = RR^\top$ with $R \in \mathbb{R}^{n\times r}$,
  $r \le n$. This covers exactly the PSD matrices of rank $\le r$; for
  large problems whose solutions are provably low rank (e.g. by Pataki's
  bound $\binom{r+1}{2} \le m$) it reduces the dimension from
  $O(n^2)$ to $O(nr)$.

Both substitutions make the problem *smooth and unconstrained* (up to the
diagonal bound), at the cost of **nonconvexity in the factor**: the
objective is invariant under orthogonal transformations $L \mapsto LQ$, so
the factor problem has symmetries and possibly spurious stationary points.
The diagonal floor keeps the full-rank factor strictly interior; for
nonconvex objectives a multistart over factors is recommended.
