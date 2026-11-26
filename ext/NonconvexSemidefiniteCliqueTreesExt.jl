# CliqueTrees.jl-backed chordal algorithms for NonconvexSemidefinite:
# loading CliqueTrees enables the MF / METIS / BT completion algorithms and
# its `ischordal` for the chordality check.

module NonconvexSemidefiniteCliqueTreesExt

using NonconvexSemidefinite: NonconvexSemidefinite
using SparseArrays: SparseMatrixCSC
using CliqueTrees: CliqueTrees

# The cliques of the chordal completion computed by CliqueTrees' clique-tree
# machinery: `cliquetree` returns (perm, tree) where perm[tree[i]] are the
# vertices of the i-th bag, i.e. a maximal clique of the completion.
function _cliquetrees_cliques(pattern::SparseMatrixCSC, ct_alg)
    perm, tree = CliqueTrees.cliquetree(
        Matrix(NonconvexSemidefinite.symmetric_pattern(pattern)); alg=ct_alg
    )
    return [sort!(perm[tree[i]]) for i in 1:length(tree)]
end

function NonconvexSemidefinite._chordal_extension(
    pattern::SparseMatrixCSC, alg::NonconvexSemidefinite.MFAlg
)
    return _cliquetrees_cliques(pattern, CliqueTrees.MF())
end

function NonconvexSemidefinite._chordal_extension(
    pattern::SparseMatrixCSC, alg::NonconvexSemidefinite.METISAlg
)
    return _cliquetrees_cliques(pattern, CliqueTrees.METIS())
end

function NonconvexSemidefinite._chordal_extension(
    pattern::SparseMatrixCSC, alg::NonconvexSemidefinite.BTAlg
)
    return _cliquetrees_cliques(pattern, CliqueTrees.BT())
end

function NonconvexSemidefinite._is_chordal(pattern::SparseMatrixCSC)
    return CliqueTrees.ischordal(Matrix(NonconvexSemidefinite.symmetric_pattern(pattern)))
end

end
