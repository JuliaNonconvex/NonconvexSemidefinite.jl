using TestItemRunner

# CI splits the suite into parallel jobs: CI_TEST_GROUP = "<index>/<n>"
# runs the files whose (1-based) position in this list is congruent to
# `index` modulo `n`. Keep the list in sync with the test files that
# actually define @testitem blocks (plain scripts without test items are
# not listed). The order is NOT alphabetical: the heavy files (chordal
# decomposition/cutting plane, the JET subprocess, the Kočvara SDD and the
# core semidefinite suites) are placed first so that each group gets
# exactly one of them.
const CI_GROUP_FILES = String[
    "chordal_sets.jl",
    "jet.jl",
    "kocvara2020.jl",
    "semidefinite.jl",
    "aqua.jl",
    "block_relaxation.jl",
    "completion.jl",
    "coverage_extra.jl",
    "features_extra.jl",
    "ks.jl",
]

function _group_filter()
    spec = get(ENV, "CI_TEST_GROUP", nothing)
    spec === nothing && return ti -> true
    parts = split(spec, '/')
    length(parts) == 2 || return ti -> true
    idx = tryparse(Int, parts[1])
    n = tryparse(Int, parts[2])
    if idx === nothing || n === nothing || !(1 <= idx <= n)
        return ti -> true
    end
    mine = CI_GROUP_FILES[idx:n:length(CI_GROUP_FILES)]
    return ti -> basename(ti.filename) in mine
end

@run_package_tests(filter=_group_filter())
