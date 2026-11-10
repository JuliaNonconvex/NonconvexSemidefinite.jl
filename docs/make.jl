using DocumenterFragments: build_fragment

# The `@docs` blocks in index.md reference the `SDPBarrierOptions` /
# `SDPBarrierAlg` alias bindings (docstrings only exist on their canonical
# `BarrierOptions` / `BarrierAlg` names) and `@ref` links to NonconvexCore
# bindings only resolve in the assembled Nonconvex.jl documentation site,
# so tolerate broken docstring lookups and cross-references in this
# standalone preview build.
build_fragment(
    @__DIR__; checkdocs=:public, doctest=false, warnonly=[:cross_references, :docs_block]
)
