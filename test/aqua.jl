@testitem "Aqua" begin
    using Aqua, NonconvexSemidefinite, NonconvexCore, ChainRulesCore

    # All Aqua quality gates are enabled. No tests are skipped. The
    # `rrule`s are deliberate ChainRules extensions.
    Aqua.test_all(NonconvexSemidefinite; piracies=(treat_as_own=[rrule],))
end
