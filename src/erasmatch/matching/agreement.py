"""Turn per-course options into a Learning Agreement draft.

Constraint: a host course can be used for only one home course. We assign greedily by
score, which is optimal when options don't conflict and a good approximation otherwise
(an exact solution would be a small integer program; LAs rarely exceed ~10 courses).
"""

from __future__ import annotations

from erasmatch.domain import MatchOption


def assign_options(
    options_by_home: dict[str, list[MatchOption]], min_coverage: float
) -> dict[str, MatchOption | None]:
    entries = [
        (opt.score, opt.coverage, home_code, opt)
        for home_code, opts in options_by_home.items()
        for opt in opts
        if opt.coverage >= min_coverage
    ]
    entries.sort(key=lambda e: (e[0], e[1]), reverse=True)

    assigned: dict[str, MatchOption | None] = dict.fromkeys(options_by_home)
    used: set[str] = set()
    for _score, _cov, home_code, opt in entries:
        if assigned[home_code] is not None:
            continue
        codes = {c.code for c in opt.host_courses}
        if codes & used:
            continue
        assigned[home_code] = opt
        used |= codes
    return assigned
