from __future__ import annotations

import numpy as np
import pytest

from erasmatch.domain import CourseRef, MatchOption
from erasmatch.matching.agreement import assign_options
from erasmatch.matching.scoring import (
    ScoringConfig,
    combined_score,
    ects_fit,
    topic_coverage,
    verdict_for,
)

CFG = ScoringConfig(topic_threshold=0.5)


def test_topic_coverage_counts_topics_above_threshold() -> None:
    home = np.eye(3, dtype=np.float32)
    host = np.array([[1, 0, 0], [0, 0.6, 0.8]], dtype=np.float32)
    cov = topic_coverage(home, host, threshold=0.5)
    assert cov.covered.tolist() == [True, True, True]
    assert cov.best_idx.tolist() == [0, 1, 1]
    assert cov.value == 1.0
    cov = topic_coverage(home, host, threshold=0.7)
    assert cov.covered.tolist() == [True, False, True]
    assert cov.value == pytest.approx(2 / 3)


def test_topic_coverage_with_no_host_topics() -> None:
    cov = topic_coverage(np.eye(2, dtype=np.float32), np.zeros((0, 2), np.float32), 0.5)
    assert cov.value == 0.0
    assert cov.best_idx.tolist() == [-1, -1]


@pytest.mark.parametrize(
    ("home", "host", "ratio", "score", "ok"),
    [
        (6, 6, 1.0, 1.0, True),
        (9, 6, 1.0, 2 / 3, False),
        (9, 12, 1.0, 1.0, True),
        (9, 6, 0.6, 1.0, True),
    ],
)
def test_ects_fit(home: float, host: float, ratio: float, score: float, ok: bool) -> None:
    s, o = ects_fit(home, host, ratio)
    assert s == pytest.approx(score)
    assert o is ok


def test_combined_score_penalises_bundles() -> None:
    single = combined_score(1.0, 0.8, 1.0, n_courses=1, cfg=CFG)
    bundle = combined_score(1.0, 0.8, 1.0, n_courses=2, cfg=CFG)
    assert single == pytest.approx((0.6 + 0.25 * 0.8 + 0.15) / 1.0)
    assert bundle == pytest.approx(single - CFG.bundle_penalty)


def test_verdicts() -> None:
    assert verdict_for(0.8, True, CFG) == "strong"
    assert verdict_for(0.8, False, CFG) == "possible"
    assert verdict_for(0.5, True, CFG) == "possible"
    assert verdict_for(0.4, True, CFG) == "weak"


def _option(codes: list[str], score: float, coverage: float = 1.0) -> MatchOption:
    return MatchOption(
        host_courses=[CourseRef(university="H", code=c, title=c, ects=6) for c in codes],
        score=score,
        verdict="strong",
        coverage=coverage,
        similarity=0.9,
        ects_total=6 * len(codes),
        ects_ok=True,
        alignments=[],
        missing_topics=[],
    )


def test_assignment_never_reuses_a_host_course() -> None:
    options = {
        "A": [_option(["X"], 0.9), _option(["Y"], 0.5)],
        "B": [_option(["X"], 0.8), _option(["Z", "W"], 0.7)],
        "C": [_option(["W"], 0.6)],
    }
    result = assign_options(options, min_coverage=0.5)
    assert result["A"] is not None and result["A"].key == "X"
    assert result["B"] is not None and result["B"].key == "W+Z"
    assert result["C"] is None  # W already used by B's bundle


def test_assignment_respects_min_coverage() -> None:
    result = assign_options({"A": [_option(["X"], 0.9, coverage=0.3)]}, min_coverage=0.5)
    assert result == {"A": None}
