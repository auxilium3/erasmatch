"""Combine several host courses when no single one covers the home course.

Typical case: a 9-credit Italian course vs two 6-credit Spanish courses that split the same
syllabus. Choosing the best subset is a small weighted set-cover problem; a greedy pass is
the classic approximation and is plenty for a pool of ~10 candidates and bundles of <= 3.
"""

from __future__ import annotations

import numpy as np

from erasmatch.domain import StoredCourse
from erasmatch.matching.scoring import Candidate, ScoringConfig


def _best_per_home_topic(home: StoredCourse, cand: Candidate) -> np.ndarray:
    if cand.stored.topic_embeddings.size == 0 or not home.topics:
        return np.full(len(home.topics), -np.inf, dtype=np.float32)
    return (home.topic_embeddings @ cand.stored.topic_embeddings.T).max(axis=1)


def greedy_bundle(
    home: StoredCourse, candidates: list[Candidate], cfg: ScoringConfig
) -> list[Candidate] | None:
    """Pick courses one by one, each time the one that covers the most new home topics.

    Stops when coverage and credits are both satisfied, when nothing adds coverage, or at
    ``max_bundle_size``. Returns None unless at least two courses were selected.
    """
    if not home.topics or len(candidates) < 2:
        return None

    pool = candidates[: cfg.bundle_pool]
    per_cand = [_best_per_home_topic(home, c) for c in pool]
    best = np.full(len(home.topics), -np.inf, dtype=np.float32)
    selected: list[int] = []
    ects = 0.0

    def coverage(vec: np.ndarray) -> float:
        return float((vec >= cfg.topic_threshold).mean())

    while len(selected) < cfg.max_bundle_size:
        current = coverage(best)
        enough_credits = ects / home.course.ects + 1e-9 >= cfg.min_ects_ratio
        if current >= cfg.target_coverage and enough_credits:
            break
        pick: int | None = None
        pick_key: tuple[float, float] = (0.0, 0.0)
        for j, cand in enumerate(pool):
            if j in selected:
                continue
            gain = coverage(np.maximum(best, per_cand[j])) - current
            key = (gain, cand.similarity)
            if gain > 0 and (pick is None or key > pick_key):
                pick, pick_key = j, key
        if pick is None:
            break
        selected.append(pick)
        best = np.maximum(best, per_cand[pick])
        ects += pool[pick].stored.course.ects

    return [pool[j] for j in selected] if len(selected) >= 2 else None
