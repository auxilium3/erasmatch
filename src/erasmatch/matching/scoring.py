"""Deterministic scoring: topic coverage, ECTS compatibility and the combined score.

Everything here is pure numpy so it is fast, reproducible and easy to unit-test.
The LLM never changes these numbers; it only explains them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from erasmatch.config import Settings
from erasmatch.domain import CourseRef, MatchOption, StoredCourse, TopicAlignment, Verdict


@dataclass(frozen=True)
class ScoringConfig:
    topic_threshold: float
    candidates: int = 20
    min_ects_ratio: float = 1.0
    target_coverage: float = 0.75
    min_coverage: float = 0.5
    max_bundle_size: int = 3
    bundle_pool: int = 10
    bundle_penalty: float = 0.03
    weight_coverage: float = 0.6
    weight_similarity: float = 0.25
    weight_ects: float = 0.15

    @classmethod
    def from_settings(cls, settings: Settings, default_threshold: float) -> ScoringConfig:
        threshold = settings.topic_threshold
        return cls(
            topic_threshold=default_threshold if threshold is None else threshold,
            candidates=settings.candidates,
            min_ects_ratio=settings.min_ects_ratio,
            target_coverage=settings.target_coverage,
            min_coverage=settings.min_coverage,
            max_bundle_size=settings.max_bundle_size,
            bundle_penalty=settings.bundle_penalty,
            weight_coverage=settings.weight_coverage,
            weight_similarity=settings.weight_similarity,
            weight_ects=settings.weight_ects,
        )


@dataclass(frozen=True)
class Candidate:
    """A host course returned by retrieval, with its course-level similarity to the home course."""

    stored: StoredCourse
    similarity: float


@dataclass(frozen=True)
class Coverage:
    best_sim: np.ndarray  # (n_home,) best similarity for each home topic
    best_idx: np.ndarray  # (n_home,) index into the stacked host topics, -1 if none
    covered: np.ndarray  # (n_home,) bool

    @property
    def value(self) -> float:
        return float(self.covered.mean()) if self.covered.size else 0.0


def topic_coverage(home: np.ndarray, host: np.ndarray, threshold: float) -> Coverage:
    """For each home topic, find the most similar host topic and check it clears the threshold."""
    n_home = home.shape[0]
    if n_home == 0 or host.shape[0] == 0:
        return Coverage(
            best_sim=np.zeros(n_home, dtype=np.float32),
            best_idx=np.full(n_home, -1),
            covered=np.zeros(n_home, dtype=bool),
        )
    sims = home @ host.T
    best_idx = sims.argmax(axis=1)
    best_sim = sims[np.arange(n_home), best_idx]
    return Coverage(best_sim=best_sim, best_idx=best_idx, covered=best_sim >= threshold)


def ects_fit(home_ects: float, host_ects: float, min_ratio: float) -> tuple[float, bool]:
    """Score in [0, 1] (1 = enough credits) and whether the minimum ratio is met."""
    ratio = host_ects / home_ects
    ok = ratio + 1e-9 >= min_ratio
    score = 1.0 if min_ratio <= 0 else min(1.0, ratio / min_ratio)
    return score, ok


def combined_score(
    coverage: float, similarity: float, ects_score: float, n_courses: int, cfg: ScoringConfig
) -> float:
    total_w = cfg.weight_coverage + cfg.weight_similarity + cfg.weight_ects
    raw = (
        cfg.weight_coverage * coverage
        + cfg.weight_similarity * max(0.0, similarity)
        + cfg.weight_ects * ects_score
    ) / total_w
    # Prefer simpler options: each extra course in a bundle costs a little.
    raw -= cfg.bundle_penalty * (n_courses - 1)
    return round(max(0.0, raw), 4)


def verdict_for(coverage: float, ects_ok: bool, cfg: ScoringConfig) -> Verdict:
    if coverage >= cfg.target_coverage and ects_ok:
        return "strong"
    if coverage >= cfg.min_coverage:
        return "possible"
    return "weak"


def build_option(home: StoredCourse, members: list[Candidate], cfg: ScoringConfig) -> MatchOption:
    """Score one host course (or a bundle of them) against a home course."""
    owners: list[tuple[StoredCourse, str]] = [
        (m.stored, topic) for m in members for topic in m.stored.topics
    ]
    matrices = [m.stored.topic_embeddings for m in members if m.stored.topic_embeddings.size]
    host_topics = (
        np.vstack(matrices) if matrices else np.zeros((0, home.embedding.shape[0]), np.float32)
    )
    cov = topic_coverage(home.topic_embeddings, host_topics, cfg.topic_threshold)

    alignments: list[TopicAlignment] = []
    for i, home_topic in enumerate(home.topics):
        idx = int(cov.best_idx[i])
        owner, host_topic = owners[idx] if idx >= 0 else (None, None)
        alignments.append(
            TopicAlignment(
                home_topic=home_topic,
                host_topic=host_topic,
                host_course=owner.code if owner else None,
                similarity=round(float(cov.best_sim[i]), 4),
                covered=bool(cov.covered[i]),
            )
        )

    ects_total = round(sum(m.stored.course.ects for m in members), 2)
    e_score, e_ok = ects_fit(home.course.ects, ects_total, cfg.min_ects_ratio)
    similarity = max(m.similarity for m in members)
    coverage = round(cov.value, 4)
    return MatchOption(
        host_courses=[CourseRef.of(m.stored) for m in members],
        score=combined_score(coverage, similarity, e_score, len(members), cfg),
        verdict=verdict_for(coverage, e_ok, cfg),
        coverage=coverage,
        similarity=round(similarity, 4),
        ects_total=ects_total,
        ects_ok=e_ok,
        alignments=alignments,
        missing_topics=[a.home_topic for a in alignments if not a.covered],
    )
