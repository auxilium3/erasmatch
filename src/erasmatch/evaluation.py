"""Offline evaluation against hand-labelled equivalences.

Gold file format (YAML):

    home_university: UNIEX
    host_university: UEJ
    pairs:
      - home: RETI
        accept: [RED, "RED+TDR"]   # "+" joins a bundle; order does not matter

Metrics: hit@1, hit@3 and mean reciprocal rank over the ranked options.
Use it to compare embedding models and to calibrate ERASMATCH_TOPIC_THRESHOLD.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from erasmatch.matching.engine import MatchEngine


def _key(spec: str) -> str:
    return "+".join(sorted(part.strip() for part in spec.split("+")))


class GoldPair(BaseModel):
    home: str
    accept: list[str] = Field(min_length=1)


class GoldSet(BaseModel):
    home_university: str
    host_university: str
    pairs: list[GoldPair] = Field(min_length=1)

    @classmethod
    def load(cls, path: Path) -> GoldSet:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class PairResult(BaseModel):
    home: str
    expected: list[str]
    predicted: str | None
    rank: int | None


class EvalReport(BaseModel):
    n: int
    hit_at_1: float
    hit_at_3: float
    mrr: float
    topic_threshold: float
    pairs: list[PairResult]


def evaluate(engine: MatchEngine, gold: GoldSet, k: int = 10) -> EvalReport:
    results: list[PairResult] = []
    for pair in gold.pairs:
        accepted = {_key(a) for a in pair.accept}
        result = engine.match(gold.home_university, pair.home, gold.host_university, top_k=k)
        keys = [o.key for o in result.options]
        rank = next((i + 1 for i, key in enumerate(keys) if key in accepted), None)
        results.append(
            PairResult(
                home=pair.home,
                expected=sorted(accepted),
                predicted=keys[0] if keys else None,
                rank=rank,
            )
        )
    n = len(results)
    return EvalReport(
        n=n,
        hit_at_1=round(sum(r.rank == 1 for r in results) / n, 3),
        hit_at_3=round(sum(r.rank is not None and r.rank <= 3 for r in results) / n, 3),
        mrr=round(sum(1 / r.rank for r in results if r.rank) / n, 3),
        topic_threshold=engine.cfg.topic_threshold,
        pairs=results,
    )
