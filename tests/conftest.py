from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel

from erasmatch.config import Settings
from erasmatch.domain import Catalog, CourseIn, UniversityIn
from erasmatch.embeddings import l2_normalize
from erasmatch.embeddings.hashing import HashingEmbedder
from erasmatch.errors import ProviderError
from erasmatch.service import Services, build_services
from erasmatch.storage.memory import InMemoryRepository

EXAMPLES = Path(__file__).resolve().parent.parent / "data" / "examples"
HOME_CATALOG = EXAMPLES / "home_uniex.yaml"
HOST_CATALOG = EXAMPLES / "host_uej.yaml"
GOLD = EXAMPLES / "gold_uniex_uej.yaml"


class ConceptEmbedder:
    """Fully predictable embedder: each known word is one dimension."""

    name = "concept-test"
    default_topic_threshold = 0.5
    CONCEPTS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta")
    dim = len(CONCEPTS) + 1  # last dimension = "unknown words"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for word in re.findall(r"[a-z]+", text.lower()):
                if word in self.CONCEPTS:
                    vec[self.CONCEPTS.index(word)] += 1
            if not vec.any():
                vec[-1] = 1
            rows.append(vec)
        return l2_normalize(np.stack(rows)) if rows else np.zeros((0, self.dim), np.float32)


class FakeLLM:
    name = "fake-llm"

    def __init__(self, factory: Callable[[type[BaseModel]], BaseModel] | None = None) -> None:
        self.factory = factory
        self.calls: list[tuple[str, str, type[BaseModel]]] = []

    def complete_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        self.calls.append((system, user, schema))
        if self.factory is None:
            raise ProviderError("LLM unavailable")
        return self.factory(schema)


def course(code: str, topics: list[str], ects: float = 6, title: str | None = None) -> CourseIn:
    return CourseIn(code=code, title=title or f"Course {code}", ects=ects, topics=topics)


def uni(code: str) -> UniversityIn:
    return UniversityIn(code=code, name=f"University {code}", country="IT", language="it")


@pytest.fixture
def settings() -> Settings:
    return Settings(
        database_url="memory://",
        embedder="hashing",
        llm="none",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def concept_services(settings: Settings) -> Services:
    return build_services(
        settings, repo=InMemoryRepository(), embedder=ConceptEmbedder(), use_llm=False
    )


@pytest.fixture
def example_services(settings: Settings) -> Services:
    services = build_services(
        settings, repo=InMemoryRepository(), embedder=HashingEmbedder(), use_llm=False
    )
    services.catalog.import_files([HOME_CATALOG, HOST_CATALOG])
    return services


def load(path: Path) -> Catalog:
    from erasmatch.ingest import load_catalog

    return load_catalog(path)
