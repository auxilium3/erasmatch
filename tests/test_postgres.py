"""Integration tests against a real PostgreSQL + pgvector database.

Run with:  ERASMATCH_TEST_DATABASE_URL=postgresql+psycopg://user:pass@host/db pytest -m postgres
WARNING: the database's public schema is dropped and recreated.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from sqlalchemy import create_engine, text

from erasmatch.config import Settings
from erasmatch.embeddings.hashing import HashingEmbedder
from erasmatch.evaluation import GoldSet, evaluate
from erasmatch.service import Services, build_services
from erasmatch.storage import run_migrations
from erasmatch.storage.memory import InMemoryRepository
from erasmatch.storage.postgres import PostgresRepository
from tests.conftest import GOLD, HOME_CATALOG, HOST_CATALOG

URL = os.environ.get("ERASMATCH_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not URL, reason="ERASMATCH_TEST_DATABASE_URL not set"),
]


@pytest.fixture(scope="module")
def pg_services() -> Services:
    assert URL
    engine = create_engine(URL)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    engine.dispose()
    embedder = HashingEmbedder()
    run_migrations(URL, embedding_dim=embedder.dim)
    settings = Settings(database_url=URL, embedder="hashing", llm="none", _env_file=None)  # type: ignore[call-arg]
    services = build_services(settings, repo=PostgresRepository(URL), embedder=embedder)
    services.catalog.import_files([HOME_CATALOG, HOST_CATALOG])
    return services


def test_reimport_is_idempotent(pg_services: Services) -> None:
    before = {c.code: c.id for c in pg_services.repo.list_courses("UEJ")}
    pg_services.catalog.import_files([HOST_CATALOG])
    after = pg_services.repo.list_courses("UEJ")
    assert len(after) == len(before) == 15
    with pg_services.repo.engine.connect() as conn:  # type: ignore[attr-defined]
        topics = conn.execute(text("SELECT count(*) FROM topics")).scalar_one()
    expected = sum(len(c.topics) for c in after) + sum(
        len(c.topics) for c in pg_services.repo.list_courses("UNIEX")
    )
    assert topics == expected


def test_round_trip_preserves_course(pg_services: Services) -> None:
    stored = pg_services.repo.get_course("UEJ", "AC")
    assert stored is not None
    assert stored.course.language == "en"
    assert stored.course.ects == 6
    assert stored.topics[2] == "ARM instruction set and assembly programming"
    assert stored.topic_embeddings.shape == (8, 1024)
    np.testing.assert_allclose(np.linalg.norm(stored.embedding), 1.0, rtol=1e-4)


def test_search_matches_in_memory_ranking(pg_services: Services) -> None:
    memory = InMemoryRepository()
    mem_services = build_services(
        pg_services.settings, repo=memory, embedder=pg_services.embedder, use_llm=False
    )
    mem_services.catalog.import_files([HOME_CATALOG, HOST_CATALOG])
    home = pg_services.repo.get_course("UNIEX", "BASIDATI")
    assert home is not None
    pg_hits = pg_services.repo.search_similar(home.embedding, "UEJ", 5)
    mem_hits = memory.search_similar(home.embedding, "UEJ", 5)
    assert [c.code for c, _ in pg_hits] == [c.code for c, _ in mem_hits]
    for (_, a), (_, b) in zip(pg_hits, mem_hits, strict=True):
        assert a == pytest.approx(b, abs=1e-4)


def test_evaluation_on_postgres(pg_services: Services) -> None:
    report = evaluate(pg_services.engine, GoldSet.load(GOLD))
    assert report.hit_at_1 >= 0.85
