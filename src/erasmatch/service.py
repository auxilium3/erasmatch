"""Wiring: builds the repository, providers and engine from settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from erasmatch.config import Settings
from erasmatch.domain import Catalog, StoredCourse
from erasmatch.embeddings import Embedder, build_embedder
from erasmatch.llm import LLMClient, build_llm
from erasmatch.llm.explainer import Explainer
from erasmatch.matching.engine import MatchEngine
from erasmatch.matching.scoring import ScoringConfig
from erasmatch.storage import build_repository
from erasmatch.storage.base import Repository
from erasmatch.text import course_document, effective_topics


class CatalogService:
    def __init__(self, repo: Repository, embedder: Embedder) -> None:
        self.repo = repo
        self.embedder = embedder

    def import_catalog(self, catalog: Catalog) -> int:
        """Embed every course and its topics, then upsert. Returns the number of courses."""
        uni = catalog.university
        self.repo.upsert_university(uni)

        topics_per_course = [effective_topics(c) for c in catalog.courses]
        course_vecs = self.embedder.embed([course_document(c) for c in catalog.courses])
        flat_topics = [t for topics in topics_per_course for t in topics]
        topic_vecs = self.embedder.embed(flat_topics)

        stored: list[StoredCourse] = []
        offset = 0
        for course, topics, vec in zip(
            catalog.courses, topics_per_course, course_vecs, strict=True
        ):
            if course.language is None:
                course = course.model_copy(update={"language": uni.language})
            stored.append(
                StoredCourse(
                    university_code=uni.code,
                    course=course,
                    embedding=vec,
                    topics=topics,
                    topic_embeddings=topic_vecs[offset : offset + len(topics)],
                    embedding_model=self.embedder.name,
                )
            )
            offset += len(topics)
        return self.repo.upsert_courses(stored)

    def import_files(self, paths: list[Path]) -> dict[str, int]:
        from erasmatch.ingest import load_catalog

        return {str(p): self.import_catalog(load_catalog(p)) for p in paths}


@dataclass
class Services:
    settings: Settings
    repo: Repository
    embedder: Embedder
    llm: LLMClient | None
    catalog: CatalogService
    engine: MatchEngine


def build_services(
    settings: Settings,
    repo: Repository | None = None,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    use_llm: bool = True,
) -> Services:
    repo = repo or build_repository(settings.database_url)
    embedder = embedder or build_embedder(settings)
    if llm is None and use_llm:
        llm = build_llm(settings)
    cfg = ScoringConfig.from_settings(settings, embedder.default_topic_threshold)
    explainer = Explainer(llm, settings.explanation_language)
    return Services(
        settings=settings,
        repo=repo,
        embedder=embedder,
        llm=llm,
        catalog=CatalogService(repo, embedder),
        engine=MatchEngine(repo, embedder.name, cfg, explainer),
    )
