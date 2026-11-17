"""Repository interface. Two implementations: in-memory and PostgreSQL + pgvector."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from erasmatch.domain import StoredCourse, UniversityIn


class Repository(Protocol):
    kind: str

    def upsert_university(self, university: UniversityIn) -> None: ...

    def get_university(self, code: str) -> UniversityIn | None: ...

    def list_universities(self) -> list[UniversityIn]: ...

    def upsert_courses(self, courses: list[StoredCourse]) -> int: ...

    def get_course(self, university_code: str, course_code: str) -> StoredCourse | None: ...

    def list_courses(self, university_code: str) -> list[StoredCourse]: ...

    def search_similar(
        self, embedding: np.ndarray, university_code: str, limit: int
    ) -> list[tuple[StoredCourse, float]]:
        """Courses of ``university_code`` ordered by cosine similarity, highest first."""
        ...

    def embedding_models(self) -> set[str]: ...
