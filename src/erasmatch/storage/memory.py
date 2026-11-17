"""In-process repository: no database needed. Used by tests and `--catalog` CLI runs."""

from __future__ import annotations

import uuid

import numpy as np

from erasmatch.domain import StoredCourse, UniversityIn


class InMemoryRepository:
    kind = "memory"

    def __init__(self) -> None:
        self._universities: dict[str, UniversityIn] = {}
        self._courses: dict[str, dict[str, StoredCourse]] = {}

    def upsert_university(self, university: UniversityIn) -> None:
        self._universities[university.code] = university
        self._courses.setdefault(university.code, {})

    def get_university(self, code: str) -> UniversityIn | None:
        return self._universities.get(code)

    def list_universities(self) -> list[UniversityIn]:
        return sorted(self._universities.values(), key=lambda u: u.code)

    def upsert_courses(self, courses: list[StoredCourse]) -> int:
        for course in courses:
            if course.university_code not in self._universities:
                raise ValueError(f"Unknown university {course.university_code}")
            existing = self._courses[course.university_code].get(course.code)
            course.id = existing.id if existing else uuid.uuid4()
            self._courses[course.university_code][course.code] = course
        return len(courses)

    def get_course(self, university_code: str, course_code: str) -> StoredCourse | None:
        return self._courses.get(university_code, {}).get(course_code)

    def list_courses(self, university_code: str) -> list[StoredCourse]:
        return sorted(self._courses.get(university_code, {}).values(), key=lambda c: c.code)

    def search_similar(
        self, embedding: np.ndarray, university_code: str, limit: int
    ) -> list[tuple[StoredCourse, float]]:
        courses = list(self._courses.get(university_code, {}).values())
        if not courses:
            return []
        matrix = np.stack([c.embedding for c in courses])
        sims = matrix @ embedding
        order = np.argsort(-sims)[:limit]
        return [(courses[i], float(sims[i])) for i in order]

    def embedding_models(self) -> set[str]:
        return {c.embedding_model for by_code in self._courses.values() for c in by_code.values()}
