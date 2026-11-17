"""PostgreSQL + pgvector repository.

Retrieval is an exact cosine search (`<=>`) filtered by university. Catalogs hold a few
hundred courses per university, so a sequential scan is fast and returns exact results;
see the README for when an HNSW index would be worth adding.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import create_engine, delete, distinct, select
from sqlalchemy.orm import Session, sessionmaker

from erasmatch.domain import CourseIn, StoredCourse, UniversityIn
from erasmatch.storage.models import CourseRow, TopicRow, UniversityRow


def _to_stored(row: CourseRow) -> StoredCourse:
    course = CourseIn(
        code=row.code,
        title=row.title,
        ects=row.ects,
        language=row.language,
        description=row.description or "",
        topics=list(row.declared_topics or []),
        semester=row.semester,
        level=row.level,  # type: ignore[arg-type]
        url=row.url,
    )
    dim = len(row.embedding)
    topic_matrix = (
        np.asarray([t.embedding for t in row.topics], dtype=np.float32)
        if row.topics
        else np.zeros((0, dim), dtype=np.float32)
    )
    return StoredCourse(
        id=row.id,
        university_code=row.university_code,
        course=course,
        embedding=np.asarray(row.embedding, dtype=np.float32),
        topics=[t.text for t in row.topics],
        topic_embeddings=topic_matrix,
        embedding_model=row.embedding_model,
    )


class PostgresRepository:
    kind = "postgres"

    def __init__(self, database_url: str) -> None:
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self._session = sessionmaker(self.engine, expire_on_commit=False)

    def session(self) -> Session:
        return self._session()

    def upsert_university(self, university: UniversityIn) -> None:
        with self.session() as s, s.begin():
            s.merge(UniversityRow(**university.model_dump()))

    def get_university(self, code: str) -> UniversityIn | None:
        with self.session() as s:
            row = s.get(UniversityRow, code)
            return _university(row) if row else None

    def list_universities(self) -> list[UniversityIn]:
        with self.session() as s:
            rows = s.scalars(select(UniversityRow).order_by(UniversityRow.code))
            return [_university(r) for r in rows]

    def upsert_courses(self, courses: list[StoredCourse]) -> int:
        with self.session() as s, s.begin():
            for stored in courses:
                # Replace-on-import keeps topics and embeddings consistent with the new text.
                s.execute(
                    delete(CourseRow).where(
                        CourseRow.university_code == stored.university_code,
                        CourseRow.code == stored.code,
                    )
                )
                c = stored.course
                row = CourseRow(
                    university_code=stored.university_code,
                    code=c.code,
                    title=c.title,
                    ects=c.ects,
                    language=c.language,
                    description=c.description,
                    declared_topics=c.topics,
                    semester=c.semester,
                    level=c.level,
                    url=c.url,
                    embedding=stored.embedding.tolist(),
                    embedding_model=stored.embedding_model,
                    topics=[
                        TopicRow(position=i, text=text, embedding=vec.tolist())
                        for i, (text, vec) in enumerate(
                            zip(stored.topics, stored.topic_embeddings, strict=True)
                        )
                    ],
                )
                s.add(row)
                s.flush()
                stored.id = row.id
        return len(courses)

    def get_course(self, university_code: str, course_code: str) -> StoredCourse | None:
        with self.session() as s:
            row = s.scalar(
                select(CourseRow).where(
                    CourseRow.university_code == university_code, CourseRow.code == course_code
                )
            )
            return _to_stored(row) if row else None

    def list_courses(self, university_code: str) -> list[StoredCourse]:
        with self.session() as s:
            rows = s.scalars(
                select(CourseRow)
                .where(CourseRow.university_code == university_code)
                .order_by(CourseRow.code)
            )
            return [_to_stored(r) for r in rows]

    def search_similar(
        self, embedding: np.ndarray, university_code: str, limit: int
    ) -> list[tuple[StoredCourse, float]]:
        distance = CourseRow.embedding.cosine_distance(embedding.tolist()).label("distance")
        with self.session() as s:
            rows = s.execute(
                select(CourseRow, distance)
                .where(CourseRow.university_code == university_code)
                .order_by(distance)
                .limit(limit)
            ).all()
            return [(_to_stored(row), 1.0 - float(dist)) for row, dist in rows]

    def embedding_models(self) -> set[str]:
        with self.session() as s:
            return set(s.scalars(select(distinct(CourseRow.embedding_model))))


def _university(row: UniversityRow) -> UniversityIn:
    return UniversityIn(code=row.code, name=row.name, country=row.country, language=row.language)
