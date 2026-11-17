"""Domain models.

Pydantic models are the public contract (catalog files, API payloads, CLI JSON output).
``StoredCourse`` is the internal representation that also carries the embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

CODE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$"

Verdict = Literal["strong", "possible", "weak"]


class UniversityIn(BaseModel):
    code: str = Field(pattern=CODE_PATTERN, description="Short unique code, e.g. 'UAL'.")
    name: str = Field(min_length=2)
    country: str = Field(min_length=2, max_length=2, description="ISO 3166-1 alpha-2")
    language: str = Field(min_length=2, max_length=5, description="Main teaching language")


class CourseIn(BaseModel):
    """A course as described in a university catalog."""

    code: str = Field(pattern=CODE_PATTERN)
    title: str = Field(min_length=2)
    ects: float = Field(gt=0, le=60, description="Credits. 1 Italian CFU = 1 ECTS.")
    language: str | None = Field(None, max_length=5)
    description: str = ""
    topics: list[str] = Field(default_factory=list, description="Syllabus items, one per entry")
    semester: str | None = None
    level: Literal["bachelor", "master"] | None = None
    url: str | None = None


class Catalog(BaseModel):
    """File/API format for importing one university with its courses."""

    university: UniversityIn
    courses: list[CourseIn] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_codes(self) -> Catalog:
        codes = [c.code for c in self.courses]
        duplicates = sorted({c for c in codes if codes.count(c) > 1})
        if duplicates:
            raise ValueError(f"Duplicate course codes: {', '.join(duplicates)}")
        return self


@dataclass
class StoredCourse:
    """A course plus everything needed to match it."""

    university_code: str
    course: CourseIn
    embedding: np.ndarray  # (dim,), L2-normalised
    topics: list[str]  # effective topics used for matching
    topic_embeddings: np.ndarray  # (len(topics), dim), L2-normalised
    embedding_model: str
    id: UUID | None = field(default=None)

    @property
    def code(self) -> str:
        return self.course.code

    @property
    def key(self) -> str:
        return f"{self.university_code}:{self.course.code}"


# ---------------------------------------------------------------------------
# Matching output
# ---------------------------------------------------------------------------


class CourseRef(BaseModel):
    university: str
    code: str
    title: str
    ects: float
    language: str | None = None

    @classmethod
    def of(cls, stored: StoredCourse) -> CourseRef:
        c = stored.course
        return cls(
            university=stored.university_code,
            code=c.code,
            title=c.title,
            ects=c.ects,
            language=c.language,
        )


class TopicAlignment(BaseModel):
    home_topic: str
    host_topic: str | None
    host_course: str | None
    similarity: float
    covered: bool


class Explanation(BaseModel):
    summary: str
    gaps: str | None = None
    source: Literal["llm", "template"]


class MatchOption(BaseModel):
    """One way to cover a home course: a single host course or a bundle of several."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    host_courses: list[CourseRef]
    score: float
    verdict: Verdict
    coverage: float = Field(description="Share of home topics matched above the threshold")
    similarity: float = Field(description="Course-level cosine similarity (max over bundle)")
    ects_total: float
    ects_ok: bool
    alignments: list[TopicAlignment]
    missing_topics: list[str]
    explanation: Explanation | None = None

    @property
    def key(self) -> str:
        return "+".join(sorted(c.code for c in self.host_courses))

    @property
    def is_bundle(self) -> bool:
        return len(self.host_courses) > 1


class MatchResult(BaseModel):
    home_course: CourseRef
    host_university: str
    topic_threshold: float
    options: list[MatchOption]


class LearningAgreementRow(BaseModel):
    home_course: CourseRef
    option: MatchOption | None
    note: str | None = None


class LearningAgreementDraft(BaseModel):
    home_university: str
    host_university: str
    rows: list[LearningAgreementRow]
    home_ects_total: float
    host_ects_total: float
    unmatched: list[str]
