"""Extract a structured course from a syllabus (PDF or text) with an LLM.

The result is returned for human review, never saved automatically: syllabus PDFs vary a
lot and a wrong credit value would silently corrupt every match that uses the course.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from erasmatch.domain import CourseIn
from erasmatch.llm import LLMClient

MAX_CHARS = 15_000

SYSTEM_PROMPT = """You extract course information from university syllabi
(e.g. Italian "scheda insegnamento", Spanish "guía docente", English "module descriptor").
Return JSON only. Rules:
- Copy facts from the text; do not invent anything. Use null when a field is absent.
- "ects": number of credits (CFU and ECTS are equivalent).
- "topics": the syllabus/contents as a list of short items (max 25), in the original language.
- "description": 1-3 sentences summarising objectives, in the original language.
- "language": ISO 639-1 code of the teaching language if stated, else of the text."""


class _Extracted(BaseModel):
    code: str | None = None
    title: str
    ects: float | None = None
    language: str | None = None
    description: str = ""
    topics: list[str] = Field(default_factory=list)
    semester: str | None = None


class ExtractionResult(BaseModel):
    course: CourseIn
    warnings: list[str]


def extract_course(text: str, llm: LLMClient, fallback_code: str = "TODO") -> ExtractionResult:
    truncated = text[:MAX_CHARS]
    raw = llm.complete_json(SYSTEM_PROMPT, f"SYLLABUS TEXT:\n{truncated}", _Extracted)

    warnings: list[str] = []
    if len(text) > MAX_CHARS:
        warnings.append(f"Text truncated to {MAX_CHARS} characters.")
    code = re.sub(r"[^A-Za-z0-9._-]+", "-", (raw.code or "").strip()).strip("-._")
    if not code:
        code = fallback_code
        warnings.append("No course code found; set 'code' before importing.")
    ects = raw.ects
    if ects is None or not 0 < ects <= 60:
        warnings.append(f"Credits missing or implausible ({ects}); please check.")
        ects = 6.0
    if not raw.topics:
        warnings.append("No syllabus topics found; matching will rely on the description.")

    course = CourseIn(
        code=code[:40],
        title=raw.title.strip(),
        ects=ects,
        language=(raw.language or None) and raw.language[:5].lower(),
        description=raw.description.strip(),
        topics=[t.strip() for t in raw.topics if t.strip()][:25],
        semester=raw.semester,
    )
    return ExtractionResult(course=course, warnings=warnings)
