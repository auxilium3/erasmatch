"""Natural-language explanations of a match.

Design rule: the LLM explains, it never scores. It receives the numbers and the topic
alignment computed by the deterministic pipeline and turns them into two short sentences a
student or coordinator can read. If the LLM is disabled or fails, a template is used.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from erasmatch.domain import Explanation, MatchOption, StoredCourse
from erasmatch.errors import ProviderError
from erasmatch.llm import LLMClient

log = logging.getLogger(__name__)

_LANGUAGES = {"en": "English", "it": "Italian", "es": "Spanish", "fr": "French", "de": "German"}

SYSTEM_PROMPT = """You help university students prepare Erasmus Learning Agreements.
You receive a course from the student's home university, one or more candidate courses
at the host university, and a topic-by-topic alignment computed by an embedding model.
Rules:
- Use only the information provided. Never invent topics, credits or course content.
- Do not change or second-guess the numbers; explain them.
- "summary": at most two sentences on how well the candidate covers the home course.
- "gaps": one sentence naming the uncovered home topics, or null if none are missing.
- Write in {language}."""


class _LLMExplanation(BaseModel):
    summary: str = Field(max_length=600)
    gaps: str | None = Field(default=None, max_length=400)


def template_explanation(home: StoredCourse, option: MatchOption) -> Explanation:
    covered = sum(a.covered for a in option.alignments)
    titles = " + ".join(c.title for c in option.host_courses)
    summary = (
        f"{titles} covers {covered} of {len(option.alignments)} topics of "
        f"{home.course.title} ({option.coverage:.0%}); credits {option.ects_total:g} vs "
        f"{home.course.ects:g} required."
    )
    gaps = "Missing: " + "; ".join(option.missing_topics) + "." if option.missing_topics else None
    return Explanation(summary=summary, gaps=gaps, source="template")


def _prompt(home: StoredCourse, option: MatchOption) -> str:
    lines = [
        f"HOME COURSE: {home.course.title} ({home.course.ects:g} ECTS, {home.university_code})",
        "CANDIDATE: "
        + " + ".join(f"{c.title} [{c.code}] ({c.ects:g} ECTS)" for c in option.host_courses),
        f"COVERAGE: {option.coverage:.0%} of home topics | VERDICT: {option.verdict} | "
        f"CREDITS OK: {'yes' if option.ects_ok else 'no'} ({option.ects_total:g} ECTS)",
        "ALIGNMENT (home topic -> closest host topic, similarity, covered):",
    ]
    for a in option.alignments:
        lines.append(
            f"- {a.home_topic} -> {a.host_topic or 'none'} "
            f"[{a.host_course or '-'}] {a.similarity:.2f} {'YES' if a.covered else 'NO'}"
        )
    return "\n".join(lines)


class Explainer:
    def __init__(self, llm: LLMClient | None, language: str = "en") -> None:
        self.llm = llm
        self.language = _LANGUAGES.get(language, language)

    def explain(self, home: StoredCourse, option: MatchOption) -> Explanation:
        if self.llm is None:
            return template_explanation(home, option)
        try:
            out = self.llm.complete_json(
                SYSTEM_PROMPT.format(language=self.language),
                _prompt(home, option),
                _LLMExplanation,
            )
        except ProviderError as exc:
            log.warning("LLM explanation failed, using template: %s", exc)
            return template_explanation(home, option)
        return Explanation(summary=out.summary, gaps=out.gaps, source="llm")
