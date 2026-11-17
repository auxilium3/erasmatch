"""Small text helpers shared by embedders and the matching pipeline."""

from __future__ import annotations

import re
import unicodedata

from erasmatch.domain import CourseIn

_BULLET = re.compile(r"^\s*(?:[-*•●▪]|\d+[.)]|[a-z][.)])\s+", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.;:!?])\s+|\n+")
_MIN_TOPIC_WORDS = 2


def fold(text: str) -> str:
    """Lowercase and strip accents so 'Programación' and 'programacion' look the same."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def clean_topic(text: str) -> str:
    text = _BULLET.sub("", text).strip()
    return text.strip(" .;:,")


def split_topics(text: str) -> list[str]:
    """Turn a free-text syllabus into topic-sized chunks (bullets or sentences)."""
    parts = (clean_topic(p) for p in _SENTENCE_SPLIT.split(text))
    return [p for p in parts if len(p.split()) >= _MIN_TOPIC_WORDS]


def effective_topics(course: CourseIn) -> list[str]:
    """Topics used for matching: declared ones, else derived from the description, else title."""
    declared = [clean_topic(t) for t in course.topics if clean_topic(t)]
    if declared:
        return declared
    derived = split_topics(course.description)
    return derived or [course.title]


def course_document(course: CourseIn) -> str:
    """The text that represents a whole course for course-level retrieval."""
    parts = [course.title]
    if course.description:
        parts.append(course.description)
    if course.topics:
        parts.append("Topics: " + "; ".join(course.topics))
    return "\n".join(parts)
