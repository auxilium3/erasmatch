from __future__ import annotations

import pytest
from pydantic import BaseModel

from erasmatch.domain import Catalog
from erasmatch.errors import ModelMismatchError, NotFoundError
from erasmatch.evaluation import GoldSet, evaluate
from erasmatch.llm.explainer import Explainer
from erasmatch.service import Services
from tests.conftest import GOLD, ConceptEmbedder, FakeLLM, course, uni


def _setup(services: Services) -> None:
    services.catalog.import_catalog(
        Catalog(
            university=uni("HOME"),
            courses=[
                course("NET", ["alpha", "beta", "gamma", "delta"], ects=9),
                course("DB", ["epsilon", "zeta"], ects=6),
            ],
        )
    )
    services.catalog.import_catalog(
        Catalog(
            university=uni("HOST"),
            courses=[
                course("N1", ["alpha", "beta"], ects=6),
                course("N2", ["gamma", "delta"], ects=6),
                course("N3", ["alpha"], ects=6),
                course("D1", ["epsilon", "zeta"], ects=6),
                course("X", ["eta"], ects=6),
            ],
        )
    )


def test_single_course_match_ranks_full_coverage_first(concept_services: Services) -> None:
    _setup(concept_services)
    result = concept_services.engine.match("HOME", "DB", "HOST")
    best = result.options[0]
    assert best.key == "D1"
    assert best.coverage == 1.0
    assert best.verdict == "strong"
    assert best.missing_topics == []
    # Courses sharing no topic are never proposed.
    assert "X" not in {o.key for o in result.options}


def test_bundle_is_proposed_when_no_single_course_is_enough(concept_services: Services) -> None:
    _setup(concept_services)
    result = concept_services.engine.match("HOME", "NET", "HOST")
    best = result.options[0]
    assert best.key == "N1+N2"
    assert best.is_bundle
    assert best.coverage == 1.0
    assert best.ects_total == 12 and best.ects_ok
    singles = [o for o in result.options if not o.is_bundle]
    assert all(o.coverage == 0.5 for o in singles if o.key in {"N1", "N2"})
    assert {a.host_course for a in best.alignments} == {"N1", "N2"}


def test_learning_agreement_uses_each_host_course_once(concept_services: Services) -> None:
    _setup(concept_services)
    draft = concept_services.engine.learning_agreement("HOME", ["NET", "DB", "NET"], "HOST")
    assert [r.home_course.code for r in draft.rows] == ["NET", "DB"]
    used = [c.code for r in draft.rows if r.option for c in r.option.host_courses]
    assert sorted(used) == ["D1", "N1", "N2"]
    assert draft.home_ects_total == 15
    assert draft.host_ects_total == 18
    assert draft.unmatched == []


def test_unknown_courses_and_universities(concept_services: Services) -> None:
    _setup(concept_services)
    with pytest.raises(NotFoundError):
        concept_services.engine.match("HOME", "NOPE", "HOST")
    with pytest.raises(NotFoundError):
        concept_services.engine.match("HOME", "NET", "NOPE")


def test_model_mismatch_is_detected(concept_services: Services) -> None:
    _setup(concept_services)

    class OtherEmbedder(ConceptEmbedder):
        name = "another-model"

    concept_services.catalog.embedder = OtherEmbedder()
    concept_services.catalog.import_catalog(
        Catalog(university=uni("HOST"), courses=[course("N9", ["alpha"])])
    )
    with pytest.raises(ModelMismatchError, match="another-model"):
        concept_services.engine.match("HOME", "NET", "HOST")


def test_explanations_come_from_llm_or_template(concept_services: Services) -> None:
    _setup(concept_services)

    def factory(schema: type[BaseModel]) -> BaseModel:
        return schema.model_validate({"summary": "Good fit.", "gaps": None})

    llm = FakeLLM(factory)
    concept_services.engine.explainer = Explainer(llm, "it")
    result = concept_services.engine.match("HOME", "DB", "HOST", top_k=1, explain=True)
    assert result.options[0].explanation is not None
    assert result.options[0].explanation.source == "llm"
    system, user, _ = llm.calls[0]
    assert "Italian" in system
    assert "COVERAGE: 100%" in user

    concept_services.engine.explainer = Explainer(FakeLLM(None))  # LLM fails -> template
    result = concept_services.engine.match("HOME", "NET", "HOST", top_k=1, explain=True)
    explanation = result.options[0].explanation
    assert explanation is not None and explanation.source == "template"
    assert "4 of 4 topics" in explanation.summary


def test_example_catalogs_regression(example_services: Services) -> None:
    """The offline embedder must keep ranking the hand-labelled pairs first."""
    report = evaluate(example_services.engine, GoldSet.load(GOLD))
    assert report.hit_at_1 >= 0.85, report.model_dump_json(indent=2)
    assert report.mrr >= 0.9
