"""Orchestrates the pipeline for one home course or a whole Learning Agreement.

1. Retrieval: the host university's courses closest to the home course (pgvector cosine).
2. Topic coverage: each home syllabus topic vs every topic of each candidate.
3. ECTS fit and combined score (deterministic, see scoring.py).
4. Bundles: greedy combination of candidates when no single course is enough.
5. Optional explanation by an LLM, which never changes the ranking.
"""

from __future__ import annotations

from erasmatch.domain import (
    CourseRef,
    LearningAgreementDraft,
    LearningAgreementRow,
    MatchOption,
    MatchResult,
    StoredCourse,
)
from erasmatch.errors import ModelMismatchError, NotFoundError
from erasmatch.llm.explainer import Explainer
from erasmatch.matching.agreement import assign_options
from erasmatch.matching.bundles import greedy_bundle
from erasmatch.matching.scoring import Candidate, ScoringConfig, build_option
from erasmatch.storage.base import Repository


class MatchEngine:
    def __init__(
        self, repo: Repository, embedder_name: str, cfg: ScoringConfig, explainer: Explainer
    ) -> None:
        self.repo = repo
        self.embedder_name = embedder_name
        self.cfg = cfg
        self.explainer = explainer

    # -- helpers ---------------------------------------------------------------------------

    def _course(self, university: str, code: str) -> StoredCourse:
        stored = self.repo.get_course(university, code)
        if stored is None:
            raise NotFoundError(f"Course {university}:{code} not found")
        self._check_model(stored)
        return stored

    def _check_model(self, stored: StoredCourse) -> None:
        if stored.embedding_model != self.embedder_name:
            raise ModelMismatchError(
                f"{stored.key} was embedded with '{stored.embedding_model}' but the configured "
                f"embedder is '{self.embedder_name}'. Re-import the catalogs with the new model."
            )

    def _require_university(self, code: str) -> None:
        if self.repo.get_university(code) is None:
            raise NotFoundError(f"University {code} not found")

    # -- public API ------------------------------------------------------------------------

    def options_for(self, home: StoredCourse, host_university: str) -> list[MatchOption]:
        """All scored options (singles + at most one bundle), best first."""
        hits = self.repo.search_similar(home.embedding, host_university, self.cfg.candidates)
        candidates = [Candidate(stored, sim) for stored, sim in hits]
        for cand in candidates:
            self._check_model(cand.stored)

        options = [build_option(home, [c], self.cfg) for c in candidates]
        bundle = greedy_bundle(home, candidates, self.cfg)
        if bundle is not None:
            bundle_option = build_option(home, bundle, self.cfg)
            best_single = max(options, key=lambda o: o.score)
            if (bundle_option.coverage, bundle_option.ects_ok) > (
                best_single.coverage,
                best_single.ects_ok,
            ):
                options.append(bundle_option)
        # A course that shares no topic with the home course cannot be an equivalent.
        options = [o for o in options if o.coverage > 0]
        options.sort(key=lambda o: (o.score, o.coverage), reverse=True)
        return options

    def match(
        self,
        home_university: str,
        home_course: str,
        host_university: str,
        top_k: int = 5,
        explain: bool = False,
    ) -> MatchResult:
        self._require_university(host_university)
        home = self._course(home_university, home_course)
        options = self.options_for(home, host_university)[:top_k]
        if explain:
            for option in options:
                option.explanation = self.explainer.explain(home, option)
        return MatchResult(
            home_course=CourseRef.of(home),
            host_university=host_university,
            topic_threshold=self.cfg.topic_threshold,
            options=options,
        )

    def learning_agreement(
        self,
        home_university: str,
        home_courses: list[str],
        host_university: str,
        explain: bool = False,
    ) -> LearningAgreementDraft:
        self._require_university(host_university)
        homes = [self._course(home_university, code) for code in dict.fromkeys(home_courses)]
        options_by_home = {h.code: self.options_for(h, host_university) for h in homes}
        assigned = assign_options(options_by_home, self.cfg.min_coverage)

        rows: list[LearningAgreementRow] = []
        for home in homes:
            option = assigned[home.code]
            note = None
            if option is None:
                note = (
                    "No host course reaches the minimum coverage "
                    f"({self.cfg.min_coverage:.0%}) without reusing a course already assigned."
                )
            else:
                if explain:
                    option.explanation = self.explainer.explain(home, option)
                if not option.ects_ok:
                    note = (
                        f"Credits below requirement ({option.ects_total:g}/{home.course.ects:g})."
                    )
            rows.append(
                LearningAgreementRow(home_course=CourseRef.of(home), option=option, note=note)
            )

        return LearningAgreementDraft(
            home_university=home_university,
            host_university=host_university,
            rows=rows,
            home_ects_total=sum(h.course.ects for h in homes),
            host_ects_total=round(sum(r.option.ects_total for r in rows if r.option), 2),
            unmatched=[r.home_course.code for r in rows if r.option is None],
        )
