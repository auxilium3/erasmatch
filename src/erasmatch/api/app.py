"""FastAPI application. Interactive docs at /docs once the server is running."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from erasmatch import __version__
from erasmatch.config import get_settings
from erasmatch.domain import (
    Catalog,
    CourseIn,
    LearningAgreementDraft,
    MatchResult,
    StoredCourse,
    UniversityIn,
)
from erasmatch.errors import LLMDisabledError, ModelMismatchError, NotFoundError, ProviderError
from erasmatch.ingest import file_to_text
from erasmatch.llm.extractor import ExtractionResult, extract_course
from erasmatch.service import Services, build_services

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class HealthResponse(BaseModel):
    status: str
    version: str
    storage: str
    embedder: str
    llm: str | None


class ImportResponse(BaseModel):
    university: str
    courses: int
    embedding_model: str


class CourseOut(CourseIn):
    university: str
    topics_used: list[str] = Field(description="Topics actually used for matching")

    @classmethod
    def of(cls, stored: StoredCourse) -> CourseOut:
        return cls(
            **stored.course.model_dump(),
            university=stored.university_code,
            topics_used=stored.topics,
        )


class MatchRequest(BaseModel):
    home_university: str
    home_course: str
    host_university: str
    top_k: int = Field(5, ge=1, le=20)
    explain: bool = False


class AgreementRequest(BaseModel):
    home_university: str
    home_courses: list[str] = Field(min_length=1, max_length=30)
    host_university: str
    explain: bool = False


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = services or build_services(get_settings())
        yield

    app = FastAPI(
        title="Erasmatch",
        version=__version__,
        description="Find equivalent courses between universities for Erasmus Learning Agreements.",
        lifespan=lifespan,
    )

    def svc(request: Request) -> Services:
        return request.app.state.services

    # -- error mapping -----------------------------------------------------------------------

    @app.exception_handler(NotFoundError)
    async def _not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ModelMismatchError)
    async def _mismatch(_: Request, exc: ModelMismatchError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ProviderError)
    async def _provider(_: Request, exc: ProviderError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(LLMDisabledError)
    async def _llm_off(_: Request, exc: LLMDisabledError) -> JSONResponse:
        return JSONResponse(status_code=501, content={"detail": str(exc)})

    # -- routes ------------------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(request: Request) -> HealthResponse:
        s = svc(request)
        return HealthResponse(
            status="ok",
            version=__version__,
            storage=s.repo.kind,
            embedder=s.embedder.name,
            llm=s.llm.name if s.llm else None,
        )

    @app.post(
        "/catalogs",
        response_model=ImportResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["catalog"],
    )
    def import_catalog(catalog: Catalog, request: Request) -> ImportResponse:
        """Create or update a university and its courses (embeddings are computed here)."""
        s = svc(request)
        count = s.catalog.import_catalog(catalog)
        return ImportResponse(
            university=catalog.university.code, courses=count, embedding_model=s.embedder.name
        )

    @app.get("/universities", response_model=list[UniversityIn], tags=["catalog"])
    def list_universities(request: Request) -> list[UniversityIn]:
        return svc(request).repo.list_universities()

    @app.get("/universities/{code}/courses", response_model=list[CourseOut], tags=["catalog"])
    def list_courses(code: str, request: Request) -> list[CourseOut]:
        repo = svc(request).repo
        if repo.get_university(code) is None:
            raise NotFoundError(f"University {code} not found")
        return [CourseOut.of(c) for c in repo.list_courses(code)]

    @app.get(
        "/universities/{code}/courses/{course_code}", response_model=CourseOut, tags=["catalog"]
    )
    def get_course(code: str, course_code: str, request: Request) -> CourseOut:
        stored = svc(request).repo.get_course(code, course_code)
        if stored is None:
            raise NotFoundError(f"Course {code}:{course_code} not found")
        return CourseOut.of(stored)

    @app.post("/match", response_model=MatchResult, tags=["matching"])
    def match(body: MatchRequest, request: Request) -> MatchResult:
        """Rank host courses (and bundles) for one home course."""
        return svc(request).engine.match(
            body.home_university,
            body.home_course,
            body.host_university,
            top_k=body.top_k,
            explain=body.explain,
        )

    @app.post("/learning-agreement", response_model=LearningAgreementDraft, tags=["matching"])
    def learning_agreement(body: AgreementRequest, request: Request) -> LearningAgreementDraft:
        """Draft a full Learning Agreement; each host course is used at most once."""
        return svc(request).engine.learning_agreement(
            body.home_university, body.home_courses, body.host_university, explain=body.explain
        )

    @app.post("/extract", response_model=ExtractionResult, tags=["catalog"])
    def extract(request: Request, file: UploadFile = File(...)) -> ExtractionResult:
        """Turn a syllabus (PDF or text) into a course to review before importing."""
        s = svc(request)
        if s.llm is None:
            raise LLMDisabledError("Extraction needs an LLM; set ERASMATCH_LLM.")
        # Sync endpoint on purpose: runs in the threadpool, so the blocking LLM call
        # doesn't stall the event loop.
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File larger than 10 MB")
        text = file_to_text(file.filename or "", data)
        if not text.strip():
            raise HTTPException(
                status_code=422, detail="No text found (scanned PDFs need OCR first)."
            )
        return extract_course(text, s.llm)

    return app
