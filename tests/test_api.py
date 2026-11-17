from __future__ import annotations

from collections.abc import Iterator

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import BaseModel

from erasmatch.api import create_app
from erasmatch.config import Settings
from erasmatch.embeddings.hashing import HashingEmbedder
from erasmatch.service import build_services
from erasmatch.storage.memory import InMemoryRepository
from tests.conftest import HOME_CATALOG, HOST_CATALOG, FakeLLM


def _extraction(schema: type[BaseModel]) -> BaseModel:
    return schema.model_validate(
        {
            "code": "ING-INF/05 RETI",
            "title": "Reti di Calcolatori",
            "ects": 9,
            "language": "IT",
            "description": "Protocolli di Internet.",
            "topics": ["TCP e UDP", " ", "Routing IP"],
        }
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    services = build_services(
        settings, repo=InMemoryRepository(), embedder=HashingEmbedder(), llm=FakeLLM(_extraction)
    )
    with TestClient(create_app(services)) as c:
        for path in (HOME_CATALOG, HOST_CATALOG):
            resp = c.post("/catalogs", json=yaml.safe_load(path.read_text()))
            assert resp.status_code == 201, resp.text
        yield c


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["storage"] == "memory"
    assert body["embedder"] == "hashing-v1"


def test_catalog_endpoints(client: TestClient) -> None:
    codes = [u["code"] for u in client.get("/universities").json()]
    assert codes == ["UEJ", "UNIEX"]
    courses = client.get("/universities/UEJ/courses").json()
    assert len(courses) == 15
    cal = client.get("/universities/UEJ/courses/CAL").json()
    assert cal["language"] == "es"  # inherited from the university
    assert len(cal["topics_used"]) == 4  # derived from the description
    assert client.get("/universities/NOPE/courses").status_code == 404
    assert client.get("/universities/UEJ/courses/NOPE").status_code == 404


def test_invalid_catalog_is_rejected(client: TestClient) -> None:
    bad = {"university": {"code": "X", "name": "X", "country": "ITA", "language": "it"}}
    assert client.post("/catalogs", json=bad).status_code == 422


def test_match(client: TestClient) -> None:
    resp = client.post(
        "/match",
        json={"home_university": "UNIEX", "home_course": "RETI", "host_university": "UEJ"},
    )
    assert resp.status_code == 200
    body = resp.json()
    best = body["options"][0]
    assert sorted(c["code"] for c in best["host_courses"]) == ["RC", "TDR"]
    assert best["ects_total"] == 12
    assert len(best["alignments"]) == 9


def test_match_not_found(client: TestClient) -> None:
    resp = client.post(
        "/match", json={"home_university": "UNIEX", "home_course": "X", "host_university": "UEJ"}
    )
    assert resp.status_code == 404
    assert "UNIEX:X" in resp.json()["detail"]


def test_learning_agreement(client: TestClient) -> None:
    resp = client.post(
        "/learning-agreement",
        json={
            "home_university": "UNIEX",
            "home_courses": ["INGSW", "WEB", "MULTIMEDIA"],
            "host_university": "UEJ",
        },
    )
    assert resp.status_code == 200
    draft = resp.json()
    assigned = [r["option"]["host_courses"][0]["code"] for r in draft["rows"]]
    assert assigned == ["ISW", "DAW", "TM"]
    assert draft["home_ects_total"] == draft["host_ects_total"] == 18


def test_extract_returns_reviewable_course(client: TestClient) -> None:
    files = {"file": ("reti.txt", b"Scheda insegnamento: Reti di Calcolatori, 9 CFU", "text/plain")}
    resp = client.post("/extract", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["course"]["code"] == "ING-INF-05-RETI"
    assert body["course"]["topics"] == ["TCP e UDP", "Routing IP"]
    assert body["course"]["language"] == "it"
    assert body["warnings"] == []


def test_extract_needs_llm(settings: Settings) -> None:
    services = build_services(
        settings, repo=InMemoryRepository(), embedder=HashingEmbedder(), use_llm=False
    )
    with TestClient(create_app(services)) as c:
        resp = c.post("/extract", files={"file": ("a.txt", b"text", "text/plain")})
    assert resp.status_code == 501


def test_duplicate_course_codes_are_rejected(client: TestClient) -> None:
    course = {"code": "A1", "title": "Algebra", "ects": 6}
    payload = {
        "university": {"code": "DUP", "name": "Dup U", "country": "ES", "language": "es"},
        "courses": [course, course],
    }
    resp = client.post("/catalogs", json=payload)
    assert resp.status_code == 422
    assert "Duplicate course codes: A1" in resp.text
