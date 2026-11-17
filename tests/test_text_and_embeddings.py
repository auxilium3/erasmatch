from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from erasmatch.domain import CourseIn
from erasmatch.embeddings.hashing import HashingEmbedder
from erasmatch.embeddings.ollama import OllamaEmbedder
from erasmatch.embeddings.openai_compat import OpenAICompatEmbedder
from erasmatch.errors import ProviderError
from erasmatch.text import course_document, effective_topics, fold, split_topics


def test_fold_removes_accents_and_case() -> None:
    assert fold("Programación ÀÉÌ") == "programacion aei"


def test_split_topics_handles_bullets_and_sentences() -> None:
    text = "- Modello relazionale\n- Algebra relazionale.\n1) Linguaggio SQL; Indici B-tree."
    assert split_topics(text) == [
        "Modello relazionale",
        "Algebra relazionale",
        "Linguaggio SQL",
        "Indici B-tree",
    ]


def test_effective_topics_fallbacks() -> None:
    declared = CourseIn(code="A", title="A course", ects=6, topics=["  - Uno due ", ""])
    assert effective_topics(declared) == ["Uno due"]
    from_description = CourseIn(code="B", title="Bx", ects=6, description="Limiti e continuità.")
    assert effective_topics(from_description) == ["Limiti e continuità"]
    title_only = CourseIn(code="C", title="Analisi", ects=6)
    assert effective_topics(title_only) == ["Analisi"]


def test_course_document_includes_topics() -> None:
    c = CourseIn(code="A", title="Reti", ects=6, description="Intro.", topics=["TCP", "UDP"])
    assert course_document(c) == "Reti\nIntro.\nTopics: TCP; UDP"


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    emb = HashingEmbedder(dim=256)
    a = emb.embed(["Programmazione con i socket", "Algebra lineare"])
    b = emb.embed(["Programmazione con i socket", "Algebra lineare"])
    assert a.shape == (2, 256)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)
    assert emb.embed([]).shape == (0, 256)


def test_hashing_embedder_sees_cognates() -> None:
    emb = HashingEmbedder()
    it, es, other = emb.embed(
        ["Programmazione con i socket", "Programación con sockets", "Puentes de medida"]
    )
    assert it @ es > it @ other + 0.2


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")


def test_ollama_embedder_batches_and_normalises() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        assert request.url.path == "/api/embed"
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0] for _ in body["input"]]})

    emb = OllamaEmbedder("http://test", "bge-m3", dim=2, batch_size=2, client=_mock_client(handler))
    out = emb.embed(["a", "b", "c"])
    assert [len(b["input"]) for b in seen] == [2, 1]
    np.testing.assert_allclose(out[0], [0.6, 0.8])
    assert emb.name == "ollama:bge-m3"
    assert emb.default_topic_threshold == 0.6


def test_ollama_embedder_reports_dimension_mismatch() -> None:
    client = _mock_client(lambda r: httpx.Response(200, json={"embeddings": [[1.0, 0.0, 0.0]]}))
    emb = OllamaEmbedder("http://test", "bge-m3", dim=2, client=client)
    with pytest.raises(ProviderError, match="3-dim"):
        emb.embed(["x"])


def test_ollama_embedder_wraps_http_errors() -> None:
    emb = OllamaEmbedder(
        "http://test", "bge-m3", dim=2, client=_mock_client(lambda r: httpx.Response(500))
    )
    with pytest.raises(ProviderError, match="ollama pull bge-m3"):
        emb.embed(["x"])


def test_openai_embedder_orders_by_index() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        data = [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]
        return httpx.Response(200, json={"data": data})

    emb = OpenAICompatEmbedder("http://test", "k", "m", dim=2, client=_mock_client(handler))
    np.testing.assert_allclose(emb.embed(["first", "second"]), [[1, 0], [0, 1]])
