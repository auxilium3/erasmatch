from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from erasmatch.errors import ProviderError
from erasmatch.llm.extractor import extract_course
from erasmatch.llm.providers import OllamaLLM, OpenAICompatLLM
from tests.conftest import FakeLLM


class Answer(BaseModel):
    summary: str


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")


def test_ollama_sends_json_schema_and_parses_reply() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["format"]["properties"]["summary"]["type"] == "string"
        assert body["options"]["temperature"] == 0
        assert body["messages"][0]["role"] == "system"
        return httpx.Response(200, json={"message": {"content": '{"summary": "ok"}'}})

    llm = OllamaLLM("http://test", "qwen2.5:7b-instruct", 10, client=_client(handler))
    assert llm.complete_json("sys", "user", Answer) == Answer(summary="ok")


def test_openai_compat_uses_response_format() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/chat/completions"
        assert body["response_format"]["json_schema"]["name"] == "Answer"
        content = '{"summary": "fine"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    llm = OpenAICompatLLM("http://test", "key", "gpt-x", 10, client=_client(handler))
    assert llm.complete_json("sys", "user", Answer).summary == "fine"


def test_invalid_json_and_http_errors_become_provider_errors() -> None:
    bad = _client(lambda r: httpx.Response(200, json={"message": {"content": '{"nope": 1}'}}))
    with pytest.raises(ProviderError, match="does not match the schema"):
        OllamaLLM("http://test", "m", 10, client=bad).complete_json("s", "u", Answer)
    down = _client(lambda r: httpx.Response(503))
    with pytest.raises(ProviderError):
        OpenAICompatLLM("http://test", None, "m", 10, client=down).complete_json("s", "u", Answer)


def test_extraction_flags_missing_fields() -> None:
    def factory(schema: type[BaseModel]) -> BaseModel:
        return schema.model_validate({"title": "Cálculo", "ects": 0, "topics": []})

    result = extract_course("x" * 20_000, FakeLLM(factory))
    assert result.course.code == "TODO"
    assert result.course.ects == 6.0
    assert len(result.warnings) == 4  # truncated, no code, bad credits, no topics
