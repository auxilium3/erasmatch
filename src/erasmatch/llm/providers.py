"""Ollama and OpenAI-compatible chat clients with JSON-schema constrained output."""

from __future__ import annotations

import httpx
from pydantic import ValidationError

from erasmatch.errors import ProviderError
from erasmatch.llm import T


def _validate(raw: str, schema: type[T], provider: str) -> T:
    try:
        return schema.model_validate_json(raw)
    except ValidationError as exc:
        raise ProviderError(
            f"{provider} returned JSON that does not match the schema: {exc}"
        ) from exc


class OllamaLLM:
    def __init__(
        self, base_url: str, model: str, timeout: float, client: httpx.Client | None = None
    ) -> None:
        self.model = model
        self.name = f"ollama:{model}"
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema.model_json_schema(),  # Ollama >= 0.5 structured outputs
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            resp = self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Ollama chat request failed ({exc}). Try: ollama pull {self.model}"
            ) from exc
        return _validate(resp.json()["message"]["content"], schema, self.name)


class OpenAICompatLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: float,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.name = f"openai:{model}"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout
        )

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            resp = self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Chat completion request failed: {exc}") from exc
        content = resp.json()["choices"][0]["message"]["content"]
        return _validate(content, schema, self.name)
