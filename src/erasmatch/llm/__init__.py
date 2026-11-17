"""LLM clients that return validated Pydantic objects (structured output)."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from erasmatch.config import Settings

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    name: str

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        """Ask for JSON matching ``schema`` and return it validated. Raises ProviderError."""
        ...


def build_llm(settings: Settings) -> LLMClient | None:
    if settings.llm == "none":
        return None
    from erasmatch.llm.providers import OllamaLLM, OpenAICompatLLM

    if settings.llm == "ollama":
        return OllamaLLM(settings.ollama_url, settings.llm_model, settings.request_timeout_s)
    return OpenAICompatLLM(
        settings.openai_base_url,
        settings.openai_api_key,
        settings.llm_model,
        settings.request_timeout_s,
    )
