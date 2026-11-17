"""Runtime configuration, read from environment variables prefixed with ``ERASMATCH_``."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ERASMATCH_", env_file=".env", extra="ignore")

    # Storage: "memory://" keeps everything in-process (handy for tests and one-off CLI runs),
    # a postgresql+psycopg:// URL uses PostgreSQL with pgvector.
    database_url: str = "memory://"

    # Embeddings
    embedder: Literal["ollama", "openai", "hashing"] = "ollama"
    embedding_model: str = "bge-m3"
    embedding_dim: int = Field(1024, description="Must match the model and the DB schema.")

    # LLM used to explain matches and to extract courses from PDFs ("none" disables it)
    llm: Literal["ollama", "openai", "none"] = "ollama"
    llm_model: str = "qwen2.5:7b-instruct"
    explanation_language: str = "en"

    # Provider endpoints
    ollama_url: str = "http://localhost:11434"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    request_timeout_s: float = 120.0

    # Matching knobs. topic_threshold=None means "use the embedder's own default".
    topic_threshold: float | None = None
    candidates: int = 20
    min_ects_ratio: float = 1.0
    target_coverage: float = 0.75
    min_coverage: float = 0.5
    max_bundle_size: int = 3
    bundle_penalty: float = 0.03
    weight_coverage: float = 0.6
    weight_similarity: float = 0.25
    weight_ects: float = 0.15


@lru_cache
def get_settings() -> Settings:
    return Settings()
