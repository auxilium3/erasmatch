"""Embedding providers.

Every embedder returns L2-normalised float32 vectors, so cosine similarity is a dot product.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from erasmatch.config import Settings


class Embedder(Protocol):
    name: str
    dim: int
    default_topic_threshold: float

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim)."""
        ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hashing":
        from erasmatch.embeddings.hashing import HashingEmbedder

        return HashingEmbedder(dim=settings.embedding_dim)
    if settings.embedder == "ollama":
        from erasmatch.embeddings.ollama import OllamaEmbedder

        return OllamaEmbedder(
            base_url=settings.ollama_url,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            timeout=settings.request_timeout_s,
        )
    from erasmatch.embeddings.openai_compat import OpenAICompatEmbedder

    return OpenAICompatEmbedder(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
        timeout=settings.request_timeout_s,
    )
