"""Embeddings from any OpenAI-compatible endpoint (OpenAI, vLLM, LM Studio, LocalAI...)."""

from __future__ import annotations

from collections.abc import Sequence

import httpx
import numpy as np

from erasmatch.embeddings import l2_normalize
from erasmatch.errors import ProviderError


class OpenAICompatEmbedder:
    default_topic_threshold = 0.45

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        dim: int,
        timeout: float = 120.0,
        batch_size: int = 64,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.dim = dim
        self.name = f"openai:{model}"
        self.batch_size = batch_size
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            payload: dict = {"model": self.model, "input": batch}
            if self.model.startswith("text-embedding-3"):
                payload["dimensions"] = self.dim
            try:
                resp = self._client.post("/embeddings", json=payload)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise ProviderError(f"Embedding request failed: {exc}") from exc
            data = sorted(resp.json()["data"], key=lambda item: item["index"])
            rows.extend(item["embedding"] for item in data)
        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.shape[1] != self.dim:
            raise ProviderError(
                f"Model '{self.model}' returned {matrix.shape[1]}-dim vectors but "
                f"ERASMATCH_EMBEDDING_DIM={self.dim}."
            )
        return l2_normalize(matrix)
