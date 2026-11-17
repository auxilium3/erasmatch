"""Embeddings from a local Ollama server (default model: bge-m3, multilingual, 1024 dims)."""

from __future__ import annotations

from collections.abc import Sequence

import httpx
import numpy as np

from erasmatch.embeddings import l2_normalize
from erasmatch.errors import ProviderError

# Starting points; calibrate with `erasmatch eval` on your own labelled pairs.
_DEFAULT_THRESHOLDS = {"bge-m3": 0.6, "nomic-embed-text": 0.55, "mxbai-embed-large": 0.6}


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        dim: int,
        timeout: float = 120.0,
        batch_size: int = 32,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.dim = dim
        self.name = f"ollama:{model}"
        self.default_topic_threshold = _DEFAULT_THRESHOLDS.get(model.split(":")[0], 0.55)
        self.batch_size = batch_size
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            try:
                resp = self._client.post("/api/embed", json={"model": self.model, "input": batch})
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"Ollama embedding request failed ({exc}). Is Ollama running and is "
                    f"'{self.model}' pulled? Try: ollama pull {self.model}"
                ) from exc
            rows.extend(resp.json()["embeddings"])
        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.shape[1] != self.dim:
            raise ProviderError(
                f"Model '{self.model}' returned {matrix.shape[1]}-dim vectors but "
                f"ERASMATCH_EMBEDDING_DIM={self.dim}. Align the setting and the DB schema."
            )
        return l2_normalize(matrix)
