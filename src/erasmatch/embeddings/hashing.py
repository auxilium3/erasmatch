"""Deterministic, offline embedder based on feature hashing.

It is NOT semantic: it only sees shared words and character n-grams. That is still enough to
catch cognates across Romance languages ("protocolli" / "protocolos") and technical terms
("TCP/IP", "SQL"), which makes it useful for tests, CI and trying the tool without a model.
For real use, configure a multilingual model such as bge-m3.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

from erasmatch.embeddings import l2_normalize
from erasmatch.text import fold

_WORD = re.compile(r"[a-z0-9+#]+")

# Function words in IT/ES/EN that would otherwise dominate short texts.
_STOPWORDS = frozenset(
    """
    a al alla alle allo agli ai con da dal dalla dei degli del della delle di e ed gli i il in
    la le lo nei nel nella per su sul sulla tra fra un una uno o
    de del el en la las los lo para por con y e o u un una unos unas al como sus su
    the of and or to in on for with a an by from into its their
    """.split()
)


class HashingEmbedder:
    name = "hashing-v1"
    default_topic_threshold = 0.3

    def __init__(self, dim: int = 1024, ngram_min: int = 3, ngram_max: int = 5) -> None:
        self.dim = dim
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max

    def _features(self, text: str) -> Counter[str]:
        feats: Counter[str] = Counter()
        for word in _WORD.findall(fold(text)):
            if word in _STOPWORDS:
                continue
            feats[f"w:{word}"] += 2
            padded = f" {word} "
            for n in range(self.ngram_min, self.ngram_max + 1):
                for i in range(len(padded) - n + 1):
                    feats[f"c:{padded[i : i + n]}"] += 1
        return feats

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat, count in self._features(text).items():
            digest = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
            sign = 1.0 if digest >> 63 else -1.0
            vec[digest % self.dim] += sign * (1.0 + math.log(count))
        return vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in texts]))
