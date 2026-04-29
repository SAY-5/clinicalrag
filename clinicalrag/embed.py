"""Embedder protocol + a deterministic stub. Production swaps in
`openai.embeddings.create` (text-embedding-3-large, 3072 dim).

The contract:
    embed(text: str) -> np.ndarray  # shape (dim,), L2-normalized
    embed_batch(texts: list[str]) -> np.ndarray  # shape (n, dim)
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic hash-bucket embedder. Each token contributes 1
    to the bucket given by `blake2b(token) % dim`; output is L2-
    normalized so cosine similarity == dot product.

    Not semantically meaningful — used for hermetic tests that
    validate the pipeline plumbing without an API key."""

    def __init__(self, dim: int = 256):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        for tok in text.lower().split():
            h = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).hexdigest(), 16)
            v[h % self._dim] += 1.0
        n = float(np.linalg.norm(v))
        if n > 0:
            v /= n
        return v

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([self.embed(t) for t in texts])
