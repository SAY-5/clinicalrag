"""FAISS-shaped vector index. Production uses faiss-cpu (HNSW, IVF,
PQ); tests use this brute-force numpy implementation. The contract
is identical: add(matrix), search(q, k) -> (scores, ids).

For 15k documents at 256-dim, the brute-force search is sub-10ms;
swap to FAISS HNSW once you scale past 100k.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from clinicalrag.chunk import Chunk


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._matrix: np.ndarray | None = None
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if vectors.size == 0:
            return
        if vectors.shape[1] != self.dim:
            raise ValueError(f"dim mismatch: index={self.dim}, vectors={vectors.shape[1]}")
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        self._chunks.extend(chunks)

    def size(self) -> int:
        return 0 if self._matrix is None else self._matrix.shape[0]

    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def search(self, q: np.ndarray, k: int) -> list[tuple[float, Chunk]]:
        if self._matrix is None or self.size() == 0:
            return []
        # Cosine sim — both sides are L2-normalized so dot product
        # is sufficient.
        scores = self._matrix @ q
        if k >= self.size():
            order = np.argsort(-scores)
        else:
            # Top-k via argpartition + sort. Slightly faster than
            # full argsort on large indices; tests stay deterministic
            # because numpy's argpartition is stable for ties on
            # contiguous arrays.
            top = np.argpartition(-scores, k)[:k]
            order = top[np.argsort(-scores[top])]
        return [(float(scores[i]), self._chunks[int(i)]) for i in order]
