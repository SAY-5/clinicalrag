"""Document chunker. Produces overlapping windows of `chunk_size`
words with `overlap` words shared between consecutive chunks.
Overlap helps retrieval find sentences that span chunk boundaries —
biomedical abstracts often have a single key claim split across
two paragraphs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def chunk_document(doc_id: str, text: str, chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")
    words = text.split()
    if not words:
        return []
    out: list[Chunk] = []
    step = chunk_size - overlap
    for i, start in enumerate(range(0, len(words), step)):
        end = min(start + chunk_size, len(words))
        out.append(Chunk(doc_id=doc_id, chunk_id=i, text=" ".join(words[start:end])))
        if end == len(words):
            break
    return out
