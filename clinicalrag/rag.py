"""RAG pipeline: ingest → embed → index → query → answer with
citations.

The LLM call is abstracted via a `Generator` protocol. The default
`StubGenerator` produces a deterministic answer that quotes back
the top retrieved chunk + its citation, so the pipeline path is
fully testable without an API key. Production wires in OpenAI's
chat completions endpoint.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, Field

from clinicalrag.chunk import chunk_document
from clinicalrag.embed import Embedder, HashEmbedder
from clinicalrag.guard import score_grounding
from clinicalrag.vector import VectorIndex


class Query(BaseModel):
    model_config = {"extra": "forbid"}
    question: str = Field(min_length=3)
    top_k: int = 5


@dataclass(frozen=True, slots=True)
class Citation:
    doc_id: str
    chunk_id: int
    score: float
    text: str


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    grounding: float
    flagged: bool  # set by the v3 hallucination guard
    flag_reason: str = ""


class Generator(Protocol):
    def generate(self, question: str, citations: list[Citation]) -> str: ...
    def stream(self, question: str, citations: list[Citation]) -> Iterator[str]: ...


class StubGenerator:
    """Quotes the top citation. The output is deterministic which
    lets the test suite assert exact strings."""

    def generate(self, question: str, citations: list[Citation]) -> str:
        if not citations:
            return f"no evidence found for: {question}"
        top = citations[0]
        return f"based on the literature, {top.text[:200]} (see [{top.doc_id}#{top.chunk_id}])."

    def stream(self, question: str, citations: list[Citation]) -> Iterator[str]:
        for word in self.generate(question, citations).split():
            yield word + " "


@dataclass
class Pipeline:
    embedder: Embedder = field(default_factory=HashEmbedder)
    generator: Generator = field(default_factory=StubGenerator)
    grounding_threshold: float = 0.3
    _index: VectorIndex | None = None

    def __post_init__(self) -> None:
        self._index = VectorIndex(dim=self.embedder.dim)

    def ingest(self, doc_id: str, text: str, chunk_size: int = 200, overlap: int = 40) -> int:
        chunks = chunk_document(doc_id, text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return 0
        vectors = self.embedder.embed_batch([c.text for c in chunks])
        assert self._index is not None
        self._index.add(chunks, vectors)
        return len(chunks)

    def index_size(self) -> int:
        assert self._index is not None
        return self._index.size()

    def query(self, q: Query) -> Answer:
        assert self._index is not None
        qv = self.embedder.embed(q.question)
        hits = self._index.search(qv, q.top_k)
        cits = tuple(
            Citation(doc_id=c.doc_id, chunk_id=c.chunk_id, score=s, text=c.text)
            for s, c in hits
        )
        text = self.generator.generate(q.question, list(cits))
        gs = score_grounding(text, [c.text for c in cits])
        flagged = gs.score < self.grounding_threshold
        reason = (
            f"grounding={gs.score:.2f} below threshold {self.grounding_threshold:.2f}"
            if flagged
            else ""
        )
        return Answer(
            text=text, citations=cits, grounding=gs.score,
            flagged=flagged, flag_reason=reason,
        )
