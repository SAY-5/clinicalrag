"""v4: cross-encoder-style reranker on top of vector retrieval.

Vector search returns top-K by embedding similarity. That's a fast
recall stage but it ranks by semantic similarity to the QUERY,
not by how well each chunk *answers* the query. v4 adds a
two-stage retrieve-then-rerank pipeline:

1. Retrieve top-K (where K is larger than what we'll show, e.g.
   K=20 to show 5).
2. Rerank by a relevance score that mixes:
   - lexical overlap (BM25-style: query terms in the chunk)
   - position bias (earlier chunks are slightly preferred — first
     paragraph of a paper is usually the abstract)
   - length bonus (longer chunks have more context)

Production replaces this with a real cross-encoder (e.g., MS MARCO
MiniLM). The contract — `rerank(query, candidates) → ranked` —
stays the same.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from clinicalrag.rag import Citation


_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN.finditer(text) if len(m.group(0)) > 2}


@dataclass(frozen=True, slots=True)
class RerankedCitation:
    citation: Citation
    rerank_score: float
    rationale: str


def rerank(query: str, candidates: list[Citation]) -> list[RerankedCitation]:
    if not candidates:
        return []
    q_tokens = _tokens(query)
    if not q_tokens:
        # No tokens to overlap on — fall back to retrieval order.
        return [
            RerankedCitation(citation=c, rerank_score=c.score, rationale="empty query")
            for c in candidates
        ]
    out: list[RerankedCitation] = []
    n = len(candidates)
    for i, c in enumerate(candidates):
        c_tokens = _tokens(c.text)
        overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
        position_bonus = 1.0 - (i / n) * 0.1   # up to 10% boost for early
        length_bonus = min(1.0, len(c.text) / 500.0)  # saturates at 500 chars
        score = (
            0.6 * overlap
            + 0.3 * position_bonus
            + 0.1 * length_bonus
            + 0.0  # leave room for cross-encoder slot
        )
        out.append(
            RerankedCitation(
                citation=c,
                rerank_score=score,
                rationale=(
                    f"overlap={overlap:.2f} pos={position_bonus:.2f} "
                    f"len={length_bonus:.2f}"
                ),
            )
        )
    out.sort(key=lambda r: r.rerank_score, reverse=True)
    return out


# Convenience wrapper that softens the score with a logistic so
# downstream callers can interpret it as a probability.
def to_probability(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-4.0 * (score - 0.5)))
