from __future__ import annotations

from clinicalrag.rag import Citation
from clinicalrag.rerank import rerank, to_probability


def _cit(doc: str, score: float, text: str) -> Citation:
    return Citation(doc_id=doc, chunk_id=0, score=score, text=text)


def test_rerank_orders_by_query_overlap() -> None:
    cands = [
        _cit("a", 0.5, "garlic bread is a popular Italian dish"),
        _cit("b", 0.5, "aspirin reduces fever and inflammation"),
    ]
    out = rerank("how does aspirin work", cands)
    assert out[0].citation.doc_id == "b"


def test_rerank_returns_same_count_as_input() -> None:
    cands = [_cit("a", 0.5, "one"), _cit("b", 0.4, "two"), _cit("c", 0.3, "three")]
    out = rerank("query", cands)
    assert len(out) == 3


def test_empty_candidates_returns_empty() -> None:
    assert rerank("anything", []) == []


def test_empty_query_falls_back_to_retrieval_order() -> None:
    cands = [_cit("a", 0.9, "x"), _cit("b", 0.1, "y")]
    out = rerank("", cands)
    assert [r.citation.doc_id for r in out] == ["a", "b"]


def test_to_probability_is_in_unit_range() -> None:
    assert 0.0 <= to_probability(-1.0) <= 1.0
    assert 0.0 <= to_probability(0.5) <= 1.0
    assert 0.0 <= to_probability(2.0) <= 1.0
