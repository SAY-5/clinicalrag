"""End-to-end pipeline tests + hallucination guard tests."""

from __future__ import annotations

from clinicalrag import HashEmbedder, Pipeline, Query, score_grounding


def test_ingest_and_query_returns_relevant_citation() -> None:
    p = Pipeline(embedder=HashEmbedder(dim=128))
    p.ingest("doc1", "Aspirin reduces fever and inflammation. It works as an NSAID.")
    p.ingest("doc2", "Garlic bread is a popular Italian-American side dish.")
    a = p.query(Query(question="how does aspirin work?", top_k=2))
    # Top citation should come from doc1 (aspirin doc).
    assert a.citations[0].doc_id == "doc1"


def test_empty_index_returns_empty_citations_and_flagged_answer() -> None:
    p = Pipeline()
    a = p.query(Query(question="anything"))
    assert a.citations == ()
    # No evidence → grounding 0 → flagged.
    assert a.flagged
    assert a.grounding == 0.0


def test_grounding_score_high_when_answer_overlaps_evidence() -> None:
    s = score_grounding(
        "aspirin reduces fever and inflammation",
        ["aspirin works by reducing fever and inflammation in many patients"],
    )
    assert s.score >= 0.6


def test_grounding_score_low_when_answer_invents_facts() -> None:
    s = score_grounding(
        "spaceships traverse galactic void using warp engines",
        ["aspirin works by reducing fever and inflammation"],
    )
    assert s.score <= 0.1


def test_query_validation_rejects_empty_question() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Query(question="")
