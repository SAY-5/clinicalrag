"""Chunker tests."""

from __future__ import annotations

import pytest

from clinicalrag import chunk_document


def test_short_doc_yields_one_chunk() -> None:
    cs = chunk_document("d1", "one two three", chunk_size=200)
    assert len(cs) == 1
    assert cs[0].chunk_id == 0
    assert cs[0].doc_id == "d1"


def test_long_doc_splits_with_overlap() -> None:
    text = " ".join(str(i) for i in range(500))
    cs = chunk_document("d1", text, chunk_size=200, overlap=40)
    # Step is 160, so chunks at offsets 0, 160, 320 → 3 windows.
    assert len(cs) >= 3
    # Adjacent chunks share `overlap` words.
    first_words = cs[0].text.split()[-40:]
    second_words = cs[1].text.split()[:40]
    assert first_words == second_words


def test_empty_doc_yields_zero_chunks() -> None:
    assert chunk_document("d1", "") == []


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_document("d1", "x", chunk_size=10, overlap=10)
