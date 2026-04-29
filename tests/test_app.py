"""FastAPI tests via TestClient."""

from __future__ import annotations

from fastapi.testclient import TestClient

from clinicalrag import HashEmbedder, Pipeline, create_app


def _client() -> TestClient:
    p = Pipeline(embedder=HashEmbedder(dim=128))
    p.ingest("d1", "aspirin reduces fever and inflammation in many patients")
    return TestClient(create_app(p))


def test_ingest_endpoint_returns_chunk_count() -> None:
    c = _client()
    r = c.post("/ingest", json={"doc_id": "d2", "text": "garlic bread is delicious"})
    assert r.status_code == 200
    body = r.json()
    assert body["chunks"] >= 1


def test_query_returns_grounded_answer() -> None:
    c = _client()
    r = c.post("/query", json={"question": "what does aspirin do?"})
    assert r.status_code == 200
    body = r.json()
    assert "citations" in body
    assert "flagged" in body
    assert "grounding" in body


def test_stream_endpoint_emits_sse_frames() -> None:
    c = _client()
    with c.stream("GET", "/query/stream?question=aspirin&top_k=1") as resp:
        body = b"".join(resp.iter_bytes()).decode()
    assert "event: citations\n" in body
    assert "event: token\n" in body
    assert "event: done\n" in body
