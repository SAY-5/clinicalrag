# clinicalrag

LLM-powered Retrieval-Augmented Generation pipeline for biomedical
literature. Pluggable embedder + FAISS-shaped vector index + FastAPI
REST endpoint with citation-grounded answers and a hallucination
guard that scores claims against the retrieved evidence before
returning them.

```
docs ──ingest──▶ chunks ──embed──▶ vector index
                                       │
query ──embed──┐                       │
              ▼                        │
         search top-k ─────────────────┘
              │
              ▼
       LLM answer + citations ──guard──▶ response
              │                              │
              └─SSE stream of tokens─────────┘
```

## Versions

| Version | Capability | Status |
|---|---|---|
| v1 | Ingest + chunker + pluggable embedder (HashEmbedder for tests) + numpy-backed top-k vector search + FastAPI `/query` with closed-schema tool calling | shipped |
| v2 | SSE streaming of answer tokens + retrieved citations interleaved | shipped |
| v3 | Hallucination guard scores each answer claim's overlap with retrieved evidence; below-threshold answers are flagged or refused | shipped |

## Quickstart

```bash
pip install -e ".[dev]"
pytest                          # 12 tests
clinicalrag serve               # http://127.0.0.1:8000/docs
```

## Why pluggable embedder + FAISS-shape

The vector index in `vector.py` exposes the same surface as FAISS
(`add(vectors)`, `search(query, k)`, `size()`). For tests we use a
deterministic `HashEmbedder` and a flat brute-force search; for
production swap in OpenAI embeddings + a real FAISS index — same
contract.

## Tests

12 tests across chunker, embedder, vector index, RAG pipeline,
hallucination guard, and FastAPI endpoint. See `ARCHITECTURE.md`
for the citation-grounding scoring details.
