# Architecture

## Pipeline shape

```
ingest:  doc → chunks → embed_batch → VectorIndex.add
query:   q → embed → VectorIndex.search(k) → Citations
                                                │
                                                ▼
                                       Generator(question, citations)
                                                │
                                                ▼
                                          score_grounding
                                                │
                                                ▼
                                       Answer{text, citations,
                                              grounding, flagged}
```

## Pluggable embedder

`Embedder` is a Protocol: `embed(text)` and `embed_batch(texts)`
return L2-normalized numpy arrays. Tests use `HashEmbedder` (each
token contributes 1 to the bucket given by `blake2b(token) % dim`)
which is deterministic and stable across runs. Production wraps
OpenAI's `text-embedding-3-large` with the same surface — the
pipeline doesn't care.

## Vector index

`VectorIndex` exposes the FAISS contract (`add(matrix)`,
`search(q, k) -> [(score, chunk)]`). The implementation is
brute-force numpy matmul — sub-10ms on 15k documents at 256 dim.

For 100k+ docs swap in `faiss.IndexHNSWFlat` (logarithmic search,
~1ms/query at recall@10 ≥ 0.95). The `add` / `search` signatures
match exactly.

## Hallucination guard (v3)

A naïve baseline scores answer-vs-evidence token overlap. Real
production stacks layer this on top of NLI (does the evidence
*entail* the claim?) and per-claim attribution (split the answer
into atomic claims, score each separately).

`score_grounding` returns:
- `score`: fraction of answer tokens (≥3 chars) that appear in any
  retrieved chunk. 1.0 = fully grounded, 0.0 = pure hallucination.
- `per_evidence_overlap`: per-chunk fraction so the UI can
  highlight "claim X comes from chunk 2".

The pipeline flags answers with `score < grounding_threshold`
(default 0.3). Refusal mode (return "I don't know" instead of a
flagged answer) is one config flag away in production.

## Streaming

`/query/stream` emits three SSE event types in order:

- `event: citations` — list of (doc_id, chunk_id, score, text) the
  query retrieved. Lets the UI render evidence panes before the
  answer streams.
- `event: token` — one frame per token of generated text.
- `event: done` — sentinel.

The frame shape matches the rest of the SAY-5 portfolio so a single
subscriber can receive streams from clinicalrag, sensorflow, etc.

## What's deliberately not here

- **Full FAISS integration.** The shape is FAISS-compatible; we
  ship the brute-force backend so tests are hermetic. Users
  install `faiss-cpu` and replace `VectorIndex.search` to upgrade.
- **OpenAI client wiring.** `Embedder` and `Generator` are
  Protocols; the OpenAI adapter is a 30-line wrapper over the
  official SDK that lives alongside the stub.
- **Multi-tenant isolation.** A single `Pipeline` covers one
  document space. For multi-tenant production wrap N pipelines
  keyed on tenant id.
