"""FastAPI server. Endpoints:

POST /ingest        — body: {doc_id, text}; returns chunk count
POST /query         — body: {question, top_k}; returns Answer JSON
GET  /query/stream  — query parameters; returns SSE token stream
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from clinicalrag.rag import Pipeline, Query


class IngestRequest(BaseModel):
    model_config = {"extra": "forbid"}
    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    chunk_size: int = 200
    overlap: int = 40


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    p = pipeline or Pipeline()
    app = FastAPI(title="clinicalrag")

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict:
        n = p.ingest(req.doc_id, req.text, chunk_size=req.chunk_size, overlap=req.overlap)
        return {"chunks": n, "index_size": p.index_size()}

    @app.post("/query")
    def query(q: Query) -> dict:
        a = p.query(q)
        return {
            "text": a.text,
            "grounding": a.grounding,
            "flagged": a.flagged,
            "flag_reason": a.flag_reason,
            "citations": [asdict(c) for c in a.citations],
        }

    @app.get("/query/stream")
    def stream(question: str, top_k: int = 5) -> StreamingResponse:
        q = Query(question=question, top_k=top_k)

        async def gen():
            qv = p.embedder.embed(q.question)
            hits = p._index.search(qv, q.top_k)
            from clinicalrag.rag import Citation as Cit
            cits = [
                Cit(doc_id=c.doc_id, chunk_id=c.chunk_id, score=s, text=c.text)
                for s, c in hits
            ]
            yield 'event: citations\ndata: ' + json.dumps([asdict(c) for c in cits]) + '\n\n'
            for tok in p.generator.stream(q.question, cits):
                yield f'event: token\ndata: {json.dumps(tok)}\n\n'
                await asyncio.sleep(0)
            yield 'event: done\ndata: {}\n\n'

        return StreamingResponse(gen(), media_type="text/event-stream")

    app.state.pipeline = p
    return app
