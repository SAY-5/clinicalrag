"""CLI: `clinicalrag serve` runs the FastAPI server."""

from __future__ import annotations

import sys

import uvicorn

from clinicalrag import Pipeline, create_app


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] != "serve":
        print("usage: clinicalrag serve [--host 127.0.0.1] [--port 8000]", file=sys.stderr)
        return 2
    host = "127.0.0.1"
    port = 8000
    a = iter(args[1:])
    for tok in a:
        if tok == "--host":
            host = next(a, host)
        elif tok == "--port":
            port = int(next(a, str(port)))
    p = Pipeline()
    p.ingest(
        "demo",
        "Aspirin reduces fever and inflammation. "
        "Paracetamol is the safer choice for children.",
    )
    app = create_app(p)
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
