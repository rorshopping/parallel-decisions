#!/usr/bin/env python3
"""Tiny HTTP server (standard library only) so non-Python callers can use the engine.

    python examples/serve.py --port 8000

    curl -s localhost:8000/decide -H 'content-type: application/json' -d '{
      "context": "The laptop will not turn on after a power cut.",
      "schema": {"fields": {"category": {"type": "enum",
                                        "choices": ["HARDWARE", "SOFTWARE"],
                                        "description": "issue type"}}}
    }'
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from parallel_decisions import DEFAULT_MODEL, Decider, Schema

DECIDER: Decider | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "parallel-decisions/0.1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"ok": True, "model": DECIDER.model_id if DECIDER else None})
        else:
            self._send(404, {"error": "use POST /decide"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/decide":
            self._send(404, {"error": "use POST /decide"})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            context = request["context"]
            schema = Schema(request["schema"])
        except Exception as exc:
            self._send(400, {"error": f"bad request: {exc}"})
            return
        try:
            result = DECIDER.decide(context, schema)
        except Exception as exc:  # pragma: no cover - runtime failure
            self._send(500, {"error": str(exc)})
            return
        self._send(200, {
            "decisions": result.full_json(),
            "latency_ms": round(result.latency_ms, 1),
            "model": result.model,
        })

    def log_message(self, fmt: str, *args) -> None:  # quieter logging
        return


def main() -> None:
    global DECIDER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    DECIDER = Decider(model_id=args.model)
    DECIDER.load()
    print(f"listening on http://{args.host}:{args.port}  (model: {DECIDER.model_id})")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
