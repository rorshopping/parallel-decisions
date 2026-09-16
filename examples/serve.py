#!/usr/bin/env python3
"""Tiny HTTP server (standard library only) so non-Python callers can use the engine.

    python examples/serve.py --port 8000

    curl -s localhost:8000/decide -H 'content-type: application/json' -d '{
      "context": "The laptop will not turn on after a power cut.",
      "schema": {"fields": {"category": {"type": "enum",
                                        "choices": ["HARDWARE", "SOFTWARE"],
                                        "description": "issue type"}}}
    }'

Concurrency: one model, so requests are **queued**, not refused. The Decider's lock
serialises calls; `--queue-timeout` bounds how long a request waits (default 600 s,
i.e. effectively queue). Set `--queue-timeout 0` to fail fast with 503 instead.
Add `--calibration path.json` (or PD_CALIBRATION) to serve calibrated confidences.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from parallel_decisions import DEFAULT_MODEL, ConcurrencyError, Decider, Schema

DECIDER: Decider | None = None
STATS = {"served": 0, "rejected": 0, "queued": 0}
STATS_LOCK = threading.Lock()


def _bump(key: str) -> None:
    with STATS_LOCK:
        STATS[key] = STATS.get(key, 0) + 1


class Handler(BaseHTTPRequestHandler):
    server_version = "parallel-decisions/0.2"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            with STATS_LOCK:
                stats = dict(STATS)
            self._send(200, {"ok": True, "model": DECIDER.model_id if DECIDER else None,
                             "busy": bool(DECIDER and DECIDER._lock.locked()), **stats})
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
        if DECIDER._lock.locked():
            _bump("queued")
        try:
            result = DECIDER.decide(context, schema)
        except ConcurrencyError as exc:
            _bump("rejected")
            self._send(503, {"error": str(exc), "retry": True})
            return
        except Exception as exc:  # pragma: no cover - runtime failure
            self._send(500, {"error": str(exc)})
            return
        _bump("served")
        self._send(200, {
            "decisions": result.full_json(),
            "calibrated": result.calibrated,
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
    parser.add_argument("--model", default=None, help=f"default: pd.toml, else {DEFAULT_MODEL}")
    parser.add_argument("--calibration", default=None, help="fitted calibrator JSON")
    parser.add_argument("--queue-timeout", type=float, default=600.0,
                        help="seconds a request may wait for the model (0 = fail fast)")
    parser.add_argument("--config", default=None, help="path to pd.toml")
    args = parser.parse_args()

    DECIDER = Decider(model_id=args.model, calibration=args.calibration,
                      lock_timeout_s=args.queue_timeout, config=args.config)
    DECIDER.load()
    mode = "calibrated" if DECIDER.calibrator else "raw softmax"
    print(f"listening on http://{args.host}:{args.port}  "
          f"(model: {DECIDER.model_id}, confidence: {mode}, queue {args.queue_timeout:g}s)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

