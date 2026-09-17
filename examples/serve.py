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
serialises calls; `--queue-timeout` bounds the lock wait, not inference time.
It overrides PD_LOCK_TIMEOUT_S / pd.toml; without either, the default is 600 s.
Set `--queue-timeout 0` to fail fast with 503 instead. This is a localhost example,
not a production HTTP service.
Add `--calibration path.json` (or PD_CALIBRATION) to serve calibrated confidences.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from parallel_decisions import DEFAULT_MODEL, ConcurrencyError, Decider, Schema, load_config

DECIDER: Decider | None = None
STATS = {"served": 0, "rejected": 0, "queued": 0}
STATS_LOCK = threading.Lock()
MAX_BODY_BYTES = 1_000_000


def _as_schema(raw) -> Schema:
    """Accept `{name: spec}`, `{"fields": {...}}` and `Schema.to_json()` output."""
    if not isinstance(raw, dict):
        raise ValueError("schema must be a JSON object")
    if isinstance(raw.get("fields"), (dict, list)):
        raw = raw["fields"]
    if isinstance(raw, list):
        fields = {}
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise ValueError("each schema field must be an object with a name")
            if item["name"] in fields:
                raise ValueError(f"duplicate field name: {item['name']}")
            fields[item["name"]] = item
        raw = fields
    return Schema(raw)


def _bump(key: str) -> None:
    with STATS_LOCK:
        STATS[key] = STATS.get(key, 0) + 1


class Handler(BaseHTTPRequestHandler):
    server_version = "parallel-decisions-example"

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
            self._send(200, {"ok": DECIDER is not None, "model": DECIDER.model_id if DECIDER else None,
                             "busy": bool(DECIDER and DECIDER._lock.locked()), **stats})
        else:
            self._send(404, {"error": "use POST /decide"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/decide":
            self._send(404, {"error": "use POST /decide"})
            return
        if DECIDER is None:
            self._send(503, {"error": "model is not ready", "retry": True})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            if length < 0 or length > MAX_BODY_BYTES:
                raise ValueError(f"content-length must be 0..{MAX_BODY_BYTES}")
            request = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(request, dict):
                raise ValueError("request body must be a JSON object")
            context = request["context"]
            if not isinstance(context, str) or not context.strip():
                raise ValueError("context must be a non-empty string")
            schema = _as_schema(request["schema"])
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


def main(argv: list[str] | None = None) -> None:
    global DECIDER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model", default=None, help=f"default: pd.toml, else {DEFAULT_MODEL}")
    parser.add_argument("--calibration", default=None, help="fitted calibrator JSON")
    parser.add_argument("--queue-timeout", type=float, default=None,
                        help="lock wait seconds (0 = fail fast; config/env, else 600)")
    parser.add_argument("--config", default=None, help="path to pd.toml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    timeout = args.queue_timeout
    if timeout is None:
        timeout = config.lock_timeout_s if config.lock_timeout_s is not None else 600.0
    if not math.isfinite(timeout) or timeout < 0:
        parser.error("queue timeout must be finite and non-negative")
    DECIDER = Decider(model_id=args.model, calibration=args.calibration,
                      lock_timeout_s=timeout, config=config)
    DECIDER.load()
    mode = "calibrated" if DECIDER.calibrator else "raw softmax"
    with ThreadingHTTPServer((args.host, args.port), Handler) as server:
        print(f"listening on http://{args.host}:{args.port}  "
              f"(model: {DECIDER.model_id}, confidence: {mode}, queue {timeout:g}s)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

