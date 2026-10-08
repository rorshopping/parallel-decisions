#!/usr/bin/env python3
"""Benchmark the llamacpp backend on a fixed decision batch, CPU vs GPU.

Runs the same smoke-test-shaped workload (context + 4-field schema with one
collision field) through Decider(backend="llamacpp") and records prefill/pass
wall time plus the decision values, so a CPU run and a GPU run can be diffed
for the "decisions unchanged for fixed inputs" gate.

    python benchmarks/bench_llamacpp_device.py --model <gguf> \
        [--n-gpu-layers -1|0] [--runs 3] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parallel_decisions import Decider, Schema

CONTEXT = """
Transaction alert TX-98421. A cardholder in Seattle, who has never made a
crypto transfer, attempts to send $49,500 to a crypto exchange in Cyprus at
03:14 UTC. The device is an unrecognised Linux browser seen for the first time
four minutes ago, from a known Tor exit node in Frankfurt. Three further
attempts fired in the previous ten minutes from Singapore, London and Frankfurt.
SMS two-factor authentication appears to have been bypassed.
"""

SCHEMA = Schema({
    "is_fraudulent": {"type": "boolean", "description": "Whether this transaction is fraudulent"},
    "risk_tier": {
        "type": "enum",
        "choices": {"low": "ordinary activity", "medium": "worth a look",
                    "high": "strong signs of fraud", "critical": "act immediately"},
        "description": "Risk tier for this transaction",
    },
    "actions": {
        "type": "multi",
        "choices": {"hold_payment": "stop the transfer",
                    "contact_cardholder": "call the customer",
                    "close_session": "terminate the active session"},
        "description": "Which containment actions apply",
    },
    "charge_scope": {
        "type": "enum",
        "choices": ["extra_approved", "extra_unapproved", "not_ours"],
        "description": "If the cardholder has a charge agreement, does this fit it?",
    },
})


def _values_dict(result):
    values_attr = getattr(result, "values", None)
    view = values_attr() if callable(values_attr) else values_attr
    return {fv.name: {"value": fv.value, "probability": round(float(fv.probability), 9)}
            for fv in view}


def vram_poller(stop, samples):
    def poll():
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                samples.append(int(out.splitlines()[0]))
            except Exception:
                pass
            stop.wait(0.2)
    threading.Thread(target=poll, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-gpu-layers", type=int, default=None,
                    help="-1 all layers on GPU, 0 CPU path, omit for auto")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import llama_cpp
    print(f"llama_cpp {llama_cpp.__version__}", flush=True)

    stop = threading.Event()
    samples = []
    vram_poller(stop, samples)

    kwargs = {}
    if args.n_gpu_layers is not None:
        kwargs["n_gpu_layers"] = args.n_gpu_layers
    t0 = time.perf_counter()
    decider = Decider(backend="llamacpp", model_id=args.model, **kwargs)
    decider.load()
    load_s = time.perf_counter() - t0
    print(f"load_s={load_s:.2f}", flush=True)

    runs = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        result = decider.decide(CONTEXT, SCHEMA)
        wall_ms = (time.perf_counter() - t0) * 1000
        tel = getattr(result, "telemetry", None) or (result.get("telemetry") if isinstance(result, dict) else {})
        values = _values_dict(result)
        rec = {
            "run": i + 1,
            "wall_ms": round(wall_ms, 1),
            "prefill_ms": tel.get("prefill_ms"),
            "pass_ms": tel.get("pass_ms"),
            "device": tel.get("device"),
            "n_gpu_layers": tel.get("n_gpu_layers"),
            "values": values,
            "telemetry": {k: v for k, v in tel.items()
                          if isinstance(v, (int, float, str, bool))},
        }
        runs.append(rec)
        print(json.dumps({k: rec[k] for k in ("run", "wall_ms", "prefill_ms", "pass_ms", "device")}),
              flush=True)

    stop.set()
    time.sleep(0.3)

    def _median(key):
        vals = [r[key] for r in runs if r.get(key) is not None]
        return round(statistics.median(vals), 1) if vals else None

    summary = {
        "model": Path(args.model).name,
        "model_size_gb": round(Path(args.model).stat().st_size / 1e9, 2),
        "llama_cpp_version": llama_cpp.__version__,
        "n_gpu_layers_arg": args.n_gpu_layers,
        "load_s": round(load_s, 2),
        "wall_ms_median": round(statistics.median(r["wall_ms"] for r in runs), 1),
        "prefill_ms_median": _median("prefill_ms"),
        "pass_ms_median": _median("pass_ms"),
        "peak_vram_mib": max(samples) if samples else None,
        "values": runs[-1]["values"],
        "runs": runs,
    }
    print("SUMMARY " + json.dumps({**summary, "runs": [
        {k: v for k, v in r.items() if k not in ("values", "telemetry")} for r in summary["runs"]
    ]}), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
