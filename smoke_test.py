#!/usr/bin/env python3
"""Opt-in smoke test against the real model. Downloads ~4.3 GB on first run.

    .venv/bin/python smoke_test.py [model-id]

Checks, in order:
  1. a plain 3-field decision (boolean + enum)
  2. a field with a deliberate token collision (exercises the exact-scoring path)
  3. latency telemetry is populated
"""

from __future__ import annotations

import sys

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
        "description": "Overall risk tier",
        "choices": {
            "LOW": "normal behaviour",
            "MEDIUM": "one weak anomaly",
            "HIGH": "several anomalies or a high-value transfer",
            "CRITICAL": "account takeover or confirmed attacker control",
        },
    },
    "recommended_action": {
        "type": "enum",
        "description": "Immediate mitigation",
        "choices": ["APPROVE", "REVIEW", "BLOCK"],
    },
})


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else None
    decider = Decider(model_id=model, verbose=True)

    print("\n--- run 1: basic decision ---")
    result = decider.decide(CONTEXT, SCHEMA)
    for name, value in result.items():
        print(f"  {name:20} {str(value.value):10} {value.probability:6.1%}")
    print(f"  latency {result.latency_ms:.0f} ms "
          f"(prefill {result.prefill_ms:.0f}, passes {result.pass_ms:.0f}, {result.chunks} chunk(s))")

    assert result["is_fraudulent"].value is True, "expected fraud=true on this context"
    assert result["risk_tier"].value in ("HIGH", "CRITICAL"), "expected high or critical risk"
    assert result["recommended_action"].value in ("BLOCK", "REVIEW"), "expected block or review"

    print("\n--- run 2: collision path (choices sharing a first token) ---")
    collision_schema = Schema({
        "line_1_scope": {
            "type": "enum",
            "description": "How this invoice line is scoped",
            "choices": ["extra_approved", "extra_unapproved", "unsure"],
        },
    })
    compiled = collision_schema.compile(decider.tokenizer)[0]
    print(f"  collision detected: {compiled.collision}")
    assert compiled.collision, "expected these choices to collide"
    result2 = decider.decide(CONTEXT, collision_schema)
    for name, value in result2.items():
        print(f"  {name:20} {str(value.value):16} {value.probability:6.1%}")
    assert result2["line_1_scope"].value in ("extra_approved", "extra_unapproved", "unsure")

    print("\nSMOKE TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
