#!/usr/bin/env python3
"""Opt-in smoke test against the real model. Downloads ~4.3 GB on first run.

    .venv/bin/python smoke_test.py [model-id]

Checks, in order:
  1. a plain 3-field decision (boolean + enum)
  2. a multi-select field (one yes/no decision per choice)
  3. a field with a deliberate token collision (exercises the exact-scoring path)
  4. calibration: a fitted calibrator changes the number, not the answer
  5. latency telemetry and chunking are populated
"""

from __future__ import annotations

import sys

from parallel_decisions import CalibrationRecord, Calibrator, Decider, Schema

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
    # `extra_approved` / `extra_unapproved` share their first token, so this field
    # must go through the exact sequence-scoring path.
    "charge_scope": {
        "type": "enum",
        "choices": ["extra_approved", "extra_unapproved", "not_ours"],
        "description": "If the cardholder has a charge agreement, does this fit it?",
    },
})


def expected_rows(schema: Schema) -> int:
    """Decision rows the engine must evaluate: one per field, plus one per
    multi-select choice."""
    return sum(len(f.choices) if f.is_multi else 1 for f in schema.fields.values())


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else None
    decider = Decider(model_id=model, verbose=True)
    result = decider.decide(CONTEXT, SCHEMA)

    print()
    for name, value in result.items():
        marker = "  [multi]" if isinstance(value.value, list) else ""
        print(f"{name:16} {str(value.value):28} {value.probability:6.1%}{marker}")
    print()
    print(f"model={result.model}")
    print(f"latency={result.latency_ms:.0f}ms "
          f"(prefill {result.prefill_ms:.0f} + passes {result.pass_ms:.0f}, "
          f"{result.chunks} chunk(s), {result.fields_evaluated} decision rows)")
    print(f"calibrated={result.calibrated}")

    failures = []
    if not isinstance(result["is_fraudulent"].value, bool):
        failures.append("boolean field did not return a bool")
    if result["risk_tier"].value not in SCHEMA["risk_tier"].choices:
        failures.append("enum field returned a value outside its choices")
    if not isinstance(result["actions"].value, list):
        failures.append("multi field did not return a list")
    if not all(a in SCHEMA["actions"].choices for a in result["actions"].value):
        failures.append("multi field returned a value outside its choices")
    if result.fields_evaluated != expected_rows(SCHEMA):
        failures.append("unexpected decision-row count "
                        f"({result.fields_evaluated} != {expected_rows(SCHEMA)})")
    if not (0.0 <= result["risk_tier"].probability <= 1.0):
        failures.append("probability out of range")
    if result.latency_ms <= 0:
        failures.append("latency telemetry missing")

    # calibration must rescale the number and leave the answer alone
    raw_value = result["risk_tier"].value
    raw_prob = result["risk_tier"].probability
    records = [CalibrationRecord({"true": 0.95, "false": 0.05}, correct=(i % 4 != 0))
               for i in range(40)]
    calibrator = Calibrator.fit(records, method="temperature")
    calibrated = calibrator.transform(result["risk_tier"].distribution)
    top = max(calibrated, key=calibrated.__getitem__)
    print()
    print(f"calibration check: {raw_prob:.1%} -> {max(calibrated.values()):.1%} "
          f"(answer {'unchanged' if top == raw_value else 'CHANGED'})")
    if top != raw_value:
        failures.append("calibration changed the chosen answer")
    if abs(max(calibrated.values()) - raw_prob) < 1e-9:
        failures.append("calibration did not change the confidence at all")

    if failures:
        print()
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print()
    print("smoke test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
