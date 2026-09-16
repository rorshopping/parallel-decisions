#!/usr/bin/env python3
"""Invoice review: a real integration shape, using the package as a library.

The question set follows TypeSafe's published invoice workflow (seven checks per
invoice line plus a document-level block). The wording here is ours, not theirs —
their eval data is fetched, never vendored — but the *shape* is the useful part and
it is the same shape an accounts-payable review needs:

  per line:   kind, scope, completion, rebilled, owner_declined, unexplained_fee,
              rate_differs
  document:   statement_not_invoice, unusual_urgency, bank_change_claimed_in_comms,
              different_entity, price_basis, billed_above_basis, too_ambiguous,
              unexplained_charges, tax_two_rates, adjustment_duplicates_line

Run it on a packet (the invoice/statement text plus the line items):

    python examples/invoice_review.py --text packet.txt --lines lines.txt
    python examples/invoice_review.py --text packet.txt --lines lines.txt \
        --calibration calibration.json --json

`--lines` is a file with one line-item description per line (that is what makes the
per-line questions concrete). `--dump-schema` writes the document-level schema to
JSON instead of running anything, so `pd decide` can be used directly.

Memory note: a 40-line invoice over an 8k-token packet is ~290 decision rows. They
are chunked automatically (`memory_budget_gb`); each chunk costs a cache copy, so
raise the budget if you have headroom and lower it if you hit memory errors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from parallel_decisions import Decider, Schema  # noqa: E402

# ---------------------------------------------------------------- templates
PER_LINE = {
    "kind": {
        "type": "enum",
        "choices": {
            "work_or_goods": "the line charges for work performed or goods supplied",
            "tax": "the line is a tax or duty",
            "charge_on_top": "the line is a fee, surcharge or handling charge",
            "credit_or_discount": "the line reduces the amount payable",
        },
        "description": "What is invoice line {n} ({line})?",
    },
    "scope": {
        "type": "enum",
        "choices": {
            "within_scope": "the work fits what the contract or approval covers",
            "extra_approved": "beyond the contract, but separately approved in writing",
            "extra_unapproved": "beyond the contract with no approval on record",
            "not_ours": "belongs to another party or project",
            "unsure": "the packet does not say",
        },
        "description": ("If invoice line {n} ({line}) is work, does it fall inside what "
                        "has been agreed or approved?"),
    },
    "completion": {
        "type": "enum",
        "choices": {
            "delivered": "the work or goods are evidenced as delivered",
            "in_progress": "partially delivered, with evidence",
            "not_delivered": "no evidence of delivery",
            "not_applicable": "delivery does not apply to this line",
        },
        "description": "If invoice line {n} ({line}) is work, what does the evidence show about completion?",
    },
    "rebilled": {
        "type": "boolean",
        "choices": {
            "true": "a specific prior invoice already billed this same item",
            "false": "no prior invoice already billed this item; do not answer true without one",
        },
        "description": "Does another invoice in the record already bill invoice line {n} ({line})?",
    },
    "owner_declined": {
        "type": "boolean",
        "choices": {"true": "the owner or approver declined this line",
                    "false": "no decline of this line is recorded"},
        "description": "Has the project owner or approver declined invoice line {n} ({line})?",
    },
    "unexplained_fee": {
        "type": "boolean",
        "choices": {"true": "the amount is charged with no basis in the packet",
                    "false": "the packet explains the amount"},
        "description": "Is invoice line {n} ({line}) charged with no explanation in the packet?",
    },
    "rate_differs": {
        "type": "boolean",
        "choices": {"true": "the rate on this line differs from the contract or rate card",
                    "false": "the rate matches, or no comparable rate is stated"},
        "description": "Does the rate on invoice line {n} ({line}) differ from the rate the contract or rate card states?",
    },
}

DOCUMENT_FIELDS = {
    "statement_not_invoice": {
        "type": "boolean",
        "choices": {"true": "the document is a statement or summary, not an invoice",
                    "false": "the document is an invoice"},
        "description": "Is the document a statement of account rather than an invoice?",
    },
    "unusual_urgency": {
        "type": "boolean",
        "choices": {"true": "a message pressures for payment in hours or days",
                    "false": "no unusual time pressure appears"},
        "description": "Does any message pressure for unusually fast payment?",
    },
    "bank_change_claimed_in_comms": {
        "type": "boolean",
        "choices": {"true": "a message claims new payment or bank details",
                    "false": "no such claim appears"},
        "description": "Does a vendor-side message claim the payment or bank details changed?",
    },
    "different_entity": {
        "type": "boolean",
        "choices": {"true": "the payee or project does not match the order",
                    "false": "the invoice belongs to the expected project and payee"},
        "description": "Does the invoice appear to belong to a different business, project or entity?",
    },
    "price_basis": {
        "type": "enum",
        "choices": {
            "not_stated": "no comparable price basis is stated",
            "amount_matches": "the amount matches the stated basis",
            "below_stated": "it is billed below the stated basis, which needs explaining",
            "above_stated": "it is billed above the stated basis",
            "no_contract": "there is no contract or statement of work in the packet",
        },
        "description": "Which option describes the price the contract or statement of work states for this invoice?",
    },
    "billed_above_basis": {
        "type": "boolean",
        "choices": {"true": "the amount billed exceeds the stated price basis",
                    "false": "the amount billed does not exceed it"},
        "description": "Is the amount billed above the price basis the contract states?",
    },
    "too_ambiguous": {
        "type": "boolean",
        "choices": {"true": "at least one substantive line is described too vaguely to check",
                    "false": "every substantive line can be checked"},
        "description": "Is at least one substantive line described too generically to verify?",
    },
    "unexplained_charges": {
        "type": "boolean",
        "choices": {"true": "a positive adjustment or surcharge has no stated basis",
                    "false": "positive adjustments are explained"},
        "description": "Do the invoice's fees, surcharges or handling charges lack a stated basis?",
    },
    "tax_two_rates": {
        "type": "boolean",
        "choices": {"true": "tax is charged at two different rates on this invoice",
                    "false": "tax is charged at one rate"},
        "description": "Is tax charged at two different rates on this invoice?",
    },
    "adjustment_duplicates_line": {
        "type": "boolean",
        "choices": {"true": "a summary adjustment charges for something a line already bills",
                    "false": "no adjustment duplicates a line"},
        "description": "Does a summary adjustment charge for something a line item already bills?",
    },
}


def read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def build_schema(lines: list[str]) -> tuple[dict, dict[str, str]]:
    """Document-level fields plus one block of questions per invoice line."""
    fields: dict[str, dict] = dict(DOCUMENT_FIELDS)
    sources: dict[str, str] = {name: "document" for name in DOCUMENT_FIELDS}
    for i, line in enumerate(lines, 1):
        for key, spec in PER_LINE.items():
            name = f"line_{i}_{key}"
            fields[name] = {
                "type": spec["type"],
                "choices": {k: v for k, v in spec["choices"].items()},
                "description": spec["description"].format(n=i, line=line),
            }
            sources[name] = line
    return fields, sources


def route(probability: float, act_at: float, review_at: float) -> str:
    if probability >= act_at:
        return "act"
    if probability >= review_at:
        return "review"
    return "refuse"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="path to the invoice/statement packet text")
    ap.add_argument("--lines", help="path to a file with one line-item description per line")
    ap.add_argument("--model", default=None)
    ap.add_argument("--calibration", default=None, help="fitted calibrator JSON")
    ap.add_argument("--dump-schema", metavar="OUT",
                    help="write the document-level schema to OUT and exit")
    ap.add_argument("--act-at", type=float, default=0.9,
                    help="confidence at or above which a field is routed to 'act'")
    ap.add_argument("--review-at", type=float, default=0.6)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.dump_schema:
        with open(args.dump_schema, "w", encoding="utf-8") as fh:
            json.dump({"fields": DOCUMENT_FIELDS}, fh, indent=2)
            fh.write("\n")
        print(f"wrote {args.dump_schema} ({len(DOCUMENT_FIELDS)} document-level fields; "
              f"per-line questions are generated at call time)")
        return

    if not args.text or not args.lines:
        raise SystemExit("--text and --lines are required (or use --dump-schema)")
    with open(args.text, encoding="utf-8") as fh:
        context = fh.read()
    lines = read_lines(args.lines)

    fields, sources = build_schema(lines)
    schema = Schema(fields)
    print(f"{len(lines)} line item(s), {len(schema)} fields, "
          f"{len(schema) + sum(1 for f in schema.fields.values() if f.is_multi)} decision rows",
          file=sys.stderr)
    if args.calibration:
        print(f"using calibrator {args.calibration}", file=sys.stderr)

    decider = Decider(model_id=args.model, calibration=args.calibration,
                      verbose=args.verbose)
    result = decider.decide(context, schema)

    findings = []
    for name, fv in result.items():
        findings.append({
            "field": name,
            "value": fv.value,
            "probability": round(fv.probability, 4),
            "raw_probability": round(fv.raw_probability or fv.probability, 4),
            "route": route(fv.probability, args.act_at, args.review_at),
            "source": sources.get(name, ""),
        })

    summary = {band: sum(1 for f in findings if f["route"] == band)
               for band in ("act", "review", "refuse")}
    actionable = [f for f in findings
                  if f["route"] == "act" and f["value"] not in (False, [], "false")]
    payload = {
        "model": result.model,
        "calibrated": result.calibrated,
        "lines": len(lines),
        "fields": len(schema),
        "latency_ms": round(result.latency_ms, 1),
        "chunks": result.chunks,
        "routing": summary,
        "flags": [f for f in actionable if f["field"] != "price_basis"],
        "all": findings,
    }

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return

    print()
    print(f"{len(lines)} lines, {len(schema)} fields, {result.chunks} chunk(s), "
          f"{result.latency_ms:.0f} ms; confidence: "
          f"{'calibrated' if result.calibrated else 'raw softmax'}")
    print(f"routing: act {summary['act']}, review {summary['review']}, "
          f"refuse {summary['refuse']}  (act >= {args.act_at}, "
          f"refuse < {args.review_at})")
    print()
    print(f"{'field':<34} {'value':<22} {'conf':>7} {'raw':>7}  route")
    print("-" * 84)
    for f in findings:
        value = f["value"] if not isinstance(f["value"], list) else ",".join(f["value"])
        mark = "  <-- act" if f["route"] == "act" else ""
        print(f"{f['field']:<34} {str(value):<22} {f['probability']:>6.1%} "
              f"{f['raw_probability']:>6.1%}  {f['route']}{mark}")
    if payload["flags"]:
        print()
        print("flagged for action (act bucket, value is not false/empty):")
        for f in payload["flags"]:
            print(f"  {f['field']}: {f['value']} ({f['probability']:.1%})")
    print()
    print("Review policy reminder: act-bucket error rates are only meaningful once the "
          "calibrator was fitted on labelled data from this domain. Measure it with "
          "examples/routing.py before automating anything.")


if __name__ == "__main__":
    main()
