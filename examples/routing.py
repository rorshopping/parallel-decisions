#!/usr/bin/env python3
"""Threshold routing: act / review / refuse, with a measured error rate.

This is the point of calibration. Given labelled decisions, it answers:

    "If I only act on answers the model is this confident about,
     what error rate do I inherit, and how much work do I skip?"

Everything is measured **out of sample**: the calibrator is refitted inside a
k-fold loop and each fold is routed by a calibrator that never saw it. Without
that step the table is optimistic, because the thresholds and the calibrator
would both be tuned on the data they are scored on.

Usage (no model needed — works on a recorded run):

    python examples/routing.py --data rlcd-research/evals/results/calibration/qwen2.5-7b.jsonl
    python examples/routing.py --data labels.jsonl --max-error 0.005 --calibrator calib.json

Labels are the same records `pd calibrate` consumes: one JSON object per line with
`distribution` (or `probability`) plus `correct` or `target`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from parallel_decisions import (  # noqa: E402
    Calibrator,
    confidence,
    fit_calibration,
    load_records,
    wilson_interval,
)


def cross_validated_confidences(records, method: str, folds: int, select_by: str):
    """Per-record confidence from a calibrator that never saw that record."""
    order = sorted(range(len(records)), key=lambda i: (records[i].correct, i))
    n_folds = max(2, min(folds, len(records)))
    confs: list[float] = [0.0] * len(records)
    kinds: list[str] = []
    for f in range(n_folds):
        test_idx = order[f::n_folds]
        train_idx = [i for i in range(len(records)) if i not in set(test_idx)]
        if not test_idx or not train_idx:
            continue
        train = [records[i] for i in train_idx]
        fit = fit_calibration(train, method="auto" if method == "auto" else method,
                              folds=max(2, min(folds, len(train))), select_by=select_by)
        cal = fit.calibrator
        kinds.append(cal.kind)
        for i in test_idx:
            confs[i] = confidence(cal.transform(records[i].distribution))
    return confs, kinds


def sweep(confs, corrects, max_errors=(0.0, 0.0025, 0.005, 0.01, 0.02, 0.05)):
    """For each error budget: the lowest threshold that meets it, and its coverage."""
    rows = []
    for budget in max_errors:
        # candidate thresholds: the observed confidences themselves
        best = None
        for t in sorted(set(round(c, 6) for c in confs)):
            taken = [ok for c, ok in zip(confs, corrects) if c >= t]
            if not taken:
                continue
            errors = sum(1 for ok in taken if not ok)
            rate = errors / len(taken)
            if rate <= budget or errors == 0:
                best = {"budget": budget, "threshold": t, "taken": len(taken),
                        "coverage": len(taken) / len(confs), "errors": errors, "risk": rate}
                break
        if best is None:  # even the strictest threshold cannot meet the budget
            t = max(confs)
            taken = [ok for c, ok in zip(confs, corrects) if c >= t]
            errors = sum(1 for ok in taken if not ok)
            best = {"budget": budget, "threshold": t, "taken": len(taken),
                    "coverage": len(taken) / len(confs), "errors": errors,
                    "risk": errors / max(1, len(taken)), "unmet": True}
        rows.append(best)
    return rows


def refuse_threshold(confs, corrects) -> tuple[float, dict | None]:
    """Where confidence stops carrying signal: the 50%-accuracy crossing.

    Returns (threshold, evidence) where evidence is None when no crossing exists —
    i.e. the data gives no reason to give up on low-confidence answers.
    """
    xs = sorted(set(round(c, 6) for c in confs))
    best = None
    for t in xs:
        low = [ok for c, ok in zip(confs, corrects) if c < t]
        if len(low) < 10:
            continue
        acc = sum(1 for ok in low if ok) / len(low)
        if acc <= 0.5:
            best = {"threshold": t, "n": len(low), "accuracy": acc}
            break
    return (best["threshold"] if best else 0.0), best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="labelled records (.jsonl/.json)")
    ap.add_argument("--calibrator", help="use this fitted calibrator instead of refitting per fold")
    ap.add_argument("--method", default="auto", choices=("auto", "temperature", "platt", "isotonic"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--select-by", default="ece_adaptive")
    ap.add_argument("--max-error", type=float, default=0.01,
                    help="error budget for the act bucket (default 1%%)")
    ap.add_argument("--json", action="store_true", help="print machine-readable output")
    ap.add_argument("--live", metavar="CONTEXT", help="also route one live decision through the model")
    ap.add_argument("--live-schema", help="schema JSON for --live")
    args = ap.parse_args()

    records = load_records(args.data)
    corrects = [r.correct for r in records]
    raw = [r.confidence for r in records]

    if args.calibrator:
        cal = Calibrator.from_json(args.calibrator)
        confs = [confidence(cal.transform(r.distribution)) for r in records]
        kinds = [cal.kind]
        honest = False
    else:
        confs, kinds = cross_validated_confidences(records, args.method, args.folds, args.select_by)
        honest = True

    from parallel_decisions import adaptive_ece, auroc

    report = {
        "n": len(records),
        "accuracy": sum(1 for ok in corrects if ok) / len(records),
        "out_of_sample": honest,
        "methods_fitted_per_fold": kinds,
        "raw": {"ece_adaptive": adaptive_ece(raw, corrects), "auroc": auroc(raw, corrects)},
        "calibrated": {"ece_adaptive": adaptive_ece(confs, corrects), "auroc": auroc(confs, corrects)},
        "sweep": sweep(confs, corrects),
    }
    t_refuse, refuse_evidence = refuse_threshold(confs, corrects)
    report["refuse"] = {"threshold": t_refuse, "evidence": refuse_evidence}

    chosen = next((row for row in report["sweep"] if row["budget"] >= args.max_error), None)
    report["policy"] = {
        "act": chosen,
        "review": {"lower": t_refuse, "upper": chosen["threshold"] if chosen else None},
    }

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        mode = "out-of-sample (calibrator refit per fold)" if honest \
            else f"calibrator {args.calibrator} applied as-is"
        print(f"routing policy from {args.data}")
        print(f"  {len(records)} labelled decisions, top-choice accuracy "
              f"{report['accuracy']:.1%}, {mode}")
        print(f"  methods fitted per fold: {', '.join(sorted(set(kinds)))}")
        print()
        print(f"  calibration: ECE(adaptive) {report['raw']['ece_adaptive']:.3f} -> "
              f"{report['calibrated']['ece_adaptive']:.3f}   "
              f"AUROC {report['raw']['auroc']:.3f} (unchanged by design)")
        print()
        print("  error budget -> threshold, coverage, act-bucket error")
        print(f"  {'budget':>7} {'threshold':>10} {'acted on':>9} {'share':>7} {'errors':>7} "
              f"{'error rate':>11} {'95% upper':>10}")
        for row in report["sweep"]:
            lo, hi = wilson_interval(row["errors"], row["taken"])
            flag = "  (budget unreachable)" if row.get("unmet") else ""
            print(f"  {row['budget']:>7.2%} {row['threshold']:>10.4f} {row['taken']:>9} "
                  f"{row['coverage']:>7.1%} {row['errors']:>7} {row['risk']:>11.2%} "
                  f"{hi:>10.2%}{flag}")
        print()
        if chosen:
            lo, hi = wilson_interval(chosen["errors"], chosen["taken"])
            print(f"  POLICY (budget {args.max_error:.2%}):")
            if chosen.get("unmet"):
                print(f"    act     conf >= {chosen['threshold']:.4f}   "
                      f"{chosen['taken']}/{len(records)} = {chosen['coverage']:.1%} of work, "
                      f"error rate {chosen['risk']:.2%} (95% upper {hi:.2%}) — "
                      f"budget NOT met, this is the best available")
            else:
                print(f"    act     conf >= {chosen['threshold']:.4f}   "
                      f"{chosen['taken']}/{len(records)} = {chosen['coverage']:.1%} of work, "
                      f"error rate {chosen['risk']:.2%} (95% upper {hi:.2%})")
            print(f"    review  {t_refuse:.4f} <= conf < {chosen['threshold']:.4f}   "
                  f"{sum(1 for c in confs if t_refuse <= c < chosen['threshold'])} decisions")
            print(f"    refuse  conf < {t_refuse:.4f}   "
                  f"{sum(1 for c in confs if c < t_refuse)} decisions"
                  + ("" if refuse_evidence else "  (no 50% crossing in this data)"))
        else:
            print("  no threshold meets any budget on this data")

    if args.live:
        if not args.live_schema:
            raise SystemExit("--live needs --live-schema")
        from parallel_decisions import Decider, Schema
        cal = Calibrator.from_json(args.calibrator) if args.calibrator else fit_calibration(records).calibrator
        decider = Decider(calibration=cal)
        result = decider.decide(args.live, Schema.from_json(args.live_schema))
        print()
        print("  live decision:")
        high = chosen["threshold"] if chosen else 1.0
        low = t_refuse
        for name, fv in result.items():
            band = "act" if fv.probability >= high else ("refuse" if fv.probability < low else "review")
            print(f"    {name:<22} {str(fv.value):<18} {fv.probability:6.1%} "
                  f"(raw {fv.raw_probability:.1%})  -> {band}")


if __name__ == "__main__":
    main()
