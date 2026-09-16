"""Command line interface: `pd validate`, `pd decide`, `pd calibrate`."""

from __future__ import annotations

import argparse
import json
import sys

from .calibration import (
    CalibrationError,
    Calibrator,
    adaptive_ece,
    confidence,
    fit_calibration,
    load_records,
    reliability_table,
    risk_coverage,
)
from .config import ConfigError, load_config
from .engine import DEFAULT_MODEL, Decider
from .schema import Schema, SchemaError


def _read_context(args) -> str:
    if args.context_file:
        with open(args.context_file, encoding="utf-8") as fh:
            return fh.read()
    if args.context:
        return args.context
    # read stdin as a last resort
    data = sys.stdin.read()
    if not data.strip():
        raise SystemExit("no context provided (use --context, --context-file, or stdin)")
    return data


def cmd_config(args) -> int:
    cfg = load_config(args.config)
    if args.json:
        print(json.dumps(cfg.resolved(), indent=2, default=str))
        return 0
    if cfg.source:
        print(f"loaded {cfg.source}")
    else:
        print("no pd.toml found (looked in ./pd.toml and ~/.config/parallel-decisions/pd.toml)")
    resolved = cfg.resolved()
    if not resolved:
        print("no settings configured; built-in defaults apply")
        return 0
    for key, value in resolved.items():
        print(f"  {key} = {value!r}")
    return 0


def cmd_validate(args) -> int:
    schema = Schema.from_json(args.schema)
    print(f"schema OK: {len(schema)} field(s)")
    for f in schema.fields.values():
        if f.is_boolean:
            print(f"  {f.name}: boolean")
        elif f.is_multi:
            print(f"  {f.name}: multi-select, {len(f.choices)} choices "
                  f"({len(f.choices)} decision rows)")
        else:
            print(f"  {f.name}: enum, {len(f.choices)} choices")
    if args.check_tokens:
        decider = Decider(model_id=args.model, warmup=False, config=args.config)
        compiled = schema.compile(decider.tokenizer)
        collisions = [c for c in compiled if c.collision]
        rows = len(compiled)
        if rows != len(schema):
            print(f"  ({rows} decision rows: multi-select fields expand per choice)")
        if collisions:
            print()
            print(f"token collisions in {len(collisions)} field(s) "
                  f"(these take an extra, slower pass):")
            for c in collisions:
                print(f"  {c.row_name}: {c.field.choices if not c.field.is_boolean else ['true', 'false']}")
        else:
            print()
            print("no token collisions")
    return 0


def cmd_decide(args) -> int:
    schema = Schema.from_json(args.schema)
    context = _read_context(args)
    decider = Decider(model_id=args.model, verbose=args.verbose or args.json is False,
                      calibration=args.calibration, config=args.config)
    result = decider.decide(context, schema)

    if args.json:
        payload = {"decisions": result.full_json()}
        if result.calibrated:
            payload["calibrated"] = True
        print(json.dumps(payload, indent=2, default=str))
    else:
        for name, value in result.items():
            raw = f"  (raw {value.raw_probability:.1%})" if result.calibrated else ""
            print(f"{name:24} {str(value.value):20} {value.probability:6.1%}{raw}")
        print()
        print(json.dumps(result.json(), indent=2, default=str))
        print()
        cal = "calibrated" if result.calibrated else "raw softmax (not calibrated)"
        print(f"model={result.model} latency={result.latency_ms:.0f}ms "
              f"(prefill {result.prefill_ms:.0f} + passes {result.pass_ms:.0f}, "
              f"{result.chunks} chunk(s)); confidence: {cal}")
    return 0


def cmd_calibrate(args) -> int:
    """Fit a calibrator from a labelled file and report before/after."""
    try:
        records = load_records(args.data)
    except CalibrationError as exc:
        print(f"calibration data error: {exc}", file=sys.stderr)
        return 2

    fit = fit_calibration(records, method=args.method, folds=args.folds,
                          select_by=args.select_by)
    best = fit.calibrator

    print(f"loaded {len(records)} labelled records from {args.data}")
    print(f"accuracy of the top choice: "
          f"{sum(1 for r in records if r.correct) / len(records):.1%}")
    print()
    print(f"cross-validated comparison ({fit.folds_used} folds, "
          f"selection metric: {args.select_by})")
    print(fit.summary())
    print()
    detail = ""
    if best.kind == "temperature":
        detail = f" (T={best.temperature:.3f})"
    elif best.kind == "platt":
        detail = f" (a={best.platt_a:.3f}, b={best.platt_b:.3f})"
    elif best.kind == "isotonic":
        detail = f" ({len(best.isotonic_x)} points)"
    print(f"selected: {best.kind}{detail}")

    raw_tab = reliability_table([r.confidence for r in records],
                                [r.correct for r in records], bins=args.bins, adaptive=True)
    cal_tab = best.reliability(records, bins=args.bins, adaptive=True)
    raw_ece = adaptive_ece([r.confidence for r in records], [r.correct for r in records])
    cal_ece = adaptive_ece([confidence(best.transform(r.distribution)) for r in records],
                           [r.correct for r in records])
    print()
    print(f"reliability (equal-count bins, adaptive)   ECE {raw_ece:.3f} -> {cal_ece:.3f}")
    print(f"{'bin':>4} {'n':>5} {'raw conf':>9} {'raw acc':>8} {'cal conf':>9} {'cal acc':>8}")
    for i, (raw_row, cal_row) in enumerate(zip(raw_tab, cal_tab)):
        print(f"{i:>4} {raw_row['n']:>5} {raw_row['confidence']:>9.3f} "
              f"{raw_row['accuracy']:>8.3f} {cal_row['confidence']:>9.3f} "
              f"{cal_row['accuracy']:>8.3f}")

    print()
    print("routing: error rate when acting on the most-confident answers")
    confs = [confidence(best.transform(r.distribution)) for r in records]
    print(f"{'coverage':>9} {'taken':>6} {'threshold':>10} {'errors':>7} {'risk':>7} {'risk(95% upper)':>16}")
    for row in risk_coverage(confs, [r.correct for r in records]):
        print(f"{row['coverage']:>9.0%} {row['taken']:>6} {row['threshold']:>10.3f} "
              f"{row['errors']:>7} {row['risk']:>7.2%} {row['risk_upper95']:>16.2%}")

    if args.out:
        best.to_json(args.out)
        print()
        print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pd", description="Typed decisions from a local MLX model")
    parser.add_argument("--model", default=None,
                        help=f"mlx-lm model id (default: pd.toml, else {DEFAULT_MODEL})")
    parser.add_argument("--config", default=None, help="path to pd.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="check a schema file")
    p_validate.add_argument("schema", help="path to a schema JSON file")
    p_validate.add_argument("--check-tokens", action="store_true",
                            help="load the tokenizer and report choice collisions")
    p_validate.set_defaults(func=cmd_validate)

    p_decide = sub.add_parser("decide", help="run one decision")
    p_decide.add_argument("--schema", required=True, help="path to a schema JSON file")
    p_decide.add_argument("--context", help="context text")
    p_decide.add_argument("--context-file", help="path to a context file")
    p_decide.add_argument("--calibration", help="path to a fitted calibrator JSON")
    p_decide.add_argument("--json", action="store_true", help="print JSON only")
    p_decide.add_argument("--verbose", action="store_true")
    p_decide.set_defaults(func=cmd_decide)

    p_cal = sub.add_parser("calibrate", help="fit a calibrator from labelled decisions")
    p_cal.add_argument("--data", required=True,
                       help="labelled records: .jsonl / .json with distribution (or probability) + correct/target")
    p_cal.add_argument("--out", help="where to write the fitted calibrator JSON")
    p_cal.add_argument("--method", default="auto", choices=("auto", "temperature", "platt", "isotonic"),
                       help="auto compares all methods by cross-validation (default)")
    p_cal.add_argument("--select-by", default="ece_adaptive",
                       choices=("ece", "ece_adaptive", "top_nll", "brier"),
                       help="cross-validation metric used to pick the method")
    p_cal.add_argument("--folds", type=int, default=5)
    p_cal.add_argument("--bins", type=int, default=10)
    p_cal.set_defaults(func=cmd_calibrate)

    p_cfg = sub.add_parser("config", help="show the pd.toml / PD_* settings in effect")
    p_cfg.add_argument("--json", action="store_true")
    p_cfg.set_defaults(func=cmd_config)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SchemaError as exc:
        print(f"schema error: {exc}", file=sys.stderr)
        return 2
    except CalibrationError as exc:
        print(f"calibration error: {exc}", file=sys.stderr)
        return 2
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
