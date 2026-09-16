"""Command line interface: `pd validate` and `pd decide`."""

from __future__ import annotations

import argparse
import json
import sys

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


def cmd_validate(args) -> int:
    schema = Schema.from_json(args.schema)
    print(f"schema OK: {len(schema)} field(s)")
    for f in schema.fields.values():
        if f.is_boolean:
            print(f"  {f.name}: boolean")
        else:
            print(f"  {f.name}: enum, {len(f.choices)} choices")
    if args.check_tokens:
        decider = Decider(model_id=args.model, warmup=False)
        compiled = schema.compile(decider.tokenizer)
        collisions = [c for c in compiled if c.collision]
        if collisions:
            print()
            print(f"token collisions in {len(collisions)} field(s) "
                  f"(these take an extra, slower pass):")
            for c in collisions:
                print(f"  {c.field.name}: {c.field.choices}")
        else:
            print()
            print("no token collisions")
    return 0


def cmd_decide(args) -> int:
    schema = Schema.from_json(args.schema)
    context = _read_context(args)
    decider = Decider(model_id=args.model, verbose=args.verbose)
    result = decider.decide(context, schema)

    if args.json:
        print(json.dumps(result.full_json(), indent=2, default=str))
    else:
        for name, value in result.items():
            print(f"{name:24} {str(value.value):20} {value.probability:6.1%}")
        print()
        print(json.dumps(result.json(), indent=2, default=str))
        print()
        print(f"model={result.model} latency={result.latency_ms:.0f}ms "
              f"(prefill {result.prefill_ms:.0f} + passes {result.pass_ms:.0f}, "
              f"{result.chunks} chunk(s))")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pd", description="Typed decisions from a local MLX model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"mlx-lm model id (default: {DEFAULT_MODEL})")
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
    p_decide.add_argument("--json", action="store_true", help="print JSON only")
    p_decide.add_argument("--verbose", action="store_true")
    p_decide.set_defaults(func=cmd_decide)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SchemaError as exc:
        print(f"schema error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
