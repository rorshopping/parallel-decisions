"""Example scripts must stay runnable — checked without a model where possible.

`examples/` is the first thing a stranger runs, so a syntax error or a schema with
collisions in it is a bad first impression. Everything here works without MLX except
the parts explicitly skipped.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLES = os.path.join(ROOT, "examples")


def example_files() -> list[str]:
    return sorted(f for f in os.listdir(EXAMPLES)
                  if f.endswith(".py"))


def test_all_examples_parse():
    for name in example_files():
        path = os.path.join(EXAMPLES, name)
        with open(path, encoding="utf-8") as fh:
            ast.parse(fh.read(), filename=path)


@pytest.mark.parametrize("name", ["routing.py", "serve.py", "mcp_server.py",
                                  "invoice_review.py", "basic.py"])
def test_examples_expose_a_main(name):
    """Each example must be runnable as a script (top-level `main()`)."""
    path = os.path.join(EXAMPLES, name)
    source = open(path, encoding="utf-8").read()
    assert "def main(" in source, f"{name} has no main()"
    assert '__name__ == "__main__"' in source, f"{name} is not runnable"


def load_example(name: str):
    spec = importlib.util.spec_from_file_location(f"example_{name[:-3]}",
                                                  os.path.join(EXAMPLES, name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"example_{name[:-3]}"] = module
    spec.loader.exec_module(module)
    return module


def test_invoice_review_builds_a_collision_free_schema():
    """The per-line questions are generated at call time; they must lint clean."""
    try:
        import mlx_lm  # noqa: F401
    except Exception as exc:  # broken import (no mlx on Windows) counts too
        pytest.skip(f"mlx_lm unavailable: {exc}")
    from parallel_decisions import Decider
    from parallel_decisions.lint import lint_schema
    from parallel_decisions.schema import Schema

    invoice = load_example("invoice_review.py")
    fields, sources = invoice.build_schema(["Managed retainer", "Calibration register"])
    schema = Schema(fields)
    assert len(schema) == len(invoice.DOCUMENT_FIELDS) + 2 * len(invoice.PER_LINE)
    assert len(sources) == len(schema)
    report = lint_schema(schema, Decider(warmup=False).tokenizer_for_schema)
    assert not report.collisions, [(f.name, f.groups) for f in report.collisions]
    assert not report.blocking


def test_invoice_review_routes_by_confidence():
    invoice = load_example("invoice_review.py")
    assert invoice.route(0.99, 0.9, 0.6) == "act"
    assert invoice.route(0.9, 0.9, 0.6) == "act"
    assert invoice.route(0.75, 0.9, 0.6) == "review"
    assert invoice.route(0.6, 0.9, 0.6) == "review"
    assert invoice.route(0.3, 0.9, 0.6) == "refuse"


def test_invoice_review_dump_schema_writes_a_loadable_schema(tmp_path):
    out = tmp_path / "invoice_schema.json"
    proc = subprocess.run(
        [sys.executable, os.path.join(EXAMPLES, "invoice_review.py"),
         "--dump-schema", str(out)],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    from parallel_decisions import Schema

    schema = Schema.from_json(str(out))
    assert len(schema) == 10
    assert "price_basis" in schema.fields
    # it must be usable as a `pd decide` schema without edits
    payload = json.loads(out.read_text())
    assert set(payload) == {"fields"}


def test_serve_health_and_decide_routes_are_declared():
    source = open(os.path.join(EXAMPLES, "serve.py"), encoding="utf-8").read()
    assert '"/health"' in source
    assert '"/decide"' in source
    assert "ConcurrencyError" in source        # queues / 503 instead of crashing


def test_routing_sweep_uses_data_derived_thresholds():
    """The policy must come from the sweep, not from a hardcoded threshold."""
    routing = load_example("routing.py")
    confs = [0.99, 0.95, 0.9, 0.85, 0.7, 0.6, 0.55, 0.5]
    corrects = [True, True, True, False, True, False, False, False]
    rows = routing.sweep(confs, corrects, max_errors=(0.0, 0.05))
    assert rows[0]["budget"] == 0.0
    assert rows[0]["taken"] < len(confs)         # a zero budget cannot take everything
    assert all(row["threshold"] >= 0.0 for row in rows)
    assert all(row["errors"] <= row["taken"] for row in rows)
    threshold, evidence = routing.refuse_threshold(confs, corrects)
    assert 0.0 <= threshold <= 1.0


def test_routing_flags_an_unreachable_budget():
    routing = load_example("routing.py")
    confs = [0.6, 0.55]
    corrects = [False, False]
    rows = routing.sweep(confs, corrects, max_errors=(0.0,))
    assert rows[0].get("unmet") is True      # every answer is wrong: no budget helps
