"""Tests for the opt-in smoke test's scaffolding (never loads a model).

The smoke test itself needs the real weights, but the pieces around it — the row
count it asserts, the schema it uses, and the calibration check — are pure logic and
are covered here so a broken smoke test cannot sit undetected until someone runs it.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_smoke():
    """Import smoke_test.py without executing its main()."""
    spec = importlib.util.spec_from_file_location("smoke_test_module",
                                                  os.path.join(ROOT, "smoke_test.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["smoke_test_module"] = module
    spec.loader.exec_module(module)
    return module


def test_smoke_schema_is_valid_and_row_count_is_right():
    smoke = load_smoke()
    schema = smoke.SCHEMA
    assert set(schema.fields) == {"is_fraudulent", "risk_tier", "actions", "charge_scope"}
    # 3 single-row fields + one multi field with 3 choices
    assert smoke.expected_rows(schema) == 6
    assert schema["actions"].is_multi
    assert not schema["risk_tier"].is_multi


def test_smoke_schema_includes_a_deliberate_collision():
    """The collision path is the one most likely to break silently, so the smoke
    schema must contain a colliding field."""
    smoke = load_smoke()
    field = smoke.SCHEMA["charge_scope"]
    common = os.path.commonprefix(field.choices)
    assert common == ""                       # no shared prefix
    assert any(c.startswith("extra_") for c in field.choices)
    assert sum(1 for c in field.choices if c.startswith("extra_")) == 2


def test_expected_rows_counts_multi_choices():
    from parallel_decisions import Schema

    schema = Schema({
        "flag": {"type": "boolean", "description": "x"},
        "pick": {"type": "enum", "choices": ["a", "b"], "description": "x"},
        "many": {"type": "multi", "choices": ["a", "b", "c", "d"], "description": "x"},
    })
    smoke = load_smoke()
    assert smoke.expected_rows(schema) == 1 + 1 + 4


def test_expected_rows_on_empty_schema_is_zero():
    from parallel_decisions import Schema

    class _Empty:
        fields = {}

    smoke = load_smoke()
    assert smoke.expected_rows(_Empty()) == 0
    Schema({"a": {"type": "boolean", "description": "x"}})   # still a valid schema
