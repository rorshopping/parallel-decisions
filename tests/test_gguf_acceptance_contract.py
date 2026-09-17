"""Harness-only tests: no binding import, native calls, downloads or model access."""
from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from parallel_decisions import Calibrator, Schema
from parallel_decisions.engine import DecisionResult, FieldValue

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_gguf_native.py"
spec = importlib.util.spec_from_file_location("gguf_harness", SCRIPT)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class TinyTokenizer:
    """Character vocabulary with one shared first token, not a GGUF substitute."""
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class FakeDecider:
    """Only the documented facade surface consumed by run_suite."""
    tokenizer = TinyTokenizer()
    max_fields_per_batch = 8
    calibrator = None

    def __init__(self):
        self.calls = []

    def decide(self, context, schema):
        self.calls.append((context, tuple(schema.fields), self.max_fields_per_batch))
        base = 0.8 if context == harness.CONTEXTS["a"] else 0.3
        values = {}
        for name, field in schema.fields.items():
            if field.is_multi:
                raw_dist = dict(zip(field.choices, [base, 0.6, 0.2]))
            else:
                rest = (1 - base) / (len(field.answers) - 1)
                raw_dist = dict(zip(field.answers, [base] + [rest] * (len(field.answers) - 1)))
            dist = dict(raw_dist)
            if self.calibrator:
                dist = ({k: self.calibrator.transform_confidence(p) for k, p in dist.items()}
                        if field.is_multi else self.calibrator.transform(dist))
            if field.is_multi:
                value = [k for k, p in dist.items() if p >= 0.5]
                p = min(dist[k] for k in value) if value else 1 - max(dist.values())
                raw = min(raw_dist[k] for k in value) if value else 1 - max(raw_dist.values())
            else:
                best = max(dist, key=dist.get)
                value = best == "true" if field.is_boolean else best
                p, raw = dist[best], raw_dist[best]
            values[name] = FieldValue(name=name, value=value, probability=p,
                                      raw_probability=raw, distribution=dist, alternatives=[],
                                      calibrated=self.calibrator is not None)
        rows = len(schema.compile(self.tokenizer))
        result = DecisionResult(values, fields_evaluated=rows,
                                chunks=harness.math.ceil(rows / self.max_fields_per_batch),
                                latency_ms=4.0, prefill_ms=2.0, pass_ms=1.0,
                                calibrated=self.calibrator is not None)
        result.telemetry = {"backend": "llamacpp"}
        return result


def fixture_result():
    decider = FakeDecider()
    schema, compiled = harness.select_schema(Schema, decider.tokenizer)
    return decider.decide(harness.CONTEXTS["a"], schema), schema, len(compiled)


def test_harness_help_needs_no_site_packages():
    completed = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"],
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0
    assert "--exclusive-model-use" in completed.stdout
    assert "NOT proof of parallelism" in " ".join(completed.stdout.split())


@pytest.mark.parametrize("args", [[], ["--run-native"], ["--exclusive-model-use"]])
def test_harness_inference_is_double_opt_in(args):
    completed = subprocess.run([sys.executable, "-S", str(SCRIPT), *args],
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 2
    assert "inference disabled" in completed.stderr
    assert "ModuleNotFoundError" not in completed.stderr


def test_harness_tiny_vocab_collision_is_exact_after_prefix_removal():
    schema, rows = harness.select_schema(Schema, TinyTokenizer())
    assert len(schema) == 4
    assert len(rows) == 6
    assert rows[-1].collision
    assert rows[-1].candidate_ids[0] == rows[-1].candidate_ids[1]
    assert rows[-1].sequences[0] != rows[-1].sequences[1]
    assert rows[2].row_name == "tags[0]"
    assert len({len(row.suffix_tokens) for row in rows}) > 1
    # Without the third choice, Schema extracts the shared prefix first.
    pair = Schema({"pair": {"type": "enum", "choices": ["parcel red", "parcel blue"]}})
    assert not pair.compile(TinyTokenizer())[0].collision


def test_harness_missing_collision_is_failure_not_skip():
    class NoCollisionTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [sum((i + 1) * ord(c) for i, c in enumerate(text))]
    with pytest.raises(harness.ContractError, match="No exact"):
        harness.select_schema(Schema, NoCollisionTokenizer())


def test_harness_valid_multi_marginals_need_not_sum_to_one():
    result, schema, rows = fixture_result()
    snapshot = harness.validate_result(result, schema, rows)
    assert sum(snapshot["fields"]["tags"]["distribution"].values()) == pytest.approx(1.6)
    result["tags"].value.append("heavy")
    assert snapshot["fields"]["tags"]["value"] == ["dry", "light"]


@pytest.mark.parametrize("mutation", [
    lambda r: setattr(r["fragile"], "value", "true"),
    lambda r: setattr(r["color"], "value", "free generated prose"),
    lambda r: setattr(r["tags"], "value", ["dry", "dry"]),
    lambda r: r["color"].distribution.update(RED=float("nan")),
    lambda r: r["color"].distribution.update(RED=-0.1),
    lambda r: r["color"].distribution.update(unlisted=0.1),
    lambda r: r["color"].distribution.update(RED=0.7),
    lambda r: setattr(r["long_named_collision_field"], "approximate", True),
    lambda r: setattr(r, "fields_evaluated", 4),
    lambda r: setattr(r, "prefill_ms", float("inf")),
    lambda r: r.telemetry.update(backend="torch"),
    lambda r: setattr(r["color"], "raw_probability", 0.1),
])
def test_harness_rejects_malformed_result(mutation):
    result, schema, rows = fixture_result()
    mutation(result)
    with pytest.raises(harness.ContractError):
        harness.validate_result(result, schema, rows)


def test_harness_comparison_checks_values_and_every_probability():
    result, schema, rows = fixture_result()
    left = harness.validate_result(result, schema, rows)
    right = copy.deepcopy(left)
    right["fields"]["color"]["distribution"]["GREEN"] += 0.01
    assert not harness.compare(left, right, 1e-4)["passed"]
    right = copy.deepcopy(left)
    right["fields"]["color"]["value"] = "BLUE"
    assert not harness.compare(left, right, 1e-4)["passed"]
    assert harness.compare(left, left, 0)["passed"]


def test_harness_complete_fake_suite_preserves_full_schema_in_row_comparisons():
    decider = FakeDecider()
    report = {}
    harness.run_suite(decider, Schema, Calibrator, repeats=2, atol=1e-4, report=report)
    assert len(report["runs"]) == 24
    assert {r["row_limit"] for r in report["runs"]} == {1, 2, 8}
    assert len({keys for _, keys, _ in decider.calls[:24]}) == 1
    assert [limit for _, _, limit in decider.calls[::4]][:6] == [8, 2, 1, 1, 2, 8]
    assert all(c["passed"] for c in report["comparisons"])
    assert report["context_sensitivity_observed"]
    assert report["singleton_schema"]["fields_evaluated"] == 1
    assert report["calibration"]["fields"]["fragile"]["raw_probability"] == 0.8


def test_harness_serial_counter_claim_does_not_make_native_proof():
    # A serial fake can pass every functional check. There is deliberately no
    # 'native proof passed' result from run_suite: that needs external evidence.
    report = {"native_parallelism": "NOT_VERIFIED"}
    harness.run_suite(FakeDecider(), Schema, Calibrator, repeats=1, atol=1e-4, report=report)
    assert report["native_parallelism"] == "NOT_VERIFIED"


def test_harness_catches_ignored_row_limit():
    class IgnoresLimit(FakeDecider):
        def decide(self, context, schema):
            result = super().decide(context, schema)
            result.chunks = 1
            return result
    with pytest.raises(harness.ContractError, match="chunk limit"):
        harness.run_suite(IgnoresLimit(), Schema, Calibrator, repeats=1, atol=1e-4, report={})


def test_harness_sha256_reads_only_explicit_file(tmp_path):
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"abc")
    assert harness.sha256(model) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
