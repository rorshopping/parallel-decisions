"""Calibration tests: metrics, all three fitting methods, and monotonicity.

No MLX, no model, no network.
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest

from parallel_decisions.calibration import (
    CalibrationError,
    CalibrationRecord,
    Calibrator,
    adaptive_ece,
    auroc,
    confidence,
    ece,
    fit_calibration,
    load_records,
    logit,
    normalize,
    reliability_table,
    risk_coverage,
    sigmoid,
    softmax,
    top_nll,
    wilson_interval,
)


def make_records(n: int = 200, confidence_level: float = 0.95, accuracy: float = 0.75,
                 seed: int = 7):
    """A synthetic overconfident run: constant confidence, known accuracy."""
    records = []
    for i in range(n):
        correct = i < round(accuracy * n)
        dist = {"true": confidence_level, "false": 1.0 - confidence_level}
        records.append(CalibrationRecord(distribution=dist, correct=correct))
    return records


# ------------------------------------------------------------------- maths
def test_logit_sigmoid_roundtrip():
    for p in (0.01, 0.2, 0.5, 0.8, 0.999):
        assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-9)


def test_clip_keeps_logit_finite():
    assert math.isfinite(logit(0.0))
    assert math.isfinite(logit(1.0))


def test_normalize_handles_zero_mass():
    out = normalize({"a": 0.0, "b": 0.0})
    assert out == {"a": 0.5, "b": 0.5}


def test_softmax_sums_to_one():
    probs = softmax([1.0, 2.0, 3.0])
    assert sum(probs) == pytest.approx(1.0)


# ----------------------------------------------------------------- metrics
def test_ece_zero_for_perfectly_calibrated_synthetic_data():
    records = make_records(n=1000, confidence_level=0.75, accuracy=0.75)
    confs = [r.confidence for r in records]
    corrects = [r.correct for r in records]
    assert ece(confs, corrects) < 0.02
    assert adaptive_ece(confs, corrects) < 0.02


def test_ece_positive_for_overconfident_data():
    records = make_records(n=1000, confidence_level=0.95, accuracy=0.75)
    confs = [r.confidence for r in records]
    corrects = [r.correct for r in records]
    assert ece(confs, corrects) == pytest.approx(0.20, abs=0.02)


def test_reliability_table_bins_sum_to_n():
    records = make_records(n=97)
    rows = reliability_table([r.confidence for r in records],
                             [r.correct for r in records], bins=5)
    assert sum(r["n"] for r in rows) == 97


def test_top_nll_rewards_being_right_confidently():
    assert top_nll([0.99, 0.99], [True, True]) < top_nll([0.6, 0.6], [True, True])
    assert top_nll([0.51, 0.51], [False, False]) < top_nll([0.99, 0.99], [False, False])


def test_auroc_is_one_for_perfect_separation():
    confs = [0.9, 0.8, 0.2, 0.1]
    corrects = [True, True, False, False]
    assert auroc(confs, corrects) == pytest.approx(1.0)
    assert auroc([0.5, 0.5, 0.5, 0.5], corrects) == pytest.approx(0.5)


def test_wilson_interval_brackets_the_rate():
    lo, hi = wilson_interval(0, 343)
    assert lo == 0.0
    assert hi < 0.02  # you cannot claim better than ~1% with zero errors in 343 tries
    lo2, hi2 = wilson_interval(5, 10)
    assert lo2 < 0.5 < hi2


def test_risk_coverage_takes_the_most_confident_first():
    confs = [0.9, 0.8, 0.7, 0.6]
    corrects = [True, False, False, False]
    rows = risk_coverage(confs, corrects, coverages=[0.25, 1.0])
    assert rows[0]["taken"] == 1
    assert rows[0]["errors"] == 0
    assert rows[1]["errors"] == 3


# ------------------------------------------------------------- calibrators
def test_temperature_fit_softens_overconfidence():
    records = make_records(n=400, confidence_level=0.95, accuracy=0.75)
    cal = Calibrator.fit(records, method="temperature")
    assert cal.kind == "temperature"
    assert cal.temperature > 1.0                      # overconfident -> soften
    assert cal.transform_confidence(0.95) == pytest.approx(0.75, abs=0.05)
    assert cal.evaluate(records)["ece"] < 0.05


def test_temperature_fit_sharpens_underconfidence():
    records = make_records(n=400, confidence_level=0.6, accuracy=0.9)
    cal = Calibrator.fit(records, method="temperature")
    assert cal.transform_confidence(0.6) > 0.6


def test_temperature_preserves_the_top_choice():
    cal = Calibrator(kind="temperature", temperature=3.0)
    dist = {"a": 0.7, "b": 0.2, "c": 0.1}
    out = cal.transform(dist)
    assert max(out, key=out.__getitem__) == "a"
    assert sum(out.values()) == pytest.approx(1.0)


def test_platt_fit_maps_confidence_to_accuracy():
    records = make_records(n=400, confidence_level=0.9, accuracy=0.7)
    cal = Calibrator.fit(records, method="platt")
    assert cal.kind == "platt"
    assert cal.platt_a > 0
    assert cal.transform_confidence(0.9) == pytest.approx(0.7, abs=0.05)


def test_isotonic_is_monotone_and_bounded():
    records = make_records(n=300, confidence_level=0.85, accuracy=0.6)
    cal = Calibrator.fit(records, method="isotonic")
    xs = sorted({r.confidence for r in records})
    ys = [cal.transform_confidence(x) for x in xs]
    assert all(0.0 <= y <= 1.0 for y in ys)
    assert ys == sorted(ys)
    assert cal.transform_confidence(0.85) == pytest.approx(0.6, abs=0.06)


def test_all_methods_preserve_ranking():
    """Calibration must never reorder answers — only rescale the numbers."""
    records = make_records(n=200)
    for method in ("temperature", "platt", "isotonic"):
        cal = Calibrator.fit(records, method=method)
        confs = [r.confidence for r in records]
        order = sorted(range(len(confs)), key=lambda i: -confs[i])
        mapped = [cal.transform_confidence(c) for c in confs]
        remapped_order = sorted(range(len(confs)), key=lambda i: -mapped[i])
        # allow ties to reorder, but no strict inversion
        for i in range(len(confs) - 1):
            a, b = order[i], order[i + 1]
            assert mapped[a] >= mapped[b] - 1e-12


def test_identity_is_default_and_transparent():
    rec = CalibrationRecord(distribution={"true": 0.9, "false": 0.1}, correct=True)
    cal = Calibrator(kind="identity")
    assert cal.transform(rec.distribution) == rec.distribution


def test_transform_redistributes_remaining_mass():
    """Confidence-map methods move the top and scale the rest in proportion."""
    cal = Calibrator(kind="platt", platt_a=1.0, platt_b=logit(0.5))
    out = cal.transform({"a": 0.8, "b": 0.15, "c": 0.05})
    assert sum(out.values()) == pytest.approx(1.0)
    # ratios among the non-top entries are preserved
    assert out["b"] / out["c"] == pytest.approx(0.15 / 0.05)


def test_temperature_renormalises_the_whole_distribution():
    """Temperature is softmax(z/T) over all classes, not a binary rescale of the top."""
    cal = Calibrator(kind="temperature", temperature=2.0)
    out = cal.transform({"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05})
    assert sum(out.values()) == pytest.approx(1.0)
    assert out["a"] > out["b"] > out["c"] > out["d"]
    assert out["a"] < 0.5  # softened
    # the true softmax(z/2) value, not the binary-view 0.5
    assert out["a"] == pytest.approx(0.379, abs=0.002)


# --------------------------------------------------------------- records
def test_record_from_probability_only_is_binary():
    rec = CalibrationRecord.from_dict({"probability": 0.93, "correct": False})
    assert confidence(rec.distribution) == pytest.approx(0.93)
    assert rec.correct is False


def test_record_from_distribution_and_target():
    rec = CalibrationRecord.from_dict({"distribution": {"a": 0.6, "b": 0.4}, "target": "a"})
    assert rec.correct is True
    assert rec.chosen == "a"


def test_record_requires_label_information():
    with pytest.raises(CalibrationError):
        CalibrationRecord.from_dict({"distribution": {"a": 0.6, "b": 0.4}})


def test_record_rejects_bad_probability():
    with pytest.raises(CalibrationError):
        CalibrationRecord.from_dict({"probability": 1.7, "correct": True})


def test_load_records_jsonl_and_json():
    rows = [{"probability": 0.9, "correct": True}, {"probability": 0.8, "correct": False}]
    with tempfile.TemporaryDirectory() as tmp:
        jl = os.path.join(tmp, "d.jsonl")
        with open(jl, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        assert len(load_records(jl)) == 2
        js = os.path.join(tmp, "d.json")
        with open(js, "w", encoding="utf-8") as fh:
            json.dump({"records": rows}, fh)
        assert len(load_records(js)) == 2


# ------------------------------------------------------- fit_calibration
def test_fit_calibration_selects_a_useful_method():
    records = make_records(n=400, confidence_level=0.95, accuracy=0.75)
    fit = fit_calibration(records, method="auto", folds=5)
    assert fit.calibrator.kind in ("temperature", "platt", "isotonic")
    assert fit.folds_used == 5
    rows = {row["method"]: row for row in fit.table}
    assert rows["identity"]["mean"] > rows[fit.calibrator.kind]["mean"]


def test_fit_calibration_only_ships_methods_that_beat_raw_out_of_sample():
    """In auto mode, raw softmax is the floor: nothing ships unless it wins on folds."""
    records = make_records(n=400, confidence_level=0.95, accuracy=0.75)
    fit = fit_calibration(records, method="auto", folds=5)
    rows = {row["method"]: row for row in fit.table}
    assert set(rows) == {"identity", "temperature", "platt", "isotonic"}
    assert fit.calibrator.kind != "identity"          # this fixture is plainly overconfident
    assert rows[fit.calibrator.kind]["mean"] < rows["identity"]["mean"]
    assert fit.folds_used == 5


def test_fit_calibration_identity_is_an_option_when_it_wins():
    """Exactly calibrated data: the rule allows choosing to do nothing."""
    records = []
    for i in range(500):
        records.append(CalibrationRecord({"true": 0.6, "false": 0.4}, correct=(i % 5 < 3)))
    for i in range(500):
        records.append(CalibrationRecord({"true": 0.9, "false": 0.1}, correct=(i % 10 < 9)))
    fit = fit_calibration(records, method="auto", folds=5)
    rows = {row["method"]: row for row in fit.table}
    if fit.calibrator.kind != "identity":
        assert rows[fit.calibrator.kind]["mean"] < rows["identity"]["mean"]
    # the fixture really is calibrated: raw ECE on the full data is ~0
    assert rows["identity"]["raw"] < 0.001


def test_fit_calibration_single_method_never_returns_identity():
    records = make_records(n=200)
    fit = fit_calibration(records, method="temperature", folds=3)
    assert fit.calibrator.kind == "temperature"


def test_fit_calibration_rejects_unknown_metric():
    with pytest.raises(CalibrationError):
        fit_calibration(make_records(n=20), select_by="nonsense")


def test_fit_calibration_rejects_empty_records():
    with pytest.raises(CalibrationError):
        fit_calibration([])


# --------------------------------------------------------- serialisation
def test_calibrator_roundtrip_json():
    records = make_records(n=200, confidence_level=0.9, accuracy=0.7)
    cal = Calibrator.fit(records, method="platt")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cal.json")
        cal.to_json(path)
        back = Calibrator.from_json(path)
    assert back.kind == cal.kind
    assert back.platt_a == pytest.approx(cal.platt_a)
    for conf in (0.5, 0.7, 0.9, 0.99):
        assert back.transform_confidence(conf) == pytest.approx(cal.transform_confidence(conf))


def test_calibrator_from_json_accepts_wrapped_object():
    cal = Calibrator.from_json(json.dumps({"calibrator": {"kind": "temperature", "temperature": 2.5}}))
    assert cal.kind == "temperature"
    assert cal.temperature == 2.5


def test_calibrator_rejects_unknown_kind():
    with pytest.raises(CalibrationError):
        Calibrator(kind="magic")


def test_calibrator_rejects_nonpositive_temperature():
    with pytest.raises(CalibrationError):
        Calibrator(kind="temperature", temperature=0.0)


def test_multiclass_temperature_keeps_sum_one():
    cal = Calibrator(kind="temperature", temperature=4.0)
    out = cal.transform({"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05})
    assert sum(out.values()) == pytest.approx(1.0)
    assert out["a"] > out["b"] > out["c"] > out["d"]
    assert out["a"] < 0.5  # softened


def test_isotonic_fit_on_ties_does_not_crash():
    records = [CalibrationRecord({"a": 0.7, "b": 0.3}, True) for _ in range(10)]
    cal = Calibrator.fit(records, method="isotonic")
    assert cal.transform_confidence(0.7) == pytest.approx(1.0)
