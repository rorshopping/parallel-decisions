"""The Laya backend, checked without importing laya or loading a model.

`engine_laya` is pure Python at import time and reaches the `laya` package only
through `LayaRuntime._build_router()`, so the whole path is testable with a stub
router. This suite must pass on a machine that has never heard of Laya (and on
one that has it installed with the default backend switched).
"""

from __future__ import annotations

import sys
import types

import pytest

from parallel_decisions import Config, Decider, Schema
from parallel_decisions import engine, engine_laya
from parallel_decisions.engine_laya import (
    DEFAULT_LAYA_MODEL,
    LayaDecisionError,
    LayaUnavailableError,
    plan_schema,
    values_from_answers,
)


# ------------------------------------------------------------------ stubs
def _auto_answers(questions):
    answers = {}
    for qid, question in questions.items():
        if question["type"] == "choice":
            labels = list(question["criteria"])
            rest = 0.4 / (len(labels) - 1)
            answers[qid] = {
                "type": "choice",
                "choice": labels[0],
                "probabilities": {label: (0.6 if i == 0 else rest)
                                  for i, label in enumerate(labels)},
                "confidence": 0.5,
                "answer_confidence": 0.6,
            }
        else:
            answers[qid] = {"type": "noul", "noul": 0.8,
                            "confidence": 0.8, "answer_confidence": 0.8}
    return answers


def _response(answers):
    return {
        "model": "laya-rl-agent",
        "answers": answers,
        "usage": {"input_tokens": 42, "output_tokens": 0},
        "routing": {"model": "english"},
    }


class FakeRouter:
    """Deterministic stand-in for `laya.Router`, recording every call."""

    def __init__(self, answers: dict | None = None, routed: str = "english"):
        self.answers = answers
        self.routed = routed
        self.calls: list[dict] = []
        self.batches: list[dict] = []
        self.loaded: list[str] = []

    def load(self, name):
        self.loaded.append(name)
        return self

    def _answer(self, questions):
        if self.answers is not None:
            return {qid: self.answers[qid] for qid in questions}
        return _auto_answers(questions)

    def predict(self, state, questions, model=None, **kwargs):
        self.calls.append({"state": state, "questions": questions, "model": model})
        response = _response(self._answer(questions))
        response["routing"] = {"model": self.routed}
        return response

    def predict_batch(self, requests, batch_size=None, **kwargs):
        self.batches.append({"requests": list(requests), "batch_size": batch_size})
        out = []
        for request in requests:
            response = _response(self._answer(request["questions"]))
            response["routing"] = {"model": self.routed}
            out.append(response)
        return out


@pytest.fixture
def fake_router(monkeypatch):
    router = FakeRouter()
    monkeypatch.setattr(engine_laya.LayaRuntime, "_build_router", lambda self: router)
    return router


# ------------------------------------------------------------------ planning
def test_plan_schema_maps_each_field_type_to_a_laya_question():
    schema = Schema({
        "category": {
            "type": "enum",
            "choices": {"billing": "the money one", "technical": "the code one"},
            "description": "Primary category",
        },
        "urgent": {
            "type": "boolean",
            "description": "Is this urgent",
            "choices": {"true": "act now", "false": "it can wait"},
        },
        "actions": {"type": "multi", "choices": ["retry", "escalate"],
                    "description": "Actions that apply"},
    })
    plan = plan_schema(schema)

    assert plan.questions["category"] == {
        "type": "choice",
        "instructions": "Primary category",
        "criteria": {"billing": "the money one", "technical": "the code one"},
    }
    assert plan.questions["urgent"] == {
        "type": "noul",
        "instructions": "Is this urgent",
        "criteria": {"true": "act now", "false": "it can wait"},
    }
    assert plan.questions["actions[0]"]["type"] == "noul"
    assert "'retry'" in plan.questions["actions[0]"]["instructions"]
    assert "Actions that apply" in plan.questions["actions[0]"]["instructions"]
    assert "'escalate'" in plan.questions["actions[1]"]["instructions"]
    assert set(plan.questions) == {"category", "urgent", "actions[0]", "actions[1]"}
    assert len(plan) == 4


def test_plan_schema_falls_back_to_question_shaped_instructions():
    plan = plan_schema(Schema({"flag": {"type": "boolean"},
                               "kind": {"type": "enum", "choices": ["a", "b"]}}))
    assert plan.questions["flag"] == {"type": "noul", "instructions": "Is `flag` true?"}
    assert plan.questions["kind"] == {
        "type": "choice",
        "instructions": "What is `kind`?",
        "criteria": {"a": None, "b": None},
    }


def test_plan_schema_rejects_colliding_question_ids():
    schema = Schema({
        "actions": {"type": "multi", "choices": ["retry", "escalate"]},
        "actions[0]": {"type": "boolean"},
    })
    with pytest.raises(ValueError, match="both map to Laya question"):
        plan_schema(schema)


# ------------------------------------------------------------- answer mapping
def test_choice_answer_maps_value_probability_and_distribution():
    plan = plan_schema(Schema({
        "category": {"type": "enum", "choices": ["billing", "technical", "account"]},
    }))
    answers = {"category": {
        "type": "choice",
        "choice": "technical",
        "probabilities": {"billing": 0.1, "technical": 0.7, "account": 0.2},
        "confidence": 0.35,           # entropy confidence: not the calibrated one
        "answer_confidence": 0.7,
    }}
    field_value = values_from_answers(plan, answers)["category"]
    assert field_value.value == "technical"
    assert field_value.probability == 0.7
    assert field_value.raw_probability == 0.7
    assert field_value.calibrated is False
    assert field_value.distribution == {"billing": 0.1, "technical": 0.7, "account": 0.2}
    assert field_value.alternatives == [("account", 0.2), ("billing", 0.1)]


def test_boolean_answer_uses_the_calibrated_confidence():
    plan = plan_schema(Schema({"urgent": {"type": "boolean"}}))
    true_value = values_from_answers(plan, {"urgent": {
        "type": "noul", "noul": 0.73, "confidence": 0.73, "answer_confidence": 0.62,
    }})["urgent"]
    assert true_value.value is True
    assert true_value.probability == 0.62
    assert true_value.distribution == {"true": 0.73, "false": 0.27}

    false_value = values_from_answers(plan, {"urgent": {
        "type": "noul", "noul": 0.3, "confidence": 0.7, "answer_confidence": 0.65,
    }})["urgent"]
    assert false_value.value is False
    assert false_value.probability == 0.65
    assert false_value.distribution == {"true": 0.3, "false": 0.7}


def test_multi_folds_like_the_causal_engine():
    plan = plan_schema(Schema({
        "actions": {"type": "multi", "choices": ["retry", "escalate", "refund"]},
    }))
    answers = {
        "actions[0]": {"type": "noul", "noul": 0.9, "answer_confidence": 0.9},
        "actions[1]": {"type": "noul", "noul": 0.4, "answer_confidence": 0.6},
        "actions[2]": {"type": "noul", "noul": 0.6, "answer_confidence": 0.6},
    }
    field_value = values_from_answers(plan, answers)["actions"]
    assert field_value.value == ["retry", "refund"]
    assert field_value.probability == 0.6          # the weakest included yes
    assert field_value.distribution == {"retry": 0.9, "escalate": 0.4, "refund": 0.6}


def test_multi_with_nothing_selected_reports_confidence_in_that():
    plan = plan_schema(Schema({"actions": {"type": "multi", "choices": ["retry", "refund"]}}))
    answers = {
        "actions[0]": {"type": "noul", "noul": 0.2, "answer_confidence": 0.8},
        "actions[1]": {"type": "noul", "noul": 0.1, "answer_confidence": 0.9},
    }
    field_value = values_from_answers(plan, answers)["actions"]
    assert field_value.value == []
    assert field_value.probability == pytest.approx(0.8)   # 1 - max(P(true))


def test_missing_or_unknown_answers_raise_a_named_error():
    plan = plan_schema(Schema({
        "urgent": {"type": "boolean"},
        "category": {"type": "enum", "choices": ["a", "b"]},
    }))
    with pytest.raises(LayaDecisionError, match="urgent"):
        values_from_answers(plan, {"category": {
            "type": "choice", "choice": "a", "probabilities": {"a": 1.0, "b": 0.0},
        }})
    with pytest.raises(LayaDecisionError, match="category"):
        values_from_answers(plan, {
            "urgent": {"type": "noul", "noul": 0.1},
            "category": {"type": "choice", "choice": "z",
                         "probabilities": {"a": 1.0, "b": 0.0}},
        })
    with pytest.raises(LayaDecisionError, match="urgent"):
        values_from_answers(plan, {
            "urgent": {"type": "noul"},
            "category": {"type": "choice", "choice": "a",
                         "probabilities": {"a": 1.0, "b": 0.0}},
        })


# --------------------------------------------------------------- backend select
def test_laya_available_detects_an_injected_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace())
    assert engine_laya.laya_available() is True


def test_laya_available_is_false_when_it_is_not_installed(monkeypatch):
    monkeypatch.delitem(sys.modules, "laya", raising=False)
    monkeypatch.setattr(engine_laya.importlib.util, "find_spec", lambda name: None)
    assert engine_laya.laya_available() is False


def test_auto_prefers_laya_when_available(monkeypatch):
    monkeypatch.setattr(engine_laya, "laya_available", lambda: True)
    assert engine._select_backend("auto") == "laya"
    assert engine._select_backend(None) == "laya"


def test_auto_falls_back_to_the_causal_backends_without_laya(monkeypatch):
    monkeypatch.setattr(engine_laya, "laya_available", lambda: False)
    monkeypatch.setattr(engine.sys, "platform", "win32")
    monkeypatch.setattr(engine.platform, "machine", lambda: "x86_64")
    assert engine._select_backend("auto") == "torch"
    monkeypatch.setattr(engine.sys, "platform", "darwin")
    monkeypatch.setattr(engine.platform, "machine", lambda: "arm64")
    assert engine._select_backend("auto") == "mlx"


@pytest.mark.parametrize("model_id,expected", [
    (None, None),
    ("convaiinnovations/laya", None),
    ("auto", None),
    ("english", "english"),
    ("EN", "english"),
    ("multi", "multilingual"),
    ("laya-multilingual", "multilingual"),
    ("typed_decisions", "typed-decisions"),
    ("laya-typed-decisions", "typed-decisions"),
    ("convaiinnovations/laya-multilingual", "multilingual"),
    ("convaiinnovations/laya-typed-decisions", "typed-decisions"),
])
def test_pinned_model_specs(model_id, expected):
    assert engine_laya.pinned_model(model_id) == expected


def test_a_causal_model_is_rejected_with_an_actionable_error():
    decider = Decider("Qwen/Qwen2.5-0.5B-Instruct", backend="laya",
                      config=Config(), warmup=False)
    with pytest.raises(ValueError, match="Laya model"):
        decider.load()


def test_missing_laya_package_gives_an_install_hint(monkeypatch):
    def missing():
        raise LayaUnavailableError(
            "the laya backend needs the 'laya' package: "
            "pip install \"parallel-decisions[laya]\"")

    monkeypatch.setattr(engine_laya, "_laya_module", missing)
    decider = Decider(backend="laya", config=Config(), warmup=False)
    with pytest.raises(LayaUnavailableError, match="pip install"):
        decider.load()


# ------------------------------------------------------------------- decider
def test_decider_laya_end_to_end_with_a_stub_router(fake_router):
    decider = Decider(backend="laya", config=Config(), warmup=False)
    assert decider.model_id == DEFAULT_LAYA_MODEL
    assert decider.backend == "laya"

    result = decider.decide("the server is down", Schema({
        "urgent": {"type": "boolean", "description": "Is this urgent"},
    }))
    assert result["urgent"].value is True
    assert result["urgent"].probability == 0.8
    assert result.model == DEFAULT_LAYA_MODEL
    assert result.chunks == 1
    assert result.telemetry["backend"] == "laya"
    assert result.telemetry["routed"] == "english"
    assert result.telemetry["prompt_tokens"] == 42

    assert len(fake_router.calls) == 1
    call = fake_router.calls[0]
    assert call["state"] == "the server is down"
    assert set(call["questions"]) == {"urgent"}
    assert call["model"] is None              # unpinned: routing decides
    assert fake_router.loaded == []           # warmup=False: no preload


def test_warmup_preloads_the_reachable_checkpoints(fake_router):
    decider = Decider(backend="laya", config=Config())   # warmup defaults to true
    decider.load()
    assert fake_router.loaded == ["english", "multilingual"]


def test_a_pinned_model_is_preloaded_and_passed_through(monkeypatch):
    router = FakeRouter(routed="multilingual")
    monkeypatch.setattr(engine_laya.LayaRuntime, "_build_router", lambda self: router)
    decider = Decider("multilingual", backend="laya", config=Config())
    result = decider.decide("Überweisung freigeben", Schema({"ok": {"type": "boolean"}}))
    assert decider.model_id == "multilingual"
    assert router.loaded == ["multilingual"]
    assert router.calls[0]["model"] == "multilingual"
    assert result.telemetry["routed"] == "multilingual"


def test_decider_laya_batches_decide_many(fake_router):
    decider = Decider(backend="laya", config=Config(), warmup=False)
    schema = Schema({"urgent": {"type": "boolean"}})
    results = decider.decide_many(["one", "two", "three"], schema)
    assert [r["urgent"].value for r in results] == [True, True, True]
    assert len(fake_router.batches) == 1
    assert len(fake_router.batches[0]["requests"]) == 3
    assert fake_router.batches[0]["batch_size"] == 32    # max_fields_per_batch default
    assert fake_router.calls == []
    assert results[0].telemetry["batch"] == 3
    assert results[0].telemetry["batch_ms"] >= results[0].latency_ms
    # empty input needs no model call
    assert decider.decide_many([], schema) == []
    assert len(fake_router.batches) == 1
    # every context is validated before any model call, as in the loop path
    with pytest.raises(ValueError, match="non-empty"):
        decider.decide_many(["ok", "  "], schema)
    assert len(fake_router.batches) == 1


def test_decider_laya_shared_prefix_is_a_noop(fake_router):
    decider = Decider(backend="laya", config=Config(), warmup=False)
    schema = Schema({"urgent": {"type": "boolean"}})
    prefix = decider.prepare(schema)
    assert not prefix.reusable
    result = decider.decide_with_prefix(prefix, "one")
    assert result["urgent"].value is True
    prefix.release()


def test_laya_prefix_release_does_not_touch_mlx(fake_router, monkeypatch):
    touched = []
    monkeypatch.setattr(engine, "_clear_mlx_cache", lambda: touched.append(True))
    decider = Decider(backend="laya", config=Config(), warmup=False)
    prefix = decider.prepare(Schema({"urgent": {"type": "boolean"}}))
    prefix.release()
    assert touched == []


def test_decider_laya_applies_a_calibrator_on_top(fake_router):
    from parallel_decisions import Calibrator

    fake_router.answers = {"urgent": {"type": "noul", "noul": 0.8,
                                      "confidence": 0.8, "answer_confidence": 0.8}}
    calibrator = Calibrator(kind="temperature", temperature=4.0)
    decider = Decider(backend="laya", config=Config(), warmup=False,
                      calibration=calibrator)
    result = decider.decide("x", Schema({"urgent": {"type": "boolean"}}))
    field_value = result["urgent"]
    assert field_value.value is True                  # calibration never reorders
    assert field_value.calibrated is True
    assert field_value.raw_probability == 0.8
    assert 0.5 < field_value.probability < 0.8        # softened toward the prior
    assert result.calibrated is True


def test_decider_laya_logs_json_telemetry(fake_router, capsys):
    decider = Decider(backend="laya", config=Config(log="json"), warmup=False)
    decider.decide("x", Schema({"urgent": {"type": "boolean"}}))
    import json

    payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert payload["event"] == "decide"
    assert payload["backend"] == "laya"
    assert payload["rows"] == 1
    assert payload["prompt_tokens"] == 42
