"""The Laya engine: typed decisions from a non-autoregressive decision model.

Laya (`convaiinnovations/laya`) is a router over non-autoregressive ModernBERT
decision checkpoints: one forward pass answers every question, in 100+ languages,
with the calibrated max-probability confidence the Laya project fits and gates on.
This module is the package's default model path, mirroring how `SnipLedger.AI`
drives the same package (`snipleger_ai/backends/laya.py`).

How a call works
----------------
1. `plan_schema()` turns each schema field into a Laya question:
   `enum` -> one `choice` question (one criterion per choice), `boolean` -> one
   `noul` question, and `multi` -> one `noul` question per choice at keys like
   `"actions[2]"` (the same per-choice rows the causal engines evaluate).
2. `LayaRuntime` sends the state plus questions through `laya.Router.predict`
   (`predict_batch` for `decide_many`), which routes English text to the English
   checkpoint and everything else to the multilingual one.
3. Answers map back to `FieldValue`s: the chosen choice, or `P(true) >= 0.5` for a
   `noul`. `probability` is Laya's `answer_confidence` (its calibrated `max(p)`),
   `distribution` is the temperature-scaled answer distribution, and `multi`
   fields fold through the causal engines' own `_assemble_multi` so one contract
   covers every backend.

The JSON is never generated; values are assembled from the question types.

Confidence caveat: `answer_confidence` is calibrated on Laya's own benchmark, not
on your domain. `Decider(calibration=...)` can still fit a calibrator on your own
labelled rows on top of it.

Context length: a checkpoint reads one window (512 tokens for `english`, 1024 for
`multilingual`/`typed-decisions`) and truncates longer states. The causal backends
chunk over long contexts; for long documents, extract the relevant text first.

This module imports neither torch nor laya at import time: `import
parallel_decisions` must keep working on machines that never run a model.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .schema import Field, Schema
from .engine import FieldValue
# Safe import order: `engine` imports this module lazily (inside `_select_backend`),
# never at module scope, so both `import parallel_decisions` and a direct
# `import parallel_decisions.engine_laya` fully load `engine` first.

__all__ = [
    "DEFAULT_LAYA_MODEL",
    "LayaDecisionError",
    "LayaPlan",
    "LayaRuntime",
    "LayaUnavailableError",
    "laya_available",
    "plan_schema",
    "values_from_answers",
]

DEFAULT_LAYA_MODEL = "convaiinnovations/laya"

# "auto" family: the bundle routes per request (English text -> english checkpoint,
# other scripts -> multilingual). The bare bundle repo id means auto here, because
# the repo's root subfolder is the English checkpoint but the bundle is what a user
# pointing `model` at "convaiinnovations/laya" expects to get.
_AUTO_NAMES = {"", "auto", "default", "laya", "convaiinnovations/laya"}

_PIN_ALIASES = {
    "english": "english",
    "en": "english",
    "multilingual": "multilingual",
    "multi": "multilingual",
    "ml": "multilingual",
    "laya-multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
    "typed_decisions": "typed-decisions",
    "typed": "typed-decisions",
    "laya-typed-decisions": "typed-decisions",
}

_STANDALONE_REPOS = {
    "convaiinnovations/laya-multilingual": "multilingual",
    "convaiinnovations/laya-typed-decisions": "typed-decisions",
}

# The two checkpoints automatic routing can reach. Preloading them makes routing
# free; `typed-decisions` is never selected automatically (as in `laya.Router`)
# and is loaded only when pinned.
_AUTO_PRELOAD = ("english", "multilingual")


class LayaUnavailableError(RuntimeError):
    """The laya runtime or one of its checkpoints cannot be used."""


class LayaDecisionError(RuntimeError):
    """A planned field has no usable answer in the model's response."""


def laya_available() -> bool:
    """True when the `laya` package can be imported, without importing it."""
    if sys.modules.get("laya") is not None:
        return True                # already imported (or a test stub): usable
    try:
        return importlib.util.find_spec("laya") is not None
    except (ImportError, ValueError):
        return False


def _laya_module():
    """The imported `laya` module, or a clean install hint."""
    module = sys.modules.get("laya")
    if module is not None:
        return module
    try:
        import laya as module   # deliberate lazy import: torch comes with it
    except ImportError as exc:
        raise LayaUnavailableError(
            "the laya backend needs the 'laya' package: "
            "pip install \"parallel-decisions[laya]\"") from exc
    return module


def pinned_model(model_id: str | None) -> str | None:
    """Resolve a model spec to a pinned Laya checkpoint name, or None for auto.

    Accepted specs: the bundle (`convaiinnovations/laya`, `auto`, `default`),
    a checkpoint name or alias (`english`/`en`, `multilingual`/`multi`/`ml`,
    `typed-decisions`/`typed`), or a standalone repo id
    (`convaiinnovations/laya-multilingual`, `convaiinnovations/laya-typed-decisions`).
    """
    key = (model_id or "").strip().lower()
    if key in _AUTO_NAMES:
        return None
    if key in _PIN_ALIASES:
        return _PIN_ALIASES[key]
    if key in _STANDALONE_REPOS:
        return _STANDALONE_REPOS[key]
    raise ValueError(
        f"the laya backend needs a Laya model, got {model_id!r}. Use "
        f"'convaiinnovations/laya' (auto routing), 'english', 'multilingual', "
        f"'typed-decisions', or a standalone convaiinnovations/laya-* repo; "
        f"use backend=\"mlx\"/\"torch\" for causal language models.")


# --------------------------------------------------------------------- planning
@dataclass(frozen=True)
class PlannedField:
    """One Laya question: a field, and (for multi) which choice it stands for."""

    name: str
    qid: str
    kind: str                      # "choice" | "noul"
    field: Field
    choice_index: int | None = None


@dataclass(frozen=True)
class LayaPlan:
    """A schema compiled to Laya questions: one question per field or choice."""

    fields: tuple[PlannedField, ...]
    questions: dict[str, dict]
    by_name: dict[str, Field]

    def __len__(self) -> int:
        return len(self.fields)


def _boolean_question(field: Field) -> dict:
    question: dict[str, Any] = {
        "type": "noul",
        "instructions": field.description.strip() or f"Is `{field.name}` true?",
    }
    criteria = {key: field.choice_descriptions[key]
                for key in ("false", "true")
                if field.choice_descriptions.get(key)}
    if criteria:
        question["criteria"] = criteria
    return question


def _choice_question(field: Field) -> dict:
    criteria = {choice: field.choice_descriptions.get(choice) or None
                for choice in field.choices}
    return {
        "type": "choice",
        "instructions": field.description.strip() or f"What is `{field.name}`?",
        "criteria": criteria,
    }


def _multi_question(field: Field, choice: str) -> dict:
    base = field.description.strip() or f"Selection for the {field.name!r} field"
    return {
        "type": "noul",
        "instructions": f"{base} Does the option {choice!r} belong in the selected subset?",
    }


def plan_schema(schema: Schema) -> LayaPlan:
    """Compile a schema into Laya questions: enum -> choice, boolean -> noul,
    multi -> one noul per choice (keyed `"<name>[<i>]"`)."""
    fields: list[PlannedField] = []
    questions: dict[str, dict] = {}
    owner: dict[str, str] = {}

    def _claim(qid: str, name: str) -> None:
        previous = owner.get(qid)
        if previous is not None:
            raise ValueError(
                f"field {name!r} and field {previous!r} both map to Laya question "
                f"{qid!r}; rename one of them")
        owner[qid] = name

    for f in schema.fields.values():
        if f.is_multi:
            for i, choice in enumerate(f.choices):
                qid = f"{f.name}[{i}]"
                _claim(qid, f.name)
                fields.append(PlannedField(f.name, qid, "noul", f, i))
                questions[qid] = _multi_question(f, choice)
        elif f.is_boolean:
            _claim(f.name, f.name)
            fields.append(PlannedField(f.name, f.name, "noul", f))
            questions[f.name] = _boolean_question(f)
        else:
            _claim(f.name, f.name)
            fields.append(PlannedField(f.name, f.name, "choice", f))
            questions[f.name] = _choice_question(f)
    return LayaPlan(tuple(fields), questions, {f.name: f for f in schema.fields.values()})


# ---------------------------------------------------------------- answer mapping
def _answer_confidence(answer: Mapping[str, Any]) -> float | None:
    """Laya's answer confidence: the calibrated `max(p)`, or the raw fallback."""
    value = answer.get("answer_confidence")
    if value is None:
        value = answer.get("confidence")
    if value is None:
        return None
    return float(value)


def _choice_value(planned: PlannedField, answer: Mapping[str, Any]) -> FieldValue:
    choices = planned.field.choices
    chosen = answer.get("choice")
    if not isinstance(chosen, str) or chosen not in choices:
        raise LayaDecisionError(
            f"field {planned.name!r}: the model chose {chosen!r}, which is not one "
            f"of the field's choices")
    raw = answer.get("probabilities") or {}
    distribution = {c: float(raw[c]) for c in choices if c in raw}
    if not distribution:
        distribution = {chosen: _answer_confidence(answer) or 0.0}
    probability = _answer_confidence(answer)
    if probability is None:
        probability = distribution.get(chosen, 0.0)
    alternatives = sorted(((c, p) for c, p in distribution.items() if c != chosen),
                          key=lambda item: -item[1])[:4]
    return FieldValue(planned.name, chosen, probability, alternatives,
                      distribution=distribution)


def _noul_value(planned: PlannedField, answer: Mapping[str, Any]) -> FieldValue:
    p_true = answer.get("noul")
    if p_true is None:
        raise LayaDecisionError(
            f"field {planned.name!r}: the model returned no noul probability")
    p_true = float(p_true)
    value = p_true >= 0.5
    distribution = {"true": p_true, "false": 1.0 - p_true}
    if planned.choice_index is None:
        # A single boolean reports the chosen side's confidence (Laya's
        # answer_confidence, which is max(noul, 1 - noul) by construction).
        probability = _answer_confidence(answer)
        if probability is None:
            probability = max(p_true, 1.0 - p_true)
    else:
        # A multi row carries P(include) -- that is the contract
        # `_assemble_multi` folds against 0.5, exactly like the causal rows.
        probability = p_true
    other = ("false", 1.0 - p_true) if value else ("true", p_true)
    return FieldValue(planned.name, value, probability, [other],
                      distribution=distribution)


def _row_value(planned: PlannedField, answer: Mapping[str, Any]) -> FieldValue:
    if planned.kind == "choice":
        return _choice_value(planned, answer)
    return _noul_value(planned, answer)


def values_from_answers(plan: LayaPlan, answers: Mapping[str, Any]) -> dict[str, FieldValue]:
    """Map one Laya response onto `FieldValue`s, folding multi rows in order."""
    values: dict[str, FieldValue] = {}
    per_choice: dict[str, dict[int, FieldValue]] = {}
    for planned in plan.fields:
        answer = answers.get(planned.qid)
        if not isinstance(answer, Mapping):
            raise LayaDecisionError(
                f"the model returned no answer for field {planned.name!r} "
                f"(question {planned.qid!r}); every planned field must be answered")
        field_value = _row_value(planned, answer)
        if planned.choice_index is None:
            values[planned.name] = field_value
        else:
            per_choice.setdefault(planned.name, {})[planned.choice_index] = field_value
    for name, by_index in per_choice.items():
        from .engine import Decider   # lazy: engine imports this module too

        values[name] = Decider._assemble_multi(plan.by_name[name], by_index)
    return values


# --------------------------------------------------------------------- runtime
def _outcome(plan: LayaPlan, result: Mapping[str, Any], *, model: str,
             elapsed_ms: float) -> dict[str, Any]:
    usage = result.get("usage") or {}
    routing = result.get("routing") or {}
    return {
        "values": values_from_answers(plan, result.get("answers") or {}),
        "model": model,
        "routed": routing.get("model"),
        "prompt_tokens": int(usage.get("input_tokens") or 0),
        "inference_ms": elapsed_ms,
        "rows": len(plan.fields),
    }


class LayaRuntime:
    """Load `laya.Router` once, then answer schemas with one batched call.

    `model_id` follows `Decider`'s model argument: the bundle means automatic
    routing, a checkpoint name or standalone repo id pins one model (see
    `pinned_model`). `device` is "cuda", "cpu", or None for the package's own
    auto-detection. `preload` loads the checkpoints routing can reach up front;
    with a pinned model, only that one.
    """

    def __init__(self, model_id: str | None = None, *, device: str | None = None,
                 preload: bool = True, verbose: bool = False):
        self.model_id = (model_id or DEFAULT_LAYA_MODEL).strip() or DEFAULT_LAYA_MODEL
        self.pinned = pinned_model(self.model_id)
        self.device = device or None
        self.preload = bool(preload)
        self.verbose = verbose
        self.router: Any = None

    # ------------------------------------------------------------------ load
    def load(self) -> "LayaRuntime":
        if self.router is None:
            router = self._build_router()
            if self.preload:
                for name in self._preload_models():
                    try:
                        router.load(name)
                    except LayaUnavailableError:
                        raise
                    except Exception as exc:  # noqa: BLE001 - add the checkpoint name
                        raise LayaUnavailableError(
                            f"laya checkpoint {name!r} could not be loaded: {exc}") from exc
            self.router = router
        return self

    def _build_router(self):
        # As in SnipLedger.AI: a TF install otherwise makes transformers probe
        # TensorFlow on import, which can hang for minutes.
        os.environ.setdefault("USE_TF", "0")
        laya = _laya_module()
        try:
            return laya.Router(device=self.device, preload=False)
        except AttributeError as exc:
            raise LayaUnavailableError(
                "the installed laya package has no Router; upgrade it with "
                "pip install -U \"laya>=0.3.21\"") from exc
        except TypeError as exc:
            raise LayaUnavailableError(
                "this laya version does not accept Router(device=..., preload=...); "
                "upgrade it with pip install -U \"laya>=0.3.21\"") from exc

    def _preload_models(self) -> list[str]:
        if self.pinned:
            return [self.pinned]
        return list(_AUTO_PRELOAD)

    # ------------------------------------------------------------- inference
    def _route_kwargs(self) -> dict[str, str]:
        return {"model": self.pinned} if self.pinned else {}

    def predict(self, state: str, questions: Mapping[str, Any]) -> dict[str, Any]:
        """Route and answer every question for one state."""
        self.load()
        return self.router.predict(state, questions, **self._route_kwargs())

    def predict_batch(self, states: Sequence[str], questions: Mapping[str, Any],
                      batch_size: int | None = None) -> list[dict[str, Any]]:
        """Answer one question set over many states in one batched call."""
        self.load()
        route = self._route_kwargs()
        requests = [{"state": state, "questions": questions, **route} for state in states]
        return self.router.predict_batch(requests, batch_size=batch_size)

    def decide(self, context: str, schema: Schema) -> dict[str, Any]:
        """One context: returns the engine-level outcome dict."""
        plan = plan_schema(schema)
        started = time.perf_counter()
        result = self.predict(context, plan.questions)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return _outcome(plan, result, model=self.model_id, elapsed_ms=elapsed_ms)

    def decide_many(self, contexts: Iterable[str], schema: Schema,
                    batch_size: int | None = None) -> list[dict[str, Any]]:
        """Many contexts, one `predict_batch`: outcomes in input order.

        Laya shares forward passes across states, so this is the throughput
        form; per-context `latency_ms` is the batch wall time amortized over
        the contexts (`batch_ms` keeps the total), while the causal backends'
        `decide_many` loop reports a real per-call latency.
        """
        contexts = list(contexts)
        plan = plan_schema(schema)
        if not contexts:
            return []
        started = time.perf_counter()
        results = self.predict_batch(contexts, plan.questions, batch_size=batch_size)
        batch_ms = (time.perf_counter() - started) * 1000
        outcomes = []
        for result in results:
            outcome = _outcome(plan, result, model=self.model_id,
                               elapsed_ms=batch_ms / len(contexts))
            outcome["latency_ms"] = outcome["inference_ms"]
            outcome["batch_ms"] = batch_ms
            outcome["batch"] = len(contexts)
            outcomes.append(outcome)
        return outcomes
