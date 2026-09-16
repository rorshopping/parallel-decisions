"""The decision engine: one prefill, batched forward passes, typed results.

How a call works
----------------
1. Build a prompt from the schema descriptions + the context, and prefill it once
   into an MLX KV cache.
2. For non-colliding fields: broadcast the cache once per field and run a single
   batched forward pass over each field's literal suffix (`  "name": `). The logits
   at each row's decision position are sliced down to that field's allowed answers
   and softmaxed.
3. For colliding fields (two answers sharing a first token): one extra batched pass
   over the full answer token sequences gives exact sequence log-probabilities.

The JSON is assembled from the selected values; it is never generated token by token,
so keys and types cannot be malformed.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import mlx.core as mx

from .prompts import build_prompt
from .schema import CompiledField, Schema

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_MEMORY_BUDGET_GB = 6.0
DEFAULT_MAX_FIELDS = 32
DEFAULT_MAX_COLLISION_ROWS = 8


@dataclass
class FieldValue:
    """One typed decision."""

    name: str
    value: Any
    probability: float
    alternatives: list[tuple[str, float]]
    approximate: bool = False

    def to_dict(self, probabilities: bool = True) -> dict:
        out: dict[str, Any] = {"value": self.value}
        if probabilities:
            out["probability"] = round(self.probability, 4)
        return out

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"{self.name}={self.value!r} ({self.probability:.1%})"


class DecisionResult(dict):
    """Maps field name -> FieldValue, plus call telemetry."""

    def __init__(self, mapping: Mapping[str, FieldValue], *, model: str = "",
                 latency_ms: float = 0.0, prefill_ms: float = 0.0, pass_ms: float = 0.0,
                 fields_evaluated: int = 0, chunks: int = 0):
        super().__init__(mapping)
        self.model = model
        self.latency_ms = latency_ms
        self.prefill_ms = prefill_ms
        self.pass_ms = pass_ms
        self.fields_evaluated = fields_evaluated
        self.chunks = chunks

    # -- output helpers ------------------------------------------------------
    def json(self) -> dict[str, Any]:
        """Plain {field: value} mapping."""
        return {k: v.value for k, v in self.items()}

    def full_json(self) -> dict[str, Any]:
        """{field: {value, probability}} mapping."""
        return {k: v.to_dict() for k, v in self.items()}

    def __repr__(self) -> str:  # pragma: no cover - convenience
        inner = ", ".join(f"{k}={v.value!r}" for k, v in self.items())
        return f"DecisionResult({inner})"


class Decider:
    """Loads a local MLX model once and answers schemas against contexts."""

    def __init__(self, model_id: str | None = None, *,
                 max_fields_per_batch: int = DEFAULT_MAX_FIELDS,
                 memory_budget_gb: float = DEFAULT_MEMORY_BUDGET_GB,
                 max_collision_rows: int = DEFAULT_MAX_COLLISION_ROWS,
                 warmup: bool = True,
                 verbose: bool = False):
        self.model_id = model_id or DEFAULT_MODEL
        self.max_fields_per_batch = max(1, int(max_fields_per_batch))
        self.memory_budget_bytes = int(memory_budget_gb * (1024 ** 3))
        self.max_collision_rows = max(1, int(max_collision_rows))
        self.warmup = warmup
        self.verbose = verbose
        self._model = None
        self._tokenizer = None
        self._kv_bytes_per_token: int | None = None

    # ------------------------------------------------------------------ load
    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[parallel-decisions] {message}", flush=True)

    def load(self) -> "Decider":
        if self._model is None:
            from mlx_lm import load
            t0 = time.perf_counter()
            self._log(f"loading {self.model_id} ...")
            self._model, self._tokenizer = load(self.model_id)
            self._kv_bytes_per_token = _estimate_kv_bytes_per_token(self._model)
            self._log(f"loaded in {time.perf_counter() - t0:.1f}s "
                      f"(kv/token ~{self._kv_bytes_per_token or 0} bytes)")
            if self.warmup:
                self._warmup()
        return self

    @property
    def model(self):
        return self.load()._model

    @property
    def tokenizer(self):
        return self.load()._tokenizer

    def _warmup(self) -> None:
        """Compile the Metal shaders with a tiny prefill + broadcast."""
        try:
            from mlx_lm.models.cache import make_prompt_cache
            toks = self._tokenizer.encode("warmup context")
            cache = make_prompt_cache(self._model)
            self._model(mx.array(toks)[None], cache=cache)
            b_cache = self._repeat_cache(cache, 4)
            self._model(mx.zeros((4, 3), dtype=mx.int32), cache=b_cache)
            mx.eval(*[c.keys for c in b_cache if getattr(c, "keys", None) is not None])
        except Exception as exc:  # pragma: no cover - warmup is best-effort
            self._log(f"warmup skipped: {exc}")

    # ------------------------------------------------------------------ cache
    @staticmethod
    def _repeat_cache(cache, n: int):
        out = []
        for layer_cache in cache:
            new_cache = copy.copy(layer_cache)
            keys = getattr(layer_cache, "keys", None)
            if keys is not None:
                new_cache.keys = mx.repeat(keys, n, axis=0)
                new_cache.values = mx.repeat(layer_cache.values, n, axis=0)
            out.append(new_cache)
        return out

    @property
    def _pad_id(self) -> int:
        pad = getattr(self._tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self._tokenizer, "eos_token_id", 0)
        return int(pad or 0)

    # ------------------------------------------------------------------ API
    def decide(self, context: str, schema: Schema | Mapping[str, Any], *,
               temperature: float = 1.0) -> DecisionResult:
        if not isinstance(context, str) or not context.strip():
            raise ValueError("context must be a non-empty string")
        if not isinstance(schema, Schema):
            schema = Schema(schema)
        self.load()

        t_start = time.perf_counter()
        compiled = schema.compile(self._tokenizer)
        prompt = build_prompt(context, schema)
        prompt_tokens = self._tokenizer.encode(prompt)

        from mlx_lm.models.cache import make_prompt_cache
        prompt_cache = make_prompt_cache(self._model)
        t_pre = time.perf_counter()
        self._model(mx.array(prompt_tokens)[None], cache=prompt_cache)
        mx.eval(*[c.keys for c in prompt_cache if getattr(c, "keys", None) is not None])
        prefill_ms = (time.perf_counter() - t_pre) * 1000
        self._log(f"prefilled {len(prompt_tokens)} tokens in {prefill_ms:.0f} ms")

        chunk_size = self._auto_chunk_size(len(prompt_tokens))
        values: dict[str, FieldValue] = {}
        pass_ms = 0.0
        chunks = 0
        for start in range(0, len(compiled), chunk_size):
            group = compiled[start:start + chunk_size]
            chunk_values, chunk_ms = self._run_chunk(prompt_cache, group, temperature)
            values.update(chunk_values)
            pass_ms += chunk_ms
            chunks += 1

        return DecisionResult(
            values,
            model=self.model_id,
            latency_ms=(time.perf_counter() - t_start) * 1000,
            prefill_ms=prefill_ms,
            pass_ms=pass_ms,
            fields_evaluated=len(compiled),
            chunks=chunks,
        )

    def decide_many(self, contexts: Iterable[str], schema: Schema | Mapping[str, Any],
                    *, temperature: float = 1.0) -> list[DecisionResult]:
        return [self.decide(ctx, schema, temperature=temperature) for ctx in contexts]

    # ------------------------------------------------------------- internals
    def _auto_chunk_size(self, prompt_tokens: int) -> int:
        if not self._kv_bytes_per_token or prompt_tokens <= 0:
            return self.max_fields_per_batch
        per_row = self._kv_bytes_per_token * prompt_tokens
        if per_row <= 0:
            return self.max_fields_per_batch
        budget_rows = max(1, int(self.memory_budget_bytes // per_row))
        return max(1, min(self.max_fields_per_batch, budget_rows))

    def _run_chunk(self, prompt_cache, fields: Sequence[CompiledField],
                   temperature: float) -> tuple[dict[str, FieldValue], float]:
        max_len = max(len(f.suffix_tokens) for f in fields)
        pad = self._pad_id
        rows = [f.suffix_tokens + [pad] * (max_len - len(f.suffix_tokens)) for f in fields]
        arr = mx.array(rows, dtype=mx.int32)

        batch_cache = self._repeat_cache(prompt_cache, len(fields))
        t0 = time.perf_counter()
        logits = self._model(arr, cache=batch_cache)
        mx.eval(logits)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        values: dict[str, FieldValue] = {}
        collisions: list[CompiledField] = []
        for row, cf in enumerate(fields):
            row_logits = logits[row, len(cf.suffix_tokens) - 1, :]
            if cf.collision:
                collisions.append(cf)
            else:
                values[cf.field.name] = self._slice_decision(cf, row_logits, temperature)
        if collisions:
            values.update(self._resolve_collisions(prompt_cache, collisions))
        return values, elapsed_ms

    def _slice_decision(self, cf: CompiledField, row_logits, temperature: float) -> FieldValue:
        scores = []
        for ids in cf.candidate_ids:
            if not ids:
                scores.append(-1e9)
            else:
                scores.append(max(float(row_logits[i]) for i in ids))
        scaled = mx.array(scores) / max(temperature, 1e-4)
        probs = mx.softmax(scaled)
        mx.eval(probs)
        plist = [float(p) for p in probs]
        best = max(range(len(plist)), key=plist.__getitem__)
        answers = cf.field.answers
        value = (answers[best] == "true") if cf.field.is_boolean else answers[best]
        alternatives = sorted(
            ((answers[i], plist[i]) for i in range(len(answers)) if i != best),
            key=lambda item: -item[1],
        )
        return FieldValue(cf.field.name, value, plist[best], alternatives[:4])

    def _resolve_collisions(self, prompt_cache, fields: Sequence[CompiledField]) -> dict[str, FieldValue]:
        """Exact sequence scoring for fields whose answers share a first token."""
        rows: list[list[int]] = []
        spans: list[tuple[str, int, int, int]] = []  # (field, choice idx, start, end)
        for cf in fields:
            suffix_len = len(cf.suffix_tokens)
            for ci, seq in enumerate(cf.sequences):
                rows.append(cf.suffix_tokens + seq)
                spans.append((cf.field.name, ci, suffix_len, suffix_len + len(seq)))

        log_probs: dict[str, dict[int, float]] = {}
        pad = self._pad_id
        cursor = 0
        while cursor < len(rows):
            batch_rows = rows[cursor:cursor + self.max_collision_rows]
            batch_spans = spans[cursor:cursor + self.max_collision_rows]
            max_len = max(len(r) for r in batch_rows)
            arr = mx.array([r + [pad] * (max_len - len(r)) for r in batch_rows], dtype=mx.int32)
            batch_cache = self._repeat_cache(prompt_cache, len(batch_rows))
            out = self._model(arr, cache=batch_cache)
            mx.eval(out)
            for local, (name, ci, start, end) in enumerate(batch_spans):
                total = 0.0
                for pos in range(start, end):
                    step_logits = out[local, pos - 1, :]
                    target = int(arr[local, pos])
                    total += float(step_logits[target] - mx.logsumexp(step_logits))
                log_probs.setdefault(name, {})[ci] = total
            cursor += len(batch_rows)

        values: dict[str, FieldValue] = {}
        for cf in fields:
            per_choice = log_probs.get(cf.field.name, {})
            lps = [per_choice.get(ci, -1e9) for ci in range(len(cf.sequences))]
            probs = mx.softmax(mx.array(lps))
            mx.eval(probs)
            plist = [float(p) for p in probs]
            best = max(range(len(plist)), key=plist.__getitem__)
            answers = cf.field.answers
            value = (answers[best] == "true") if cf.field.is_boolean else answers[best]
            alternatives = sorted(
                ((answers[i], plist[i]) for i in range(len(answers)) if i != best),
                key=lambda item: -item[1],
            )
            values[cf.field.name] = FieldValue(
                cf.field.name, value, plist[best], alternatives[:4], approximate=False,
            )
        return values


def _estimate_kv_bytes_per_token(model) -> int | None:
    """Best-effort KV cache size per token, from the model's config if available."""
    try:
        args = getattr(model, "args", None) or getattr(model, "config", None)
        if args is None:
            return None
        layers = getattr(args, "num_hidden_layers", None) or getattr(args, "n_layers", None)
        kv_heads = getattr(args, "num_key_value_heads", None) or getattr(args, "n_kv_heads", None)
        head_dim = getattr(args, "head_dim", None)
        hidden = getattr(args, "hidden_size", None)
        n_heads = getattr(args, "num_attention_heads", None)
        if head_dim is None and hidden and n_heads:
            head_dim = hidden // n_heads
        if not (layers and kv_heads and head_dim):
            return None
        return int(layers) * int(kv_heads) * int(head_dim) * 2 * 2  # K + V, fp16
    except Exception:
        return None
