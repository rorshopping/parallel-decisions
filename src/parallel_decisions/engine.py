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
import json
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .calibration import Calibrator
from .config import Config, load_config
from .prompts import build_prompt
from .schema import CompiledField, Field, Schema


class _LazyMLX:
    """Import mlx on first attribute access.

    Everything that does not need the GPU — schema validation, calibration fitting,
    the lint, `pd config` — must work on a machine without MLX installed. Importing
    it at module scope made `import parallel_decisions` fail there, which is a bad
    failure mode for a library whose helpers are useful without a model.
    """

    _module: Any = None

    def __getattr__(self, name: str) -> Any:
        if _LazyMLX._module is None:
            import mlx.core as mx
            _LazyMLX._module = mx
        return getattr(_LazyMLX._module, name)


mx = _LazyMLX()

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_MEMORY_BUDGET_GB = 6.0
MEMORY_HEADROOM = 0.65      # model + KV broadcast may use at most this share of RAM
DEFAULT_MAX_FIELDS = 32
DEFAULT_MAX_COLLISION_ROWS = 8


class ConcurrencyError(RuntimeError):
    """Raised when another thread is already inside the model."""


class UnsupportedPlatformError(RuntimeError):
    """Raised on machines this package does not support (non-arm64)."""


@dataclass
class FieldValue:
    """One typed decision.

    `probability` is the confidence of the chosen value. Without calibration it is
    the raw softmax slice; with a `Calibrator` attached to the `Decider` it is the
    calibrated value and `raw_probability` keeps the original for comparison.

    `distribution` holds the probability of *every* allowed answer (not just the
    runner-up in `alternatives`), which is what the calibrator consumes.
    """

    name: str
    value: Any
    probability: float
    alternatives: list[tuple[str, float]]
    approximate: bool = False
    distribution: dict[str, float] = field(default_factory=dict)
    raw_probability: float | None = None
    calibrated: bool = False

    def __post_init__(self) -> None:
        if self.raw_probability is None:
            self.raw_probability = self.probability

    def to_dict(self, probabilities: bool = True) -> dict:
        out: dict[str, Any] = {"value": self.value}
        if probabilities:
            out["probability"] = round(self.probability, 4)
            if self.calibrated and self.raw_probability is not None:
                out["raw_probability"] = round(self.raw_probability, 4)
        return out

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"{self.name}={self.value!r} ({self.probability:.1%})"


class DecisionResult(dict):
    """Maps field name -> FieldValue, plus call telemetry."""

    def __init__(self, mapping: Mapping[str, FieldValue], *, model: str = "",
                 latency_ms: float = 0.0, prefill_ms: float = 0.0, pass_ms: float = 0.0,
                 fields_evaluated: int = 0, chunks: int = 0, calibrated: bool = False):
        super().__init__(mapping)
        self.model = model
        self.latency_ms = latency_ms
        self.prefill_ms = prefill_ms
        self.pass_ms = pass_ms
        self.fields_evaluated = fields_evaluated
        self.chunks = chunks
        self.calibrated = calibrated

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
    """Loads a local MLX model once and answers schemas against contexts.

    One model, one call at a time: `decide()` takes a lock, so a threaded server
    cannot interleave two forward passes on the same MLX context (which surfaces as
    a Metal error, not a clean exception). Pass `lock_timeout_s` to bound the wait.

    Any argument left as `None` is taken from `pd.toml` / `PD_*` environment
    variables when present, then from the module defaults. See `config.py`.
    """

    def __init__(self, model_id: str | None = None, *,
                 max_fields_per_batch: int | None = None,
                 memory_budget_gb: float | None = None,
                 max_collision_rows: int | None = None,
                 calibration: str | "Calibrator" | Mapping[str, Any] | None = None,
                 warmup: bool | None = None,
                 lock_timeout_s: float | None = None,
                 config: str | Config | None = None,
                 verbose: bool = False):
        cfg = config if isinstance(config, Config) else load_config(config)
        self.config = cfg
        if model_id is None:
            model_id = cfg.model
        if calibration is None:
            calibration = cfg.calibration
        if max_fields_per_batch is None:
            max_fields_per_batch = cfg.max_fields_per_batch
        if memory_budget_gb is None:
            memory_budget_gb = cfg.memory_budget_gb
        if max_collision_rows is None:
            max_collision_rows = cfg.max_collision_rows
        if warmup is None:
            warmup = cfg.warmup
        if lock_timeout_s is None:
            lock_timeout_s = cfg.lock_timeout_s

        self.model_id = model_id or DEFAULT_MODEL
        self.max_fields_per_batch = max(1, int(max_fields_per_batch or DEFAULT_MAX_FIELDS))
        self.memory_budget_bytes = int((memory_budget_gb or DEFAULT_MEMORY_BUDGET_GB) * (1024 ** 3))
        self.max_collision_rows = max(1, int(max_collision_rows or DEFAULT_MAX_COLLISION_ROWS))
        self.calibrator = _coerce_calibrator(calibration)
        self.warmup = True if warmup is None else bool(warmup)
        self.lock_timeout_s = float(lock_timeout_s) if lock_timeout_s else 0.0
        self.log_mode = (cfg.log or os.environ.get("PD_LOG", "")).strip().lower()
        self.verbose = verbose
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._kv_bytes_per_token: int | None = None
        self._model_bytes: int | None = None

    # ------------------------------------------------------------------ load
    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[parallel-decisions] {message}", flush=True)

    def load_tokenizer(self) -> "Decider":
        """Load the tokenizer alone, for schema compilation and lints.

        `pd validate --check-tokens` and the MCP lint tool only need to tokenize, so
        they must not pull a multi-gigabyte model into memory. Older `mlx-lm` has no
        tokenizer-only entry point; falling back to `load()` is correct, just heavier.
        """
        if self._tokenizer is None:
            self._check_platform()
            t0 = time.perf_counter()
            try:
                from mlx_lm.utils import load_tokenizer
            except ImportError:  # pragma: no cover - older mlx-lm
                self.load()
                return self
            self._tokenizer = load_tokenizer(self.model_id)
            self._log(f"tokenizer loaded in {time.perf_counter() - t0:.1f}s "
                      f"({self.model_id})")
        return self

    def log_event(self, event: str, **fields: Any) -> None:
        """Structured telemetry: one JSON line on stderr when PD_LOG=json."""
        if self.log_mode not in ("json", "1", "true", "yes", "on"):
            return
        payload = {"event": event, "model": self.model_id, **fields}
        print(json.dumps(payload, default=str), file=sys.stderr, flush=True)

    def _check_platform(self) -> None:
        if os.environ.get("PD_ALLOW_NON_ARM") in ("1", "true", "yes"):
            return
        machine = platform.machine()
        if machine not in ("arm64", "aarch64"):
            raise UnsupportedPlatformError(
                f"parallel-decisions needs an Apple Silicon (arm64) machine; this is "
                f"{machine!r} on {sys.platform}. See README: the torch cross-check path "
                f"(core/engine_torch.py in the research tree) is not part of the package. "
                f"Set PD_ALLOW_NON_ARM=1 to try anyway (mlx may still work).")

    def load(self) -> "Decider":
        if self._model is None:
            self._check_platform()
            with self._acquire_lock("load"):
                if self._model is None:
                    self._load_unlocked()
        return self

    def _load_unlocked(self) -> None:
        from mlx_lm import load
        t0 = time.perf_counter()
        self._log(f"loading {self.model_id} ...")
        self._model, self._tokenizer = load(self.model_id)
        self._kv_bytes_per_token = _estimate_kv_bytes_per_token(self._model)
        self._model_bytes = _estimate_model_bytes(self._model)
        elapsed = time.perf_counter() - t0
        self._log(f"loaded in {elapsed:.1f}s "
                  f"(kv/token ~{self._kv_bytes_per_token or 0} bytes)")
        self._clamp_memory_budget()
        self.log_event("load", seconds=round(elapsed, 3),
                       kv_bytes_per_token=self._kv_bytes_per_token,
                       model_bytes=self._model_bytes,
                       memory_budget_bytes=self.memory_budget_bytes,
                       config=self.config.source)
        if self.warmup:
            self._warmup()

    def _clamp_memory_budget(self) -> None:
        """Keep the model plus its KV broadcast below a share of physical RAM.

        Field rows are broadcast copies of the whole cache, so an over-optimistic
        budget does not fail cleanly — macOS swaps, and a 16 GB machine that swaps
        during a 38k-token prefill runs several times slower than one that chunks
        more. Measured on an M5 Air: a 7 GB budget on a 4 GB model with 38k-token
        prompts pushed swap to 15 GB and stalled a 48-row case for 13+ minutes.
        """
        ram = _physical_ram_bytes()
        if not ram or not self._model_bytes:
            return
        allowance = int(ram * MEMORY_HEADROOM) - self._model_bytes
        if allowance <= 0:
            self._log("model alone exceeds the RAM headroom; leaving budget unchanged")
            return
        if self.memory_budget_bytes > allowance:
            self._log(f"memory budget {self.memory_budget_bytes / 1024**3:.1f} GB "
                      f"exceeds the safe allowance "
                      f"({allowance / 1024**3:.1f} GB of {ram / 1024**3:.0f} GB RAM "
                      f"minus {self._model_bytes / 1024**3:.1f} GB of weights); "
                      f"clamping to avoid swapping")
            self.memory_budget_bytes = allowance

    # ------------------------------------------------------------------ lock
    def _acquire_lock(self, what: str):
        acquired = self._lock.acquire(timeout=max(0.0, self.lock_timeout_s))
        if not acquired:
            raise ConcurrencyError(
                f"another thread is already running the model ({what}); "
                f"waited {self.lock_timeout_s:.1f}s. One call at a time: either "
                f"serialise calls, raise lock_timeout_s, or run a second Decider.")
        return _LockGuard(self._lock)

    @property
    def model(self):
        return self.load()._model

    @property
    def tokenizer(self):
        """The tokenizer, loading the model if it is not already loaded.

        `load()` needs the real tokenizer that came with the weights; using a
        separately loaded one risks a mismatch, so the model load wins here.
        """
        return self.load()._tokenizer

    @property
    def tokenizer_for_schema(self):
        """Tokenizer for schema compilation and lints: no model load."""
        if self._tokenizer is None:
            self.load_tokenizer()
        return self._tokenizer

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
        with self._acquire_lock("decide"):
            return self._decide_locked(context, schema, temperature)

    def _decide_locked(self, context: str, schema: Schema,
                       temperature: float) -> DecisionResult:
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
        self._log(f"{len(compiled)} decision rows (multi-select fields expand per choice), "
                  f"chunk size {chunk_size}")
        plain: dict[str, FieldValue] = {}
        per_choice: dict[str, dict[int, FieldValue]] = {}
        pass_ms = 0.0
        chunks = 0
        start = 0
        while start < len(compiled):
            group = compiled[start:start + chunk_size]
            try:
                rows, chunk_ms = self._run_chunk(prompt_cache, group, temperature)
            except Exception as exc:  # noqa: BLE001 - narrowed by _is_memory_error
                if len(group) <= 1 or not _is_memory_error(exc):
                    raise
                chunk_size = max(1, len(group) // 2)
                self._log(f"pass over {len(group)} field(s) failed "
                          f"({type(exc).__name__}: {exc}); retrying with {chunk_size}")
                _clear_mlx_cache()
                continue
            for cf, fv in rows:
                if cf.choice_index is None:
                    plain[cf.field.name] = fv
                else:
                    per_choice.setdefault(cf.field.name, {})[cf.choice_index] = fv
            pass_ms += chunk_ms
            chunks += 1
            start += len(group)

        values: dict[str, FieldValue] = dict(plain)
        for name, by_index in per_choice.items():
            values[name] = self._assemble_multi(schema[name], by_index)

        if self.calibrator is not None:
            values = {name: self._apply_calibration(schema[name], fv)
                      for name, fv in values.items()}

        result = DecisionResult(
            values,
            model=self.model_id,
            latency_ms=(time.perf_counter() - t_start) * 1000,
            prefill_ms=prefill_ms,
            pass_ms=pass_ms,
            fields_evaluated=len(compiled),
            chunks=chunks,
            calibrated=self.calibrator is not None,
        )
        self.log_event(
            "decide",
            fields=len(schema),
            rows=len(compiled),
            chunks=chunks,
            context_chars=len(context),
            prompt_tokens=len(prompt_tokens),
            prefill_ms=round(prefill_ms, 1),
            pass_ms=round(pass_ms, 1),
            latency_ms=round(result.latency_ms, 1),
            calibrated=self.calibrator is not None,
            calibration_kind=self.calibrator.kind if self.calibrator else None,
        )
        return result

    def _apply_calibration(self, f: Field, fv: FieldValue) -> FieldValue:
        """Replace the raw softmax confidence with the calibrated one.

        The chosen value never changes: every supported calibrator is monotone in
        the top confidence, so calibration moves the *number*, not the answer.
        Multi-select fields are a set of independent binary decisions, so each
        choice's probability is calibrated on its own.
        """
        assert self.calibrator is not None
        raw = fv.probability
        if f.is_multi:
            scores = {c: self.calibrator.transform_confidence(p)
                      for c, p in fv.distribution.items()}
            included = [c for c in f.choices if scores.get(c, 0.0) >= 0.5]
            if included:
                value: Any = included
                calibrated = min(scores[c] for c in included)
            else:
                value = []
                calibrated = 1.0 - max(scores.values()) if scores else 0.0
            alternatives = sorted(((c, p) for c, p in scores.items() if c not in included),
                                  key=lambda item: -item[1])[:4]
            return FieldValue(name=fv.name, value=value, probability=calibrated,
                              alternatives=alternatives, approximate=fv.approximate,
                              distribution=scores, raw_probability=raw, calibrated=True)

        dist = self.calibrator.transform(fv.distribution) if fv.distribution else None
        if dist:
            top = max(dist, key=dist.__getitem__)
            calibrated = float(dist[top])
            alternatives = sorted(
                ((k, v) for k, v in dist.items() if k != top), key=lambda item: -item[1])[:4]
        else:  # no distribution recorded (shouldn't happen): map the confidence alone
            calibrated = self.calibrator.transform_confidence(raw)
            alternatives = fv.alternatives
        return FieldValue(
            name=fv.name,
            value=fv.value,
            probability=calibrated,
            alternatives=list(alternatives),
            approximate=fv.approximate,
            distribution=dist or {},
            raw_probability=raw,
            calibrated=True,
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
                   temperature: float) -> tuple[list[tuple[CompiledField, FieldValue]], float]:
        max_len = max(len(f.suffix_tokens) for f in fields)
        pad = self._pad_id
        rows = [f.suffix_tokens + [pad] * (max_len - len(f.suffix_tokens)) for f in fields]
        arr = mx.array(rows, dtype=mx.int32)

        batch_cache = self._repeat_cache(prompt_cache, len(fields))
        t0 = time.perf_counter()
        logits = self._model(arr, cache=batch_cache)
        mx.eval(logits)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        out: list[tuple[CompiledField, FieldValue]] = []
        collisions: list[CompiledField] = []
        for row, cf in enumerate(fields):
            row_logits = logits[row, len(cf.suffix_tokens) - 1, :]
            if cf.collision:
                collisions.append(cf)
            else:
                out.append((cf, self._slice_decision(cf, row_logits, temperature)))
        if collisions:
            resolved = self._resolve_collisions(prompt_cache, collisions)
            for cf in collisions:
                out.append((cf, resolved[cf.row_name]))
        return out, elapsed_ms

    @staticmethod
    def _assemble_multi(f: Field, by_index: Mapping[int, FieldValue]) -> FieldValue:
        """Fold per-choice yes/no rows into one list value.

        `distribution` is the per-choice P(include) — not a distribution over a
        partition, so it does not sum to 1. `probability` is the weakest "yes"
        among the included choices (the confidence of the *set*, not of one member);
        with nothing included it is the confidence in leaving the set empty.
        """
        scores: dict[str, float] = {}
        for i, choice in enumerate(f.choices):
            fv = by_index.get(i)
            scores[choice] = float(fv.probability) if fv is not None else 0.0
        included = [c for c in f.choices if scores[c] >= 0.5]
        if included:
            value: Any = included
            probability = min(scores[c] for c in included)
        else:
            value = []
            probability = 1.0 - max(scores.values()) if scores else 0.0
        alternatives = sorted(((c, p) for c, p in scores.items() if c not in included),
                              key=lambda item: -item[1])
        return FieldValue(name=f.name, value=value, probability=probability,
                          alternatives=alternatives[:4], distribution=dict(scores))

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
        return self._field_value(cf, plist)

    @staticmethod
    def _field_value(cf: CompiledField, plist: Sequence[float]) -> FieldValue:
        """Assemble a FieldValue from one probability per allowed answer."""
        answers = cf.field.answers
        best = max(range(len(plist)), key=plist.__getitem__)
        value = (answers[best] == "true") if cf.field.is_boolean else answers[best]
        alternatives = sorted(
            ((answers[i], plist[i]) for i in range(len(answers)) if i != best),
            key=lambda item: -item[1],
        )
        distribution = {answers[i]: float(plist[i]) for i in range(len(answers))}
        return FieldValue(cf.field.name, value, float(plist[best]), alternatives[:4],
                          distribution=distribution)

    def _resolve_collisions(self, prompt_cache, fields: Sequence[CompiledField]) -> dict[str, FieldValue]:
        """Exact sequence scoring for fields whose answers share a first token."""
        rows: list[list[int]] = []
        spans: list[tuple[str, int, int, int]] = []  # (row key, choice idx, start, end)
        for cf in fields:
            suffix_len = len(cf.suffix_tokens)
            for ci, seq in enumerate(cf.sequences):
                rows.append(cf.suffix_tokens + seq)
                spans.append((cf.row_name, ci, suffix_len, suffix_len + len(seq)))

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
            per_choice = log_probs.get(cf.row_name, {})
            lps = [per_choice.get(ci, -1e9) for ci in range(len(cf.sequences))]
            probs = mx.softmax(mx.array(lps))
            mx.eval(probs)
            plist = [float(p) for p in probs]
            values[cf.row_name] = self._field_value(cf, plist)
        return values


class _LockGuard:
    """Context manager that always releases the model lock."""

    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        self._lock.release()
        return False


def _is_memory_error(exc: BaseException) -> bool:
    """Heuristic: is this MLX/Metal saying 'not enough memory'?

    MLX raises different things across versions (`MemoryError`, `RuntimeError` with
    a Metal message, or a `std::bad_alloc` surfaced through pybind), so match on the
    text and keep the recovery path narrow: only ever retried with a smaller batch.
    """
    if isinstance(exc, MemoryError):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(key in text for key in
               ("memory", "alloc", "out of", "metal", "buffer", "resource limit"))


def _clear_mlx_cache() -> None:
    """Release MLX's cached buffers so the retry has room (best effort)."""
    try:
        import mlx.core as mx
        clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
        if clear is not None:
            clear()
    except Exception:  # pragma: no cover - cache clearing is best effort
        pass


def _coerce_calibrator(value: "str | Calibrator | Mapping[str, Any] | None") -> Calibrator | None:
    """Accept a path, a dict, or a Calibrator instance."""
    if value is None:
        return None
    if isinstance(value, Calibrator):
        return value
    if isinstance(value, Mapping):
        return Calibrator.from_dict(value)
    if isinstance(value, (str, bytes)) or hasattr(value, "__fspath__"):
        return Calibrator.from_json(str(value))
    raise TypeError(f"calibration must be a path, dict or Calibrator, got {type(value).__name__}")


def _physical_ram_bytes() -> int | None:
    """Total physical memory, or None if the platform does not report it."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and size > 0:
            return int(pages) * int(size)
    except (AttributeError, ValueError, OSError):
        pass
    return None


def _estimate_model_bytes(model) -> int | None:
    """Rough size of the loaded weights, summed over the parameter tree."""
    try:
        total = sum(int(getattr(leaf, "nbytes", 0) or 0)
                    for leaf in _tree_leaves(model.parameters()))
    except Exception:  # pragma: no cover - best effort telemetry
        return None
    return total or None


def _tree_leaves(tree: Any) -> Iterable[Any]:
    if hasattr(tree, "nbytes"):
        yield tree
        return
    if isinstance(tree, Mapping):
        for value in tree.values():
            yield from _tree_leaves(value)
        return
    if isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _tree_leaves(value)


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
