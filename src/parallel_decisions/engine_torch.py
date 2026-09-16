"""Torch (CUDA / CPU) backend: the same decoding, different hardware.

The primary engine (`engine.py`) runs on Apple Silicon via MLX. This module
implements the same three-step scheme on PyTorch so an NVIDIA-GPU or CPU-only
machine can run the package too — the E1 cross-check from the roadmap:

1. one prefill of `build_prompt()` into a HF `DynamicCache`,
2. the cache broadcast to one row per decision field, one batched forward pass
   over the padded suffix rows, logits sliced per field,
3. colliding fields resolved by exact sequence scoring in extra batched passes.

Answers, collisions, calibration and telemetry are identical to the MLX path;
only the tensor plumbing differs. Import torch lazily: like `engine.py`, this
module must never break `import parallel_decisions` on a machine without it.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Sequence

from .prompts import build_prompt
from .schema import CompiledField

DEFAULT_TORCH_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class TorchRuntime:
    """Holds the model + tokenizer for one backend instance.

    Kept separate from the `Decider` facade so tests can construct a runtime
    around a tiny model without touching the MLX code path.
    """

    def __init__(self, model_id: str | None = None, *, dtype: str | None = None,
                 device: str | None = None, verbose: bool = False):
        self.model_id = model_id or DEFAULT_TORCH_MODEL
        self._dtype_pref = dtype
        self._device_pref = device
        self.verbose = verbose
        self.torch = None
        self.model = None
        self.tokenizer = None
        self.device = None
        self.dtype = None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[parallel-decisions/torch] {message}", flush=True)

    def load(self) -> "TorchRuntime":
        if self.model is not None:
            return self
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = self._device_pref or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False")
        if self._dtype_pref:
            self.dtype = getattr(torch, self._dtype_pref)
        elif self.device.startswith("cuda"):
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.dtype = torch.float32

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        load_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
        if self.dtype is not None:
            load_kwargs["dtype"] = self.dtype  # transformers>=5; torch_dtype deprecated
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self._log(f"loaded {self.model_id} on {self.device} ({self.dtype}) "
                  f"in {time.perf_counter() - t0:.1f}s")
        return self

    # ------------------------------------------------------------- primitives
    def prefill(self, tokens: list[int]):
        """One forward pass over the full prompt; returns (cache, prompt_len)."""
        torch = self.torch
        inputs = torch.tensor([tokens], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(inputs, use_cache=True)
        return out.past_key_values, inputs.shape[1]

    def broadcast(self, cache, n: int):
        """Repeat a batch-1 cache to n identical rows without touching the original.

        The upstream port calls `DynamicCache.batch_repeat_interleave`, which mutates
        the layer objects in place — and after a shallow copy those layers are still
        *shared* with the caller's cache. A second pass over the same prompt cache
        (collision resolution, chunking, prefix reuse) then finds an already-broadcast
        cache and dies in `torch.cat` with a batch mismatch. So: copy every layer
        object and write freshly repeated tensors into the copies, leaving the
        original cache exactly as it was.
        """
        torch = self.torch
        layers = getattr(cache, "layers", None)
        if layers is None:  # legacy tuple-of-tuples cache
            if n == 1:
                return copy.deepcopy(cache)
            return tuple(
                tuple(t.repeat(n, 1, 1, 1) for t in layer) for layer in cache)
        new_cache = type(cache)()
        new_layers = []
        for layer in layers:
            new_layer = copy.copy(layer)
            for attr in ("keys", "values"):
                tensor = getattr(layer, attr, None)
                if (tensor is not None and tensor.dim() >= 1
                        and tensor.shape[0] == 1 and n > 1):
                    tensor = tensor.repeat_interleave(n, dim=0)
                setattr(new_layer, attr, tensor)
            new_layers.append(new_layer)
        new_cache.layers = new_layers
        return new_cache

    def batched_pass(self, cache, rows: list[list[int]], pad_id: int):
        """One forward pass over padded suffix/answer rows against a broadcast cache.

        The cache holds the prompt prefix; each row appends its own suffix, so the
        attention mask marks every position valid (suffix padding is trailing and
        never read at the scored positions). With `in_place=True` the broadcast
        happens inside the caller's cache copy — one deep copy fewer per pass.
        """
        torch = self.torch
        max_len = max(len(r) for r in rows)
        arr = torch.tensor([r + [pad_id] * (max_len - len(r)) for r in rows],
                           dtype=torch.long, device=self.device)
        batched = self.broadcast(cache, len(rows))
        prefix_len = batched.get_seq_length() if hasattr(batched, "get_seq_length") else 0
        mask = torch.ones((len(rows), prefix_len + max_len), dtype=torch.long,
                          device=self.device)
        with torch.no_grad():
            out = self.model(arr, past_key_values=batched, attention_mask=mask,
                             use_cache=True)
        return out.logits, arr

    def pad_id(self) -> int:
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self.tokenizer, "eos_token_id", 0)
        return int(pad or 0)


def slice_decision(rt: TorchRuntime, cf: CompiledField, row_logits, temperature: float):
    """Logit slice + softmax over the candidate first tokens, MLX-path semantics."""
    import torch
    import torch.nn.functional as F

    scores = []
    for ids in cf.candidate_ids:
        scores.append(max(float(row_logits[i]) for i in ids) if ids else -1e9)
    tensor = torch.tensor(scores, dtype=torch.float32) / max(temperature, 1e-4)
    probs = F.softmax(tensor, dim=-1).tolist()
    return probs


def resolve_collision_rows(rt: TorchRuntime, cache, groups: Sequence[CompiledField],
                           max_collision_rows: int):
    """Exact sequence log-probabilities for fields whose answers share a first token."""
    rows: list[list[int]] = []
    spans: list[tuple[str, int, int, int]] = []
    for cf in groups:
        suffix_len = len(cf.suffix_tokens)
        for ci, seq in enumerate(cf.sequences):
            rows.append(cf.suffix_tokens + seq)
            spans.append((cf.row_name, ci, suffix_len, suffix_len + len(seq)))

    import torch
    import torch.nn.functional as F

    log_probs: dict[str, dict[int, float]] = {}
    pad = rt.pad_id()
    cursor = 0
    while cursor < len(rows):
        batch_rows = rows[cursor:cursor + max_collision_rows]
        batch_spans = spans[cursor:cursor + max_collision_rows]
        logits, arr = rt.batched_pass(cache, batch_rows, pad)
        log_softmax = F.log_softmax(logits.float(), dim=-1)
        for local, (name, ci, start, end) in enumerate(batch_spans):
            total = 0.0
            for pos in range(start, end):
                total += float(log_softmax[local, pos - 1, int(arr[local, pos])])
            log_probs.setdefault(name, {})[ci] = total
        cursor += len(batch_rows)
    return log_probs


def decide_torch(rt: TorchRuntime, context: str, schema, compiled, *, temperature: float = 1.0,
                 max_collision_rows: int = 8, fields_per_pass: int | None = None,
                 calibration=None) -> dict[str, Any]:
    """The MLX `decide()` pipeline, on torch. Returns a telemetry dict.

    `compiled` is the schema's CompiledField list (multi fields already expanded),
    `calibration` an optional callable applied to each {answer: prob} distribution.
    """
    from .engine import FieldValue

    t_start = time.perf_counter()
    rt.load()
    pad = rt.pad_id()
    prompt = build_prompt(context, schema)
    tokens = rt.tokenizer.encode(prompt)

    t_pre = time.perf_counter()
    cache, _ = rt.prefill(tokens)
    prefill_ms = (time.perf_counter() - t_pre) * 1000

    fields_per_pass = fields_per_pass or len(compiled)
    values: dict[str, FieldValue] = {}
    plain: dict[str, FieldValue] = {}
    per_choice: dict[str, dict[int, FieldValue]] = {}
    pass_ms = 0.0
    chunks = 0
    start = 0
    while start < len(compiled):
        group = compiled[start:start + fields_per_pass]
        t0 = time.perf_counter()
        logits, _ = rt.batched_pass(cache, [cf.suffix_tokens for cf in group], pad)
        pass_ms += (time.perf_counter() - t0) * 1000
        chunks += 1

        collisions: list[CompiledField] = []
        for local, cf in enumerate(group):
            if cf.collision:
                collisions.append(cf)
                continue
            probs = slice_decision(rt, cf, logits[local, len(cf.suffix_tokens) - 1, :],
                                   temperature)
            fv = _field_value(cf, probs)
            if cf.choice_index is None:
                plain[cf.field.name] = fv
            else:
                per_choice.setdefault(cf.field.name, {})[cf.choice_index] = fv
        if collisions:
            log_probs = resolve_collision_rows(rt, cache, collisions, max_collision_rows)
            for cf in collisions:
                lps = [log_probs.get(cf.row_name, {}).get(ci, -1e9)
                       for ci in range(len(cf.sequences))]
                import torch.nn.functional as F
                probs = F.softmax(rt.torch.tensor(lps, dtype=torch.float32), dim=-1).tolist()
                fv = _field_value(cf, probs)
                if cf.choice_index is None:
                    plain[cf.field.name] = fv
                else:
                    per_choice.setdefault(cf.field.name, {})[cf.choice_index] = fv
        start += len(group)

    values.update(plain)
    for name, by_index in per_choice.items():
        field = schema.fields[name]
        scores = {c: float(by_index[i].probability) if i in by_index else 0.0
                  for i, c in enumerate(field.choices)}
        included = [c for c in field.choices if scores[c] >= 0.5]
        if included:
            value: Any = included
            probability = min(scores[c] for c in included)
        else:
            value = []
            probability = 1.0 - max(scores.values()) if scores else 0.0
        values[name] = FieldValue(name, value, probability,
                                  sorted(((c, p) for c, p in scores.items() if c not in included),
                                         key=lambda item: -item[1])[:4],
                                  distribution=dict(scores))

    if calibration is not None:
        values = {name: calibration(fv) for name, fv in values.items()}

    return {
        "values": values,
        "model": rt.model_id,
        "device": str(rt.device),
        "latency_ms": (time.perf_counter() - t_start) * 1000,
        "prefill_ms": prefill_ms,
        "pass_ms": pass_ms,
        "fields_evaluated": len(compiled),
        "chunks": chunks,
        "prompt_tokens": len(tokens),
    }


def _field_value(cf: CompiledField, plist: Sequence[float]) -> "FieldValue":
    from .engine import FieldValue

    answers = cf.field.answers
    best = max(range(len(plist)), key=lambda i: plist[i])
    value = (answers[best] == "true") if cf.field.is_boolean else answers[best]
    distribution = {answers[i]: float(plist[i]) for i in range(len(answers))}
    alternatives = sorted(((k, v) for k, v in distribution.items() if k != answers[best]),
                          key=lambda item: -item[1])
    return FieldValue(cf.field.name, value, float(plist[best]), alternatives[:4],
                      distribution=distribution)
