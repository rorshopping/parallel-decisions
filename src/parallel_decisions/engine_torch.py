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
import logging
import time
from typing import Any, Sequence

from .prompts import build_prompt, build_prompt_parts
from .schema import CompiledField

DEFAULT_TORCH_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class TorchCudaGraphCache:
    """Bounded, runtime-local suffix graphs; a shape's first use stays eager.

    Prefill is deliberately NOT captured. In transformers 5.17 DynamicLayer.update
    replaces keys/values with torch.cat, while StaticLayer.update uses index_copy_
    and an in-place tensor length counter. Use the latter with stable staged inputs,
    resetting every counter and broadcasting fresh prefix KV *inside* each replay.
    Qwen2 accepts a prebuilt mask mapping, avoiding HF's data-dependent mask setup.
    No torch.compile/Triton dependency is needed (including on Windows).
    """

    def __init__(self, runtime, *, max_graphs: int = 4, bucket_size: int = 128):
        self.runtime = runtime
        self.max_graphs = max_graphs
        self.bucket_size = bucket_size
        self.entries: dict[tuple[int, int, int], Any] = {}
        self.seen: set[tuple[int, int, int]] = set()
        self.disabled = False
        self.disable_reason: str | None = None
        self.capture_ms = 0.0  # cumulative setup + warmup + capture, not replay
        self.replays = 0

    def run(self, cache, rows, pad_id):
        rt = self.runtime
        if self.disabled:
            return rt._batched_pass_eager(cache, rows, pad_id)
        try:
            prefix_len = cache.get_seq_length()
            suffix_len = max(map(len, rows))
            bucket = ((prefix_len + self.bucket_size - 1) // self.bucket_size) * self.bucket_size
            key = (len(rows), suffix_len, bucket)
            entry = self.entries.get(key)
            if entry is None:
                if key not in self.seen:
                    if len(self.seen) >= 128:
                        self.seen.clear()
                    self.seen.add(key)
                    return rt._batched_pass_eager(cache, rows, pad_id)
                # Keep graph-private memory bounded; uncached shapes remain eager.
                if len(self.entries) >= self.max_graphs:
                    return rt._batched_pass_eager(cache, rows, pad_id)
                rt.synchronize()
                started = time.perf_counter()
                try:
                    entry = self._capture(cache, rows, pad_id, key)
                finally:
                    rt.synchronize()
                    self.capture_ms += (time.perf_counter() - started) * 1000
                self.entries[key] = entry
            self._stage(entry, cache, rows, pad_id)
            entry["graph"].replay()
            self.replays += 1
            # Callers may retain results across another replay (collision passes).
            return entry["logits"].clone(), entry["tokens"].clone()
        except Exception as exc:
            self.disabled = True
            self.disable_reason = f"{type(exc).__name__}: {exc}"
            self.entries.clear()
            logging.getLogger(__name__).warning(
                "CUDA graph fallback: disabling graphs for this runtime (%s: %s)",
                type(exc).__name__, exc)
        return rt._batched_pass_eager(cache, rows, pad_id)

    def _stage(self, entry, cache, rows, pad_id):
        torch = self.runtime.torch
        length = cache.get_seq_length()
        entry["length"].fill_(length)
        width = entry["tokens"].shape[1]
        entry["tokens"].copy_(torch.tensor(
            [r + [pad_id] * (width - len(r)) for r in rows],
            dtype=torch.long, device=self.runtime.device))
        for source, (keys, values) in zip(cache.layers, entry["prefix"]):
            keys[..., :length, :].copy_(source.keys)
            values[..., :length, :].copy_(source.values)
            # A shorter prompt must not leave old data, including NaNs, in padding.
            keys[..., length:, :].zero_()
            values[..., length:, :].zero_()

    def _capture(self, cache, rows, pad_id, key):
        from transformers.cache_utils import DynamicLayer, StaticCache, StaticLayer

        rt = self.runtime
        torch = rt.torch
        config = rt.model.config
        # Only this architecture/API has been verified. Unknown or sliding/hybrid
        # caches must not silently produce a full-attention answer instead.
        if (config.model_type != "qwen2" or rt.model.training
                or config._attn_implementation not in ("eager", "sdpa")
                or config.rope_parameters["rope_type"] != "default"
                or any(type(layer) is not DynamicLayer for layer in cache.layers)):
            raise ValueError("graphs currently support eval-mode full-attention Qwen2 only")
        n, suffix_len, bucket = key
        static = StaticCache(config=config, max_cache_len=bucket + suffix_len)
        if any(type(layer) is not StaticLayer for layer in static.layers):
            raise ValueError("sliding/hybrid StaticCache is not graph-supported")
        if len(static.layers) != len(cache.layers):
            raise ValueError("incomplete prompt cache")
        # Refuse oversized shapes before allocating: a conservative bound on KV,
        # repeated attention KV, logits and activations. This is not a VRAM quota.
        kv_bytes = sum((l.keys.numel() + l.values.numel()) * l.keys.element_size()
                       for l in cache.layers) * n * (bucket + suffix_len) / max(1, cache.get_seq_length())
        logits_bytes = n * suffix_len * config.vocab_size * 4
        # Attention expands KV one layer at a time, not all layers concurrently.
        attention_bytes = kv_bytes / len(cache.layers) * (
            2 * config.num_attention_heads / config.num_key_value_heads)
        estimated_bytes = kv_bytes * 2 + attention_bytes + logits_bytes * 3
        free_bytes, _ = torch.cuda.mem_get_info(rt.device)
        if estimated_bytes > min(2 * 1024**3, free_bytes // 2):
            raise ValueError(
                f"graph shape exceeds 2 GiB or half of free VRAM: "
                f"kv_bytes={kv_bytes:.0f}, attention_bytes={attention_bytes:.0f}, "
                f"logits_bytes={logits_bytes}, estimated_bytes={estimated_bytes:.0f}, "
                f"free_bytes={free_bytes}")
        entry = {
            "cache": static,
            "tokens": torch.empty((n, suffix_len), dtype=torch.long, device=rt.device),
            "length": torch.zeros((), dtype=torch.long, device=rt.device),
            "prefix": [],
        }
        for source, target in zip(cache.layers, static.layers):
            target.lazy_initialization(source.keys.expand(n, -1, -1, -1),
                                       source.values.expand(n, -1, -1, -1))
            entry["prefix"].append(tuple(torch.zeros(
                (1, tensor.shape[1], bucket, tensor.shape[-1]),
                dtype=tensor.dtype, device=tensor.device)
                for tensor in (source.keys, source.values)))
        self._stage(entry, cache, rows, pad_id)
        query = torch.arange(suffix_len, device=rt.device)
        keys = torch.arange(bucket + suffix_len, device=rt.device)
        entry["query_indices"] = query
        entry["key_indices"] = keys

        def forward():
            for layer, (k, v) in zip(static.layers, entry["prefix"]):
                layer.keys[..., :bucket, :].copy_(k)
                layer.values[..., :bucket, :].copy_(v)
                layer.cumulative_length.copy_(entry["length"])
            positions = query + entry["length"]
            allowed = keys[None, :] <= positions[:, None]
            mask = torch.zeros((suffix_len, bucket + suffix_len),
                               dtype=rt.dtype, device=rt.device)
            mask.masked_fill_(~allowed, torch.finfo(rt.dtype).min)
            return rt.model(
                entry["tokens"], past_key_values=static,
                position_ids=positions[None, :],
                attention_mask={"full_attention": mask[None, None, :, :]},
                use_cache=True).logits

        # cuBLAS and allocator initialization must happen outside capture and on a
        # side stream. Keep all static tensors and the output alive with the graph.
        with torch.cuda.device(rt.device), torch.no_grad():
            stream = torch.cuda.Stream(device=rt.device)
            stream.wait_stream(torch.cuda.current_stream(rt.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    forward()
            torch.cuda.current_stream(rt.device).wait_stream(stream)
            rt.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                entry["logits"] = forward()
            entry["graph"] = graph
        return entry


class TorchRuntime:
    """Holds the model + tokenizer for one backend instance.

    Kept separate from the `Decider` facade so tests can construct a runtime
    around a tiny model without touching the MLX code path.
    """

    def __init__(self, model_id: str | None = None, *, dtype: str | None = None,
                 device: str | None = None, verbose: bool = False,
                 cuda_graph: bool = False):
        self.model_id = model_id or DEFAULT_TORCH_MODEL
        self._dtype_pref = dtype
        self._device_pref = device
        self.verbose = verbose
        self.torch = None
        self.model = None
        self.tokenizer = None
        self.device = None
        self.dtype = None
        self.cuda_graph = cuda_graph
        self.graph_cache = TorchCudaGraphCache(self)

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

    def synchronize(self):
        """Wait for this device before recording wall-clock inference timings."""
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------- primitives
    def prefill(self, tokens: list[int], cache=None):
        """One forward pass over the given tokens; returns (cache, n_prompt_tokens).

        With `cache` given, the tokens are appended to it (shared-prefix reuse);
        the caller owns the cache and must pass a private copy.
        """
        torch = self.torch
        inputs = torch.tensor([tokens], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(inputs, use_cache=True, past_key_values=cache)
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
        if n == 1:
            return self._copy_torch_cache(cache)
        torch = self.torch
        layers = getattr(cache, "layers", None)
        if layers is None:  # legacy tuple-of-tuples cache
            return tuple(
                tuple(t.repeat(n, 1, 1, 1) for t in layer) for layer in cache)
        new_cache = type(cache)()
        new_layers = []
        for layer in layers:
            new_layer = copy.copy(layer)
            for attr in ("keys", "values"):
                tensor = getattr(layer, attr, None)
                if (tensor is not None and tensor.dim() >= 1
                        and tensor.shape[0] == 1):
                    tensor = tensor.repeat_interleave(n, dim=0)
                setattr(new_layer, attr, tensor)
            new_layers.append(new_layer)
        new_cache.layers = new_layers
        return new_cache

    @staticmethod
    def _copy_torch_cache(cache):
        """Private copy for one call: layer objects copied, tensors shared.

        HF's in-place `update()` appends into whatever layer objects the cache
        holds; sharing them lets one call's append grow (and corrupt) a cache the
        caller still holds — the prompt cache when a pass has a single row, the
        prepared prefix in decide_with_prefix(). Copying the layer objects makes
        each call's appends land in its own private cache.
        """
        try:
            from transformers.cache_utils import DynamicLayer
        except ImportError:  # legacy transformers cache API
            return copy.deepcopy(cache)

        layers = getattr(cache, "layers", None)
        # Only ordinary DynamicLayer is known to replace rather than modify its
        # tensors. Clone unknown/hybrid/legacy layouts conservatively.
        if layers is None or any(type(layer) is not DynamicLayer for layer in layers):
            return copy.deepcopy(cache)
        new_cache = copy.copy(cache)
        new_cache.layers = [copy.copy(layer) for layer in layers]
        return new_cache

    def batched_pass(self, cache, rows: list[list[int]], pad_id: int):
        if self.cuda_graph and str(self.device).startswith("cuda"):
            return self.graph_cache.run(cache, rows, pad_id)
        return self._batched_pass_eager(cache, rows, pad_id)

    def _batched_pass_eager(self, cache, rows: list[list[int]], pad_id: int):
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


def prepare_torch(rt: TorchRuntime, schema, compiled) -> dict[str, Any]:
    """Prefill the schema block once for reuse across many contexts.

    Mirrors `Decider.prepare()` on the MLX path: the schema block and the user
    turn's opening are byte-identical for every context decided against the same
    schema, so their prefill can be shared. The split is only valid when the
    tokenizer keeps the prefix intact (`encode(prefix)` is a token-prefix of the
    whole prompt); when it merges across the boundary the cache is returned as
    None and the caller falls back to a full prefill, exactly like MLX does.
    """
    rt.load()
    prefix_text, _ = build_prompt_parts("", schema)
    prefix_tokens = rt.tokenizer.encode(prefix_text)
    probe_tokens = rt.tokenizer.encode(build_prompt("x", schema))
    if list(probe_tokens[:len(prefix_tokens)]) != list(prefix_tokens):
        return {"cache": None, "tokens": prefix_tokens, "compiled": compiled}
    cache, _ = rt.prefill(prefix_tokens)
    rt.synchronize()
    return {"cache": cache, "tokens": prefix_tokens, "compiled": compiled}


def decide_torch(rt: TorchRuntime, context: str, schema, compiled, *, temperature: float = 1.0,
                 max_collision_rows: int = 8, fields_per_pass: int | None = None,
                 calibration=None, prefix: dict[str, Any] | None = None) -> dict[str, Any]:
    """The MLX `decide()` pipeline, on torch. Returns a telemetry dict.

    `compiled` is the schema's CompiledField list (multi fields already expanded),
    `calibration` an optional callable applied to each {answer: prob} distribution.
    `prefix` is a prepared dict from `prepare_torch()`: its schema-block cache is
    reused and only the context's own tokens are prefilled. The copy before the
    append exists so the shared prefix cache is never grown by a call.
    """
    from .engine import FieldValue

    t_start = time.perf_counter()
    rt.load()
    pad = rt.pad_id()
    # Validate the actual context, not just prepare()'s probe: a boundary merge
    # can depend on its first character. Use full tokenization on either path.
    prompt_tokens_list = rt.tokenizer.encode(build_prompt(context, schema))
    shared_prefix = False
    if prefix is not None and prefix.get("cache") is not None:
        n_prefix = len(prefix["tokens"])
        shared_prefix = (list(prefix["tokens"])
                         == list(prompt_tokens_list[:n_prefix]))
        remainder_tokens = prompt_tokens_list[n_prefix:]
    if shared_prefix:
        prompt_tokens = len(prompt_tokens_list)
        rt.synchronize()
        t_pre = time.perf_counter()
        cache = rt._copy_torch_cache(prefix["cache"])
        cache, _ = rt.prefill(remainder_tokens, cache=cache)
        rt.synchronize()
        prefill_ms = (time.perf_counter() - t_pre) * 1000
        shared_prefix = True
    else:
        prompt_tokens = len(prompt_tokens_list)
        rt.synchronize()
        t_pre = time.perf_counter()
        cache, _ = rt.prefill(prompt_tokens_list)
        rt.synchronize()
        prefill_ms = (time.perf_counter() - t_pre) * 1000
        shared_prefix = False

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
                import torch
                import torch.nn.functional as F
                probs = F.softmax(rt.torch.tensor(lps, dtype=torch.float32), dim=-1).tolist()
                fv = _field_value(cf, probs)
                if cf.choice_index is None:
                    plain[cf.field.name] = fv
                else:
                    per_choice.setdefault(cf.field.name, {})[cf.choice_index] = fv
        rt.synchronize()
        pass_ms += (time.perf_counter() - t0) * 1000
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
        "prompt_tokens": prompt_tokens,
        "shared_prefix": shared_prefix,
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
