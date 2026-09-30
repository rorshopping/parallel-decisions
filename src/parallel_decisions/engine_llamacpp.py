"""Local GGUF CPU backend using native multi-sequence KV branching.

No completion/chat API, generated JSON, alternate prompt, model downloader or
serial field prefill is used. Optional native/numpy imports happen only at load.
The 0.3.35 binding's low-level API is deliberately version-pinned in the extra.
"""
from __future__ import annotations

import math
import os
import time
from contextlib import ExitStack, closing
from pathlib import Path

from .prompts import build_prompt


class GGUFTokenizer:
    """Schema-compatible adapter; only full prompts parse ChatML special tokens."""

    def __init__(self, model):
        self.model = model

    def encode(self, text, add_special_tokens=True):
        return self.model.tokenize(text.encode("utf-8"),
                                   add_bos=add_special_tokens,
                                   special=add_special_tokens)


class LlamaCppRuntime:
    def __init__(self, model_id, *, n_ctx=4096, n_batch=512, n_threads=None,
                 max_rows=32, verbose=False):
        self.model_id = str(model_id)
        self.n_ctx = _positive_int("n_ctx", n_ctx)
        self.n_batch = _positive_int("n_batch", n_batch)
        self.n_threads = _positive_int("n_threads", n_threads if n_threads is not None
                                       else min(8, os.cpu_count() or 1))
        self.max_rows = _positive_int("max_rows", max_rows)
        if self.n_batch > self.n_ctx:
            raise ValueError("n_batch must not exceed n_ctx")
        if self.max_rows > self.n_batch:
            raise ValueError("n_batch must be at least the maximum field/collision row limit")
        self.verbose = verbose
        self.model = self.tokenizer = self.ctx = self.memory = self.batch = None
        self._stack = ExitStack()
        self.stats = {}

    def load(self):
        if self.model is not None:
            return self
        path = Path(self.model_id).expanduser()
        if path.suffix.lower() != ".gguf" or not path.is_file():
            raise ValueError("llamacpp requires an existing local .gguf model file; no downloads")
        try:
            import llama_cpp
            from llama_cpp import llama_cpp as native
            from llama_cpp import _internals
            import numpy as np
        except ImportError as exc:
            raise ImportError("llamacpp requires the optional llama-cpp-python==0.3.35 dependency") from exc
        if llama_cpp.__version__ != "0.3.35":
            raise RuntimeError("llamacpp currently requires llama-cpp-python==0.3.35 (native ABI)")
        required = ("llama_get_memory", "llama_memory_seq_cp", "llama_memory_seq_rm",
                    "llama_memory_clear", "llama_batch_init", "llama_batch_free",
                    "llama_decode", "llama_get_logits_ith", "llama_n_ctx_seq",
                    "llama_synchronize")
        if any(not hasattr(native, name) for name in required):
            raise RuntimeError("llama-cpp-python lacks required multi-sequence native APIs")
        if self.max_rows + 1 > native.llama_max_parallel_sequences():
            raise ValueError("row limits exceed native sequence capacity (including prefill sequence)")
        self.native, self.np = native, np
        # Process-global backend initialization is idempotent. Do not backend_free:
        # another Decider in this process can still own a native context.
        native.llama_backend_init()
        stack = ExitStack()
        try:
            mp = native.llama_model_default_params()
            mp.n_gpu_layers = 0
            # Repacked CPU kernels failed strict cross-row-limit agreement on
            # 0.3.35 (max marginal delta .00953). The original GGUF buffer
            # layout passes that gate; this does NOT guarantee arbitrary
            # decode-shape equivalence (a separate single-token probe still
            # differed by ~0.711 raw logit). See NOTES_GGUF_NATIVE_RUN.md for the
            # native run record and NOTES_GGUF_DIVERGENCE.md for the controls
            # and remaining limits.
            mp.use_extra_bufts = False
            model = stack.enter_context(closing(_internals.LlamaModel(
                path_model=str(path), params=mp, verbose=self.verbose)))
            metadata = model.metadata()
            # Only full-attention Qwen2 GGUF is currently verified. A recurrent or
            # hybrid model may not support the shared-prefix seq_cp semantics.
            if metadata.get("general.architecture") != "qwen2":
                raise ValueError("llamacpp currently supports full-attention Qwen2 GGUF only")
            cp = native.llama_context_default_params()
            cp.n_ctx = self.n_ctx
            cp.n_batch = self.n_batch
            cp.n_ubatch = self.n_batch
            cp.n_seq_max = self.max_rows + 1
            cp.kv_unified = True
            cp.n_threads = cp.n_threads_batch = self.n_threads
            cp.embeddings = False
            cp.offload_kqv = False
            cp.op_offload = False
            ctx = stack.enter_context(closing(_internals.LlamaContext(
                model=model, params=cp, verbose=self.verbose)))
            memory = native.llama_get_memory(ctx.ctx)
            if not memory:
                raise RuntimeError("model has no branchable native KV memory")
            batch = native.llama_batch_init(self.n_batch, 0, 1)
            stack.callback(native.llama_batch_free, batch)
            if not batch.token:
                raise MemoryError("native batch allocation failed")
            self.n_ctx_seq = native.llama_n_ctx_seq(ctx.ctx)
            self.n_vocab = model.n_vocab()
            self.metadata = {"architecture": metadata["general.architecture"],
                             "parameters": model.n_params(), "description": model.desc(),
                             "llama_cpp_python": llama_cpp.__version__}
            self.model, self.ctx, self.memory, self.batch = model, ctx, memory, batch
            self.tokenizer = GGUFTokenizer(model)
            self._stack = stack.pop_all()
        finally:
            stack.close()
        return self

    def close(self):
        self._stack.close()
        self.model = self.tokenizer = self.ctx = self.memory = self.batch = None

    def __del__(self):
        # ExitStack itself does not run callbacks at GC. Native batch ownership
        # belongs here, including callers that simply drop their Decider.
        stack = getattr(self, "_stack", None)
        if stack is not None:
            stack.close()

    def reset(self):
        # Clear buffers as well as membership: no stale per-request KV survives.
        self.native.llama_memory_clear(self.memory, True)

    def _decode(self, entries):
        """entries=(token, absolute position, sequence id, request logits).

        Copy every requested output before another native decode can recycle it.
        Positive get_logits_ith indices refer to *batch token* indices, not the
        compact output index: sparse logits flags are safe, including index zero.
        """
        batch = self.batch
        batch.n_tokens = len(entries)
        for i, (token, pos, seq, logits) in enumerate(entries):
            batch.token[i] = token
            batch.pos[i] = pos
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = seq
            batch.logits[i] = bool(logits)
        code = self.native.llama_decode(self.ctx.ctx, batch)
        if code != 0:
            raise RuntimeError(f"llama_decode failed ({code}); request discarded, no serial fallback")
        self.stats["decode_calls"] += 1
        rows = len({e[2] for e in entries if e[2] != 0})
        self.stats["max_parallel_rows"] = max(self.stats["max_parallel_rows"], rows)
        if rows > 1:
            self.stats["parallel_decode_calls"] += 1
        out = {}
        for i, entry in enumerate(entries):
            if entry[3]:
                ptr = self.native.llama_get_logits_ith(self.ctx.ctx, i)
                if not ptr:
                    raise RuntimeError("native logits missing for requested batch position")
                out[i] = self.np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).copy()
        return out

    def prefill(self, tokens):
        if not tokens or len(tokens) >= min(self.n_ctx, self.n_ctx_seq):
            raise ValueError("prompt exceeds GGUF context capacity (suffix space required)")
        self.stats["prefill_calls"] += 1
        for start in range(0, len(tokens), self.n_batch):
            entries = [(t, start + i, 0, False)
                       for i, t in enumerate(tokens[start:start + self.n_batch])]
            self._decode(entries)
            self.stats["prefill_decode_calls"] += 1
        self.native.llama_synchronize(self.ctx.ctx)
        self.prompt_length = len(tokens)

    def batched_pass(self, rows, positions):
        """Branch seq 0, decode ragged independent rows together without padding.

        positions is one set of scored offsets per row. Time-major packing ensures
        multiple independent sequence IDs participate even when rows > n_batch
        tokens in total. Chunking never changes any row's causal history.
        """
        if not rows or len(rows) > self.max_rows:
            raise ValueError("invalid native row count")
        if (self.prompt_length + sum(map(len, rows)) > self.n_ctx
                or self.prompt_length + max(map(len, rows)) > self.n_ctx_seq):
            raise ValueError("GGUF KV capacity exceeded; reduce row limits/context length or raise n_ctx")
        if any(not row for row in rows):
            raise ValueError("empty compiled suffix is unsupported")
        outputs = [{} for _ in rows]
        try:
            for seq in range(1, len(rows) + 1):
                self.native.llama_memory_seq_cp(self.memory, 0, seq, 0, self.prompt_length)
                self.stats["sequence_copies"] += 1
            entries, keys = [], []
            for offset in range(max(map(len, rows))):
                for row, tokens in enumerate(rows):
                    if offset >= len(tokens):
                        continue
                    entries.append((tokens[offset], self.prompt_length + offset,
                                    row + 1, offset in positions[row]))
                    keys.append((row, offset))
                    if len(entries) == self.n_batch:
                        for index, logits in self._decode(entries).items():
                            r, p = keys[index]
                            outputs[r][p] = logits
                        entries, keys = [], []
            if entries:
                for index, logits in self._decode(entries).items():
                    r, p = keys[index]
                    outputs[r][p] = logits
            return outputs
        finally:
            for seq in range(1, len(rows) + 1):
                if not self.native.llama_memory_seq_rm(self.memory, seq, -1, -1):
                    raise RuntimeError("native sequence cleanup failed")


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _softmax(scores):
    peak = max(scores)
    weights = [math.exp(s - peak) for s in scores]
    total = sum(weights)
    return [w / total for w in weights]


def _log_probability(rt, logits, target):
    # float64 accumulation over the entire vocabulary, not candidate-only logits.
    scores = logits.astype(rt.np.float64)
    peak = float(scores.max())
    return float(scores[target]) - peak - math.log(float(rt.np.exp(scores - peak).sum()))


def decide_llamacpp(rt, context, schema, compiled, *, temperature=1.0,
                    fields_per_pass=32, max_collision_rows=8):
    from .engine import Decider

    if not math.isfinite(temperature):
        raise ValueError("temperature must be finite")
    rt.stats = {key: 0 for key in ("prefill_calls", "prefill_decode_calls", "decode_calls",
                                 "sequence_copies", "parallel_decode_calls", "max_parallel_rows")}
    prompt = build_prompt(context, schema)
    tokens = rt.tokenizer.encode(prompt)
    # Compile and prefill refer to exactly the same causal token prefix. Do not
    # retokenize a partial schema/context or inject BOS into suffixes. Keep the
    # compiler's intentional suffix/answer boundary fallback unchanged.
    for cf in compiled:
        full = rt.tokenizer.encode(prompt + cf.suffix)
        if full != tokens + cf.suffix_tokens:
            raise ValueError(f"prompt/suffix token boundary mismatch for {cf.row_name!r}")
        ids = [i for candidate in cf.candidate_ids for i in candidate]
        ids += [i for seq in cf.sequences for i in seq]
        if not cf.suffix_tokens or any(i < 0 or i >= rt.n_vocab for i in ids):
            raise ValueError(f"invalid compiled candidates for {cf.row_name!r}")
    plain, multi = {}, {}
    chunks = 0
    rt.reset()
    try:
        t_pre = time.perf_counter()
        rt.prefill(tokens)
        prefill_ms = (time.perf_counter() - t_pre) * 1000
        t_pass = time.perf_counter()
        for start in range(0, len(compiled), fields_per_pass):
            group = compiled[start:start + fields_per_pass]
            ordinary = [cf for cf in group if not cf.collision]
            row_probs = {}
            if ordinary:
                logits = rt.batched_pass([cf.suffix_tokens for cf in ordinary],
                                         [{len(cf.suffix_tokens) - 1} for cf in ordinary])
                for cf, row in zip(ordinary, logits):
                    last = row[len(cf.suffix_tokens) - 1]
                    scores = [max(float(last[i]) for i in ids) if ids else -1e9
                              for ids in cf.candidate_ids]
                    row_probs[cf.row_name] = _softmax([s / max(temperature, 1e-4) for s in scores])
            collision_jobs = [(cf, ci, seq) for cf in group if cf.collision
                              for ci, seq in enumerate(cf.sequences)]
            totals = {}
            for offset in range(0, len(collision_jobs), max_collision_rows):
                jobs = collision_jobs[offset:offset + max_collision_rows]
                # Last answer token need not be decoded: its probability comes
                # from the previous position. This is exact teacher-forced scoring.
                rows = [cf.suffix_tokens + seq[:-1] for cf, _, seq in jobs]
                positions = [set(range(len(cf.suffix_tokens) - 1,
                                       len(cf.suffix_tokens) + len(seq) - 1))
                             for cf, _, seq in jobs]
                logits = rt.batched_pass(rows, positions)
                for (cf, ci, seq), row in zip(jobs, logits):
                    total = sum(_log_probability(rt, row[len(cf.suffix_tokens) - 1 + j], tok)
                                for j, tok in enumerate(seq))
                    totals.setdefault(cf.row_name, {})[ci] = total
            for cf in group:
                # Existing MLX/Torch semantics: temperature only affects the
                # first-token path; exact collision sequence scores are unscaled.
                probs = (_softmax([totals[cf.row_name][i] for i in range(len(cf.sequences))])
                         if cf.collision else row_probs[cf.row_name])
                fv = Decider._field_value(cf, probs)
                if cf.choice_index is None:
                    plain[cf.field.name] = fv
                else:
                    # Shared assembly reads P(true) from the boolean distribution.
                    multi.setdefault(cf.field.name, {})[cf.choice_index] = fv
            chunks += 1
        values = dict(plain)
        for name, by_index in multi.items():
            values[name] = Decider._assemble_multi(schema.fields[name], by_index)
        pass_ms = (time.perf_counter() - t_pass) * 1000
        return {"values": values, "prefill_ms": prefill_ms, "pass_ms": pass_ms,
                "chunks": chunks, "fields_evaluated": len(compiled),
                "telemetry": {"backend": "llamacpp", "device": "cpu",
                              "shared_prefix": False, "prompt_tokens": len(tokens),
                              "n_ctx": rt.n_ctx, "n_ctx_seq": rt.n_ctx_seq,
                              "n_batch": rt.n_batch, **rt.metadata, **rt.stats}}
    finally:
        rt.reset()
