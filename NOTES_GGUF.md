# GGUF backend (`backend="llamacpp"`)

Status: implemented, unit-tested and native-smoke-tested on this machine.
Branch `feat/gguf-engine`, version 0.4.0.

## Public contract

```python
Decider(model_id=r"C:\models\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        backend="llamacpp", max_fields_per_batch=8, max_collision_rows=8,
        n_ctx=4096, n_batch=512, n_threads=4, warmup=False)
```

- `model_id` must be an existing local `.gguf` file (first shard for a split GGUF;
  companion shards must sit alongside). No download, ever.
- New settings `n_ctx` (total unified KV cells), `n_batch` (tokens per native
  decode, must be >= the larger of `max_fields_per_batch`/`max_collision_rows`)
  and `n_threads` (default `min(8, cpu_count)`), with `PD_N_CTX`, `PD_N_BATCH`,
  `PD_N_THREADS` env vars and `pd.toml` keys. Precedence unchanged.
- Fail closed on: CUDA Graphs, `torch_*` settings, `memory_budget_gb`, non-Qwen2
  GGUF architecture, missing native multi-sequence APIs, llama-cpp-python other
  than 0.3.35, prompt/suffix tokenization boundary mismatch, KV capacity
  overflow, invalid compiled candidates.
- `prepare`, `decide_with_prefix`, `decide_many(shared_prefix=True)` raise
  `NotImplementedError` — no shared prefix is claimed on this backend.
- `warmup=True` performs no extra native inference (CPU, nothing to compile);
  it is accepted for facade compatibility.
- Optional extra: `pip install ".[llamacpp]"` → `llama-cpp-python==0.3.35`.

## Native design

- `build_prompt` and `Schema.compile` are used unchanged; full-prompt
  tokenization must contain each compiled suffix as an exact token continuation,
  otherwise the request fails before prefill. BOS: the adapter requests
  `add_bos`, and the Qwen2 GGUF vocab has `add_bos=false`, so prompt tokens
  begin directly with ChatML 151644 — identical to HF tokenization, no double
  BOS (verified read-only against the pinned model).
- One logical context prefill on sequence 0 per `decide()` (native n_batch
  slices, counted in telemetry), then `llama_memory_seq_cp` to one independent
  sequence ID per decision row, then ragged time-major `llama_decode` batches
  carrying explicit token/pos/seq_id/logits — no padding tokens, no serial
  re-prefill, no completion API. Outputs are copied out of the native logits
  buffer before any further decode.
- Exact collision scoring: teacher-forced full answer sequences per branch, the
  final answer token is not decoded (its probability is read from the previous
  position), float64 log-softmax over the whole vocabulary, unscaled by
  temperature (existing MLX/Torch semantics).
- Sequence branches are removed and the KV memory is cleared (buffers included)
  before and after every request, including exceptions; the lock is always
  released by the existing `_LockGuard`.

## Fake-native regression evidence (`tests/test_llamacpp.py`)

The fake binding asserts, per decode, that each row carries exactly one sequence
ID, that positions are contiguous per independent sequence history, that branch
sequences are copied from 0 and removed afterwards, and that a reused logits
buffer (NaN-filled between decodes) cannot leak into results. It proves: one
prefill, `max_parallel_rows == rows` with all nonzero sequence IDs, chunked /
singleton equality, context-sensitivity, exact collision math against an
independent whole-vocabulary computation, decode-failure cleanup and reuse,
boundary-mismatch and capacity fail-closed, unsupported-config rejection,
stale-compilation avoidance, calibration and lock/exception cleanup.

## Native smoke (real 7B, this machine)

Model: pinned official Qwen2.5-7B-Instruct GGUF Q4_K_M
(`bb5d59e06d9551d752d08b292a50eb208b07ab1f`, 7.6B, 2 shards, read-only private
path), llama-cpp-python 0.3.35 official Windows CPU wheel, Python 3.12.10,
Sortkasten `.venv` interpreter read-only.

Mixed 5-row schema (2 boolean, 1 enum, 2 multi rows, 1 colliding enum):
- Telemetry per request: `prefill_calls=1`, `parallel_decode_calls=2`,
  `max_parallel_rows=4`, `sequence_copies=7`, `architecture=qwen2`,
  `parameters=7615616512`.
- Reordering contexts and re-running as singleton chunks (`fields_per_batch=1`,
  `max_collision_rows=1`) produced identical selected answers; maximum
  per-field probability difference 0.00164 (float reassociation). Do not promise
  bit-identical probabilities across batch shapes.
- Wall time on this desktop CPU (not a throughput claim): ~3.4–3.8 s prefill,
  ~1.7–1.9 s suffix+collision per 105–108-token prompt, ~5.1–5.7 s total.
- The model chose a poor multi-select subset in one context — the typed output
  is not evidence of decision accuracy; no GGUF accuracy score exists yet.

## Limits / not done

- CPU only (no offload in this backend); full-attention Qwen2 GGUF only.
- No shared-prefix reuse, no cross-request KV persistence, no CUDA Graphs.
- `llama_max_parallel_sequences()` bounds rows (>= 64 on this build; not a limit
  in practice). `n_ctx` is total KV cells: prompt + longest live branch must fit.
- Raw softmax and the historical MLX 73.8% number do not describe this backend.
- The 0.3.35 binding logs loudly (stderr) at context creation; suppress at the
  app level if needed (`verbose=False` already passes through the wrapper).
