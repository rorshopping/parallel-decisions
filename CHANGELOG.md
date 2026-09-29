# Changelog

All notable changes to `parallel-decisions`. The package follows
[Semantic Versioning](https://semver.org/); `0.x` means the API can still move.

## 0.3.1 — unreleased

Integration-ready packaging for the torch backend.

### Fixed

- **Collision scoring is trie-batched (torch).** A field whose answers share a
  first token previously scored one row per candidate through the prompt cache:
  the 240-choice browser action space needed 248 rows in 33 batched passes
  (2.7 s at 7k prompt tokens). Only trie nodes with children are scored now —
  25 rows in 4 passes — cutting the pass stage to 0.38 s; the summed sequence
  log-probabilities are unchanged and pinned by an equivalence test against the
  implementation this replaced.
- **Torch prefill is now chunked, bounding prompt memory on GPUs without an
  efficient SDPA kernel (Turing and older).** A long prompt previously ran one
  math-backend attention pass — 7.2k tokens took 11.1 s and peaked at 7.4 GiB on
  an RTX 2060 SUPER, which Windows can silently page into shared memory for a
  10–100× slowdown. Prompts are now read in `torch_prefill_chunk` segments
  (default 2048; `PD_TORCH_PREFILL_CHUNK`, `0` disables) through the same KV
  cache: 0.70 s / 1.6 GiB at 1024-token segments in the same measurement.
  Last-token logits differ by at most 0.04 from fp16 reassociation.
- **Torch backend installs and defaults are now coherent.** A fresh
  `pip install ".[torch]"` provides torch, transformers and accelerate; MLX
  dependencies are conditioned to Apple Silicon macOS, so Windows/Linux users no
  longer get an unusable MLX runtime. With no model configured, the torch backend
  defaults to `Qwen/Qwen2.5-0.5B-Instruct` (MLX keeps the legacy 7B 4-bit
  `DEFAULT_MODEL`); explicit/model config still wins.
- `examples/serve.py` now accepts documented schema wrappers and serialized field
  lists, validates context/body, and returns JSON 400/503 errors; `pd` keeps
  JSON-only stdout with verbose logs on stderr, rejects empty contexts before
  loading a model, and uses clean exit statuses.
- Added model-free regression tests for default selection, emitted metadata,
  loopback HTTP, lock contention, CLI JSON and MCP dictionary schemas.

### Added

- **Laya backend, and it is the default model path when installed.** `backend =
  "laya"` drives the non-autoregressive `convaiinnovations/laya` decision model
  the same way `SnipLedger.AI` does (`snipleger_ai/backends/laya.py`): each schema
  field becomes a typed Laya question (`enum` -> `choice`, `boolean` -> `noul`,
  `multi` -> one question per choice at `"<name>[<i>]"`), one
  `Router.predict`/`predict_batch` call answers them all, and probabilities are
  Laya's calibrated `answer_confidence` (`raw_probability` equals it until you
  fit your own calibrator on top). Install with `pip install
  "parallel-decisions[laya]"`; model assets come from the Hugging Face cache.
  `Decider()` then resolves `auto` to Laya; `english`, `multilingual`,
  `typed-decisions` or a standalone `convaiinnovations/laya-*` repo pin a
  checkpoint; `laya_device` / `PD_LAYA_DEVICE` select CUDA or CPU;
  `decide_many` is one batched call. The MLX and Torch causal engines remain
  selectable (`backend="mlx"` / `"torch"`), and schema/calibration code is
  unchanged.
- `INTEGRATION.md`: an application-integration guide (Python lifespan, schema
  and prefix use, the local HTTP contract, CLI, failure handling, and confidence
  caveats).

### Changed

- **`backend = "auto"` prefers Laya when the `laya` package is importable**;
  the previous platform default (MLX on Apple Silicon, Torch elsewhere) applies
  when it is not. `pd validate --check-tokens` prints why the causal-token lint
  is skipped on the Laya backend, and `pd decide` names Laya's confidence for
  what it is (calibrated on Laya's benchmark, not on your domain). Per the
  repository convention, the version is bumped in a separate release commit.

## 0.3.0 — 2026-09-16

The torch backend: the same parallel constrained decoding on NVIDIA GPUs and CPU.

### Added

- **Torch shared-prefix reuse**: `prepare`, `decide_with_prefix` and
  `decide_many(shared_prefix=True)` prefill a schema once. Private cache copies,
  per-context full-tokenization validation, locked prepare/use/release and truthful
  fallback telemetry protect reuse. Model-free tiny CPU-model regression tests
  cover collisions, chunks, BOS, boundary merges, calibration and cache integrity.
- Synchronized CUDA phase timing and a paired, raw-JSON prefix benchmark. On one
  RTX 2060 SUPER fp16 workload: 88.5 → 61.9 ms/request; results are workload-specific.
- **Torch backend (`parallel_decisions.engine_torch`)**: a PyTorch port of the
  decoding scheme — one prefill, one broadcast KV cache, one batched pass, exact
  collision resolution — for CUDA GPUs and CPU-only machines. Port origin:
  `harshatheg/Qwen-2.5-1B-RLCD` commit `031d1a8` (2026-09-16, Apache-2.0),
  integrated here on top of the package's schema, collision and calibration
  machinery rather than vendored (the upstream port lacks collision handling,
  multi-select and chunking).
- **`Decider(backend=...)`** / `pd.toml backend = "auto" | "mlx" | "torch"` with
  `PD_BACKEND` env override; `torch_dtype` / `torch_device` tune the torch side.
  `auto` keeps MLX on Apple Silicon and selects torch (CUDA if present) elsewhere,
  replacing the hard non-arm64 failure with a working backend.
- Windows support for the pure-Python surface: `_physical_ram_bytes()` now uses
  `GlobalMemoryStatusEx`, so memory clamping and the test suite work off macOS.
- **Experimental CUDA graph replay for the torch backend** (`cuda_graph` in
  `pd.toml`, `PD_CUDA_GRAPH`, off by default): repeated suffix passes with stable
  shapes are captured once (StaticCache, eager prefill outside the graph) and
  replayed with fresh staged inputs; any capture/replay failure falls back
  permanently to the eager path. Off by default; helps shape-stable workloads.

### Changed

- Tests that require `mlx`/`mlx_lm` skip cleanly when the package cannot import
  (it has no Windows build) instead of erroring through `pytest.importorskip`.

## 0.2.0 — 2026-09-16

Calibration lands: confidences become numbers you can threshold on.

### Added

- **Calibration** (`parallel_decisions.calibration`): `Calibrator` with three
  post-hoc methods — `temperature` (`p' ∝ p**(1/T)`, fitted by NLL), `platt`
  (IRLS logistic fit on `logit(confidence)`), and `isotonic` (pool-adjacent-violators).
  All three preserve the top choice, so accuracy is unchanged and only the reported
  confidence moves.
- **`fit_calibration()`**: compares raw vs all three methods by label-stratified
  k-fold cross-validation, picks the winner by held-out ECE (equal-count bins by
  default), and refuses to ship a method that loses to raw softmax out of sample.
- **`Decider(calibration=...)`**: accepts a path, dict or `Calibrator`. Calibrated
  results carry both `probability` (calibrated) and `raw_probability`.
- **`pd calibrate`** and `tools/fit_calibration.py`: fit from a labelled file, print
  the cross-validated comparison, the before/after reliability table and the routing
  sweep, then write the calibrator JSON.
- **`examples/routing.py`**: act / review / refuse policy derived from data, with the
  act-bucket error rate and a Wilson interval, measured out of sample.
- **Multi-select fields** (`"type": "multi"`): any subset of the choices, decided as
  one independent yes/no question per choice in the same batched forward pass. The
  value is a list; `probability` is the confidence of the *set* (its weakest member);
  `distribution` holds per-choice P(include).
- **`pd.toml` + `PD_*` environment variables** (`parallel_decisions.config`), with
  `pd config` to show what is in effect. Precedence: argument > env > file > default.
- **Structured logging**: `PD_LOG=json` emits one JSON line per load and per decision
  to stderr (fields, rows, chunks, prompt tokens, latencies, calibration state).
- **Concurrency safety**: a lock around the model, `lock_timeout_s` to bound the wait,
  and a clear `ConcurrencyError` instead of a Metal failure. `examples/serve.py` now
  queues requests (and returns 503 instead of crashing when it cannot).
- **Graceful memory recovery**: if a batched pass fails with a memory error, the
  chunk size halves and the pass retries instead of the whole call dying.
- **Platform guard**: a clear `UnsupportedPlatformError` on non-arm64 (override with
  `PD_ALLOW_NON_ARM=1`).
- `PD_ALLOW_NON_ARM`, `ConcurrencyError`, `UnsupportedPlatformError` in the public API.

### Changed

- `Decider` constructor arguments default to `None` and resolve from `pd.toml`/env,
  then the built-in defaults. Explicit arguments still win.
- The KV budget is clamped so weights plus the broadcast stay under 65% of physical
  RAM. Exceeding it does not raise — macOS swaps and calls get several times slower
  (measured: a 48-row invoice case went from ~11 minutes to 13+ minutes stalled with
  15 GB of swap).
- `mlx` is imported lazily, so schema, calibration, config and lint work on a machine
  without it.
- The version is read from `parallel_decisions.__version__`; the wheel previously
  built as 0.1.0 while the package reported 0.2.0.
- `FieldValue` gained `distribution` (every allowed answer, not just the runner-up),
  `raw_probability` and `calibrated`.
- `pd validate` now reports multi-select fields and per-row collisions.

### Fixed

- **Hybrid architectures work.** The KV broadcast only repeated `keys`/`values`, so
  models that keep a convolutional state on linear-attention layers (Qwen3.5) failed
  with a shape error. Every per-sequence array in the cache is now broadcast, read
  from instance storage only — `KVCache.state` is a *view* for serialisation, and
  writing through it silently truncated the cache.
- The per-token cache measurement sampled prompts where `mlx-lm`'s 256-token
  preallocation makes every size identical, so it reported no growth. It now spans
  512–3072 tokens and reports the constant per-row state separately (49 MB for
  Qwen3.5-4B), which chunk sizing counts too.
- Adaptive ECE binning was order-dependent when many answers shared a confidence;
  bin edges are now empirical quantiles, so the number no longer depends on shuffling.
- Multi-class temperature scaling is a true renormalisation over all classes (the
  earlier draft used a binary rescale of the top choice, which is wrong for >2 choices).

## 0.1.0 — 2026-09-16

First tagged release: typed, locally-run decisions on Apple Silicon.

- `Decider.decide(context, schema)` — one prefill, one batched forward pass per chunk,
  logits sliced to each field's allowed answers. JSON is assembled, never generated.
- Exact sequence scoring for fields whose choices share a first token.
- `pd validate` / `pd decide`, `examples/serve.py` (stdlib HTTP), `examples/basic.py`.
- Memory-aware chunking (`fields × context` KV broadcast).
- Reference numbers: 73.8% agreement with TypeSafe's reference consensus on their
  public cases (two-run mean, 7B 4-bit, out of the box).
