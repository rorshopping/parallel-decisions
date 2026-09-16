# Changelog

All notable changes to `parallel-decisions`. The package follows
[Semantic Versioning](https://semver.org/); `0.x` means the API can still move.

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
