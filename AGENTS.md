# AGENTS.md — working in this repository

Notes for AI coding agents (and humans) who will extend this package.

## What this is

A small library that turns a local model into a typed decision engine: context +
schema in, typed answers with probabilities out — one batched forward pass for
Laya, batched constrained decoding for the causal engines. It is a packaged,
validated version of the "parallel constrained decoding" prototype, with the
best-performing local causal model we measured (Qwen2.5-7B-Instruct-4bit, 73.8%
agreement on TypeSafe's public eval suite), confidence calibration so the
probabilities can be thresholded, and — since 0.3.1 — the Laya
non-autoregressive decision model as the default path when its package is
installed.

Read `README.md` and `GPU_SETUP.md` first for the user-facing API and Windows
setup. This file is about modifying the code. The project also has a Torch
CUDA/CPU backend, shared-prefix reuse and opt-in StaticCache CUDA Graph replay.
The older MLX accuracy numbers below do not describe the 0.5B CUDA latency model
or the Laya checkpoints.

## Windows / CUDA handoff (2026-09-17)

- Release checkout: `C:\Users\Richard\Documents\Projects\parallel-decisions`.
- Research mirror: sibling `jev-on-a-laptop`; private research: `rlcd-research`;
  HF source reference: `rlcd-upstream`.
- Reusable environment currently lives at
  `C:\Users\Richard\Documents\Projects\parallel-decisions_wt\gpu\.venv`.
  It is an editable install pointing at that worktree, NOT whichever checkout
  happens to be the shell working directory. Always pin and print the import path:

```powershell
$env:PYTHONPATH = "C:\Users\Richard\Documents\Projects\parallel-decisions\src"
$py = "C:\Users\Richard\Documents\Projects\parallel-decisions_wt\gpu\.venv\Scripts\python.exe"
& $py -c "import parallel_decisions.engine_torch as m; print(m.__file__)"
& $py -m pytest -q
exit $LASTEXITCODE
```

Set `PYTHONPATH` to **your** worktree's `src` when working in isolation. Do not
reinstall the shared venv against a different worktree while another agent uses it.
PowerShell commands may print harmless stderr warnings as error records; inspect
`$LASTEXITCODE` immediately after Python. Do not truncate failing tracebacks or use
`cmd /c "... & echo %ERRORLEVEL%"` (expansion can report a stale exit status).
The latest documented suite is 273 passed / 3 skipped on this CUDA machine (the
committed tree; an untracked calibration test file in the working tree adds ~28);
counts vary with optional dependencies and hardware. A pass using another checkout
is not verification of your changes.

The shared venv does **not** have `laya` installed; the Snipledger2 venv does
(`C:\Users\Richard\Documents\Projects\Snipledger2\src\SnipLedger.AI\.venv`), and
the `convaiinnovations/laya` checkpoints are already in the Hugging Face cache.
A real end-to-end Laya check (no download) is therefore:

```powershell
$env:PYTHONPATH = "C:\Users\Richard\Documents\Projects\parallel-decisions\src"
& "C:\Users\Richard\Documents\Projects\Snipledger2\src\SnipLedger.AI\.venv\Scripts\python.exe" `
  -m pytest -q tests/test_laya_engine.py   # passes both with and without laya installed
& "C:\Users\Richard\Documents\Projects\Snipledger2\src\SnipLedger.AI\.venv\Scripts\python.exe" `
  -c "from parallel_decisions import Decider, Schema; d = Decider(); print(d.backend, d.model_id)"
```

### Torch-specific implementation invariants

- `engine_torch.py`: `TorchRuntime`, `prepare_torch`, `decide_torch`,
  `TorchCudaGraphCache`; facade/config plumbing remains in `engine.py` / `config.py`.
- `prepare`/use/release share the model lock. Verify actual full-tokenization prefix
  IDs per context; slice the remainder from full tokenization to avoid double BOS.
- Never shallow-copy only a cache wrapper: HF layer updates can mutate the caller's
  cache. Plain `DynamicLayer` can share immutable tensor contents with copied layer
  objects; unknown mutable-state layouts require conservative isolation. Test
  repeated use, a singleton row, collisions and chunks with changed context.
- Graphs are off by default (`cuda_graph`, `PD_CUDA_GRAPH`). Prefill stays eager;
  repeated suffix shapes use StaticCache and staged lengths/positions/masks/KV.
  Stage fresh inputs before every replay, clone outputs, and test changed lengths
  within the same bucket. Tests live in `test_cuda_graph.py` / `test_torch_prefix.py`.
- Graph capture supports Qwen2 full attention (eval, eager/SDPA), at most four
  shapes. Failures disable graphs for the runtime and record `disable_reason`.
  Respect the VRAM guard; do not kill the user's desktop/ASR processes to benchmark.
- CUDA timings must synchronize. Separate download/load, warmup, capture,
  preparation, prefill, suffix and total request time. Compare matching schemas,
  dtype, model and lengths, alternate paired order, preserve raw JSON. Do not
  multiply speedups from different workloads or claim browser-click speedups from
  HTTP requests plus DOM updates. `GPU_SETUP.md` is the evidence index.
- Current limitations: Torch fixed-row chunking does not use MLX's adaptive
  memory budget; reported bf16 availability is not proof of native bf16
  performance on Turing GPUs. The packaging limitations (no torch extra, MLX
  default model off Mac) were fixed in 0.3.1; document any new gaps, don't
  silently imply they are fixed. Explicit model/device/fp16 settings are used
  in the GPU guide.
- A 7B fp16 model does not fit 8 GB VRAM. The tested latency model is 0.5B;
  fp16 is the starting point on this Turing GPU, not auto-selected bf16.

The original sections below describe the MLX architecture unless stated otherwise.

## Architecture in one screen

```
schema.py       Field/Schema: parse user dict -> validate -> compile(tokenizer)
                -> CompiledField {suffix, suffix_tokens, candidate_ids, sequences,
                                  collision, choice_index}
                  candidate_ids are read off encode(suffix + answer): the token the
                  model actually emits next. `collision` is exactly "two answers emit
                  the same next token". A `multi` field compiles to one boolean row
                  per choice at keys `name[i]`.
calibration.py  Calibrator (temperature / platt / isotonic) + metrics (ECE over
                equal-width and equal-count bins, AUROC, Brier, top-NLL, Wilson,
                risk_coverage) + fit_calibration() which compares methods by
                label-stratified k-fold CV and only ships one that wins out of sample.
engine.py       Decider: load (RAM-aware KV budget), decide() = one prefill, then per
                chunk: repeat KV cache N times -> one batched pass over field suffixes
                -> slice logits -> FieldValue. Multi rows fold into list values.
                Colliding fields go through _resolve_collisions(): one extra batched
                pass scoring the full answer sequences. Then optional calibration.
engine_laya.py  The default model path when the `laya` package is installed.
                plan_schema(): Field -> Laya question (enum -> choice, boolean ->
                noul, multi -> one noul per choice at `name[i]`). LayaRuntime wraps
                laya.Router (predict / predict_batch); values_from_answers() maps
                answers back to FieldValue (probability = Laya's calibrated
                answer_confidence; multi rows are P(include) and fold through
                Decider._assemble_multi). No torch import at module scope.
config.py       pd.toml + PD_* env vars; explicit argument > env > file > default.
lint.py         collision lint with concrete rename advice (no model load needed).
prompts.py      build_prompt: ChatML, descriptions + option lists, one `{`.
cli.py          `pd validate` / `decide` / `calibrate` / `config`.
```

## Key invariants

- **The JSON is never generated.** Values are chosen from candidate ids and
  assembled in `DecisionResult`. Never add a code path that lets the model write
  free text into a field.
- **One prefill per `decide()` call.** All passes reuse the same `prompt_cache`.
- **The prompt is a measured artifact.** `tests/test_prompt_equivalence.py` asserts
  `build_prompt` is byte-identical to the research runner
  (`rlcd-research/source/Qwen-2.5-1B-RLCD/core/engine_mlx.py`) on all 373 published
  question slots. A silent prompt change invalidates every number in the READMEs.
  Do not "tidy" the prompt format without updating those numbers.
- **Candidates and collisions are exact.** `CompiledField._next_token` encodes
  `suffix + answer`; when the tokenizer merges across the boundary (normal for
  booleans — the suffix ends in a space token) it falls back to the answer's own
  first token, which is what the reference implementation scored. See the tests in
  `tests/test_schema.py` before changing this.
- **Calibration never changes an answer.** Every calibrator is monotone in the top
  confidence, so `probability` moves and `value` does not. Do not add a calibrator
  that can reorder answers without renaming it and documenting the break.
- **Memory is `fields × context`.** The chunk size comes from
  `_auto_chunk_size()`, using a *measured* per-token cache growth
  (`_measure_kv_bytes_per_token`), clamped against physical RAM
  (`_clamp_memory_budget`, `MEMORY_HEADROOM`). Do not raise the budget past the
  clamp: exceeding it does not raise, macOS swaps, and calls get several times
  slower (measured: 13+ minutes for one 48-row case with 15 GB of swap).
- **One model, one call at a time.** `decide()` holds a lock; the second caller
  waits or raises `ConcurrencyError` after `lock_timeout_s`.
- **`import parallel_decisions` must not need MLX.** `mx` is a lazy proxy in
  `engine.py`; schema/calibration/config/lint work on any machine. The same holds
  for Laya and Torch: `engine_laya.py` imports neither at module scope, and the
  `laya` package is only reached through `LayaRuntime._build_router()`.
- **The Laya answer contract is exact.** `probability` is the model's
  `answer_confidence` (its calibrated `max(p)`), `distribution` is the raw
  per-answer distribution, and a `multi` row's probability is P(include) so it
  folds through the same `_assemble_multi` as the causal rows: decisions use raw
  P(true) >= 0.5, calibration uses the distribution. A missing or unknown answer
  raises `LayaDecisionError`; never invent a value. Automatic routing and
  checkpoint pinning live in `engine_laya.pinned_model` and go through Router's
  own aliases; do not re-implement its model table. Tests must stub the router
  (`tests/test_laya_engine.py`) — no download, no torch.
- **Version has one source of truth**: `__version__` in `__init__.py`, read by
  pyproject's `dynamic` version. `tests/test_packaging.py` enforces it.

## Environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,laya]"   # drop ,laya for the causal backends
.venv/bin/pytest                      # ~300 tests, no model needed
.venv/bin/pd validate examples/fraud.json --check-tokens  # tokenizer only, no model
.venv/bin/pd decide --schema examples/fraud.json --context "..."   # needs the model
```

`pd validate --check-tokens` prints a note and skips the token lint on the Laya
backend (there are no candidate tokens); the check is causal-LM specific.

`tests/test_docs.py` keeps the README honest (its python blocks must parse, its
commands must exist, every `pd.toml` key must be documented), and
`tests/test_examples.py` keeps `examples/` runnable and collision-free.

The machine this was developed on: MacBook Air M5, 16 GB, macOS 26.5. Do not
assume more memory; respect the chunking design.

## Testing policy

- `tests/` must stay runnable **without** MLX, PyTorch, `laya`, a model download,
  or the research tree present (the prompt-equivalence test skips itself when the
  tree is absent). Stub tokenizers are fine and encouraged —
  `tests/test_schema.py::_VocabTokenizer` is a longest-match tokenizer that models
  space attachment, which is the thing that actually matters for the candidate
  logic. Stub the Laya router the same way (`tests/test_laya_engine.py`), and keep
  `auto`-backend tests deterministic with
  `monkeypatch.setattr(engine_laya, "laya_available", ...)` so a machine with the
  package installed answers the same as one without.
- Model-dependent checks belong in `examples/` or as opt-in scripts, not in pytest.
- If you change `engine.py`, run `examples/basic.py` (or `pd decide`) and check the
  timings: prefill scales with context, the batched pass stays roughly flat as
  fields are added, and `chunks` should stay at 1 for small workloads.
- New behaviour needs a test that fails before the change. The calibration module
  has one test per method plus one per failure mode (identity, degenerate fits,
  ordering); match that standard.

## Extension points (in order of usefulness)

1. **More labelled data.** Everything about calibration is limited by it. The
   routing demo is only meaningful with a few hundred labelled rows from the domain
   it will run in. See `evals/domains/*` in the research tree for the format.
2. **True cross-context batching.** `decide_many` is a loop; `shared_prefix=True`
   removes the schema-block prefill but each context still gets its own pass for its
   own text. A single batched forward pass over several contexts needs per-row
   sequence lengths in the KV cache, and `mlx-lm`'s forward path takes no attention
   mask, so a padded batch attends over the padding. Doing this properly means
   padding-aware attention (a mask, or a custom per-row attention) — a real
   engineering task, not a loop rewrite, and the largest remaining performance win
   since prefill is 84.6% of case wall time.
3. **Order-robust choice decisions.** The model picks the first-listed choice in 82%
   of choice fields, and accuracy is 96.7% when the truth is listed first versus
   24.5% when it is not. Scoring each option under a few rotations of the option
   list and averaging would reduce that; it costs extra prefills, which is why it is
   not implemented.
4. **Numeric outputs.** The method covers bounded choices. A scalar `score` head
   would need a different technique (see the roadmap).

## Things not to do

- Do not add network calls beyond the initial model download.
- Do not vendor the model weights into the repo.
- Do not change the prompt format silently (see invariants).
- Do not add dependencies beyond `mlx` / `mlx-lm` without a strong reason. The
  server example and the MCP example use optional imports, not dependencies.
- Do not present raw softmax numbers as calibrated, or calibrated numbers without
  the data they were fitted on (`Calibrator.meta` records both).

## Direction

The phased plan for this package lives in the research tree at
`~/Documents/projects/rlcd-research/ROADMAP.md`, with measured results in
`evals/analysis/REPORT.md`, `evals/RESULTS.md` and `CALIBRATION.md`.

## Provenance

- Method origin: `harshatheg/Qwen-2.5-1B-RLCD` (Apache-2.0) community artifact.
- Default model path: `convaiinnovations/laya` (Apache-2.0 package and
  checkpoints, Convai Innovations); integration mirrors `SnipLedger.AI`'s
  `snipleger_ai/backends/laya.py`.
- Evaluation and model choice: `rlcd-research` (`evals/`), 2026-09.
- Concept (Jev / TypeSafe AI): unrelated to this package; no affiliation.
