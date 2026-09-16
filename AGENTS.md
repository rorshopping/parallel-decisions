# AGENTS.md — working in this repository

Notes for AI coding agents (and humans) who will extend this package.

## What this is

A small library that turns a local MLX LLM into a typed decision engine:
context + schema in, typed answers with probabilities out, computed as batched
forward passes. It is a packaged, validated version of the "parallel constrained
decoding" prototype, with the best-performing local model we measured
(Qwen2.5-7B-Instruct-4bit, 73.8% agreement on TypeSafe's public eval suite), plus
confidence calibration so the probabilities can be thresholded.

Read `README.md` first for the user-facing API. This file is about modifying the code.

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
  `engine.py`; schema/calibration/config/lint work on any machine.
- **Version has one source of truth**: `__version__` in `__init__.py`, read by
  pyproject's `dynamic` version. `tests/test_packaging.py` enforces it.

## Environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest                      # 123 tests, no model needed
.venv/bin/pd validate examples/fraud.json --check-tokens  # tokenizer only, no model
.venv/bin/pd decide --schema examples/fraud.json --context "..."   # needs the model
```

`tests/test_docs.py` keeps the README honest (its python blocks must parse, its
commands must exist, every `pd.toml` key must be documented), and
`tests/test_examples.py` keeps `examples/` runnable and collision-free.

The machine this was developed on: MacBook Air M5, 16 GB, macOS 26.5. Do not
assume more memory; respect the chunking design.

## Testing policy

- `tests/` must stay runnable **without** MLX, a model download, or the research
  tree present (the prompt-equivalence test skips itself when the tree is absent).
  Stub tokenizers are fine and encouraged — `tests/test_schema.py::_VocabTokenizer`
  is a longest-match tokenizer that models space attachment, which is the thing
  that actually matters for the candidate logic.
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
2. **True cross-context batching.** `decide_many` is a loop. Batching contexts with
   a shared schema would amortise prefill across requests — the biggest remaining
   performance win, since prefill is 84.6% of case wall time.
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
- Evaluation and model choice: `rlcd-research` (`evals/`), 2026-09.
- Concept (Jev / TypeSafe AI): unrelated to this package; no affiliation.
