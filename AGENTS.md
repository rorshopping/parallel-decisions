# AGENTS.md — working in this repository

Notes for AI coding agents (and humans) who will extend or integrate this package.

## What this is

A small library that turns a local MLX LLM into a typed decision engine:
context + schema in, typed answers with probabilities out, computed as batched
forward passes. It is a packaged, cleaned-up version of the "parallel constrained
decoding" prototype, with the best-performing local model we measured
(Qwen2.5-7B-Instruct-4bit, 73.8% agreement on TypeSafe's public eval suite).

Read `README.md` first for the user-facing API. This file is about modifying the code.

## Architecture in one screen

```
schema.py   Field/Schema: parse user dict -> validate -> compile(tokenizer)
            -> CompiledField {suffix_tokens, candidate_ids, sequences, collision}
prompts.py  build_prompt(context, schema): ChatML prompt, descriptions + option lists
engine.py   Decider: load model once; decide() = prefill once, then per chunk:
            repeat KV cache N times -> one batched pass over field suffixes ->
            slice logits per field -> FieldValue. Colliding fields go through
            _resolve_collisions(): one extra batched pass scoring full sequences.
cli.py      `pd validate` / `pd decide`
```

Key invariants:

- **The JSON is never generated.** Values are chosen from `candidate_ids` and
  assembled in `DecisionResult`. Never add a code path that lets the model write
  free text into a field.
- **One prefill per `decide()` call.** All passes reuse the same `prompt_cache`.
- **Chunking exists to bound memory.** KV memory = `fields × prompt_tokens × bytes/token`.
  `_auto_chunk_size()` estimates it from the model config; if a chunk still fails
  (MLX raises), halving the chunk size and retrying is the intended recovery.
- **Collisions are exact, not guessed.** `_resolve_collisions` computes true
  sequence log-probabilities; do not reintroduce the old "fallback to first choice"
  behaviour.

## Environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest                      # schema tests, no model needed
.venv/bin/python examples/basic.py    # needs the model (first run downloads ~4.3 GB)
```

The machine this was developed on: MacBook Air M5, 16 GB, macOS 26.5. Do not
assume more memory; respect the chunking design.

## Testing policy

- `tests/` must stay runnable **without** MLX or a model download (stub tokenizers
  are fine, see `test_compilation_and_collision_detection`).
- Model-dependent checks belong in `examples/` or as opt-in scripts, not in pytest.
- If you change `engine.py`, run a real smoke test (examples/basic.py) and check
  the timing printed at the end: prefill should scale with context, the batched
  pass should stay roughly flat as fields are added.

## Extension points (in order of usefulness)

1. **More field types.** The method supports any bounded answer set. Numeric
   buckets and small string sets map onto enum already; multi-token answers work
   via `sequences`. A true scalar `score` type would need a different head
   (softmax over digit tokens), out of scope for now.
2. **Batching several contexts.** `decide_many` is a loop; true batching across
   contexts (shared suffix pass, different prompts) would help pipelines.
3. **Confidence calibration.** `probability` is raw softmax. There is a measured
   reliability table in `CALIBRATION.md`: ECE ≈ 0.094, almost everything lands in the
   0.9+ bucket, and wrong answers are nearly as confident as right ones. The intended
   improvement is temperature scaling (`Decider(calibration=...)`) fitted on a labelled
   set. Do not present these numbers as calibrated without doing that work.

## Things not to do

- Do not add network calls beyond the initial model download.
- Do not vendor the model weights into the repo.
- Do not silently change the prompt format: accuracy numbers in the README were
  measured with `prompts.build_prompt` as-is.
- Do not add dependencies beyond `mlx` / `mlx-lm` without a strong reason; the
  server example deliberately uses the standard library only.

## Provenance

- Method origin: `harshatheg/Qwen-2.5-1B-RLCD` (Apache-2.0) community artifact.
- Evaluation and model choice: `jev-on-a-laptop` repo (`evals/`), 2026-09.
- Concept (Jev / TypeSafe AI): unrelated to this package; no affiliation.
