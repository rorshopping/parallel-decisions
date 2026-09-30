# GGUF native acceptance: independent harness, not yet native evidence

## Status and ownership

- Branch `test/gguf-validation`; only this note, `scripts/verify_gguf_native.py`,
  and `tests/test_gguf_acceptance_contract.py` are owned here.
- Baseline facade inspected: `_select_backend` accepts only MLX/Torch;
  `engine_llamacpp.py` is absent. Concurrent implementer's code is not in this tree.
  No production changes are proposed or made here.
- Initial deliverable is **harness contract tests/help only**. All 24 harness
  tests pass; combined harness/schema/calibration run: **81 passed** using the
  isolated Python 3.12 environment (pytest 9.1.1). No GGUF loaded, no native decode
  performed, no accuracy or latency results collected. Do not infer Windows/CUDA
  correctness from passing fake tests.
- Installed binding source inspected read-only at
  `C:\Users\Richard\Documents\Projects\sortkasten\.venv\Lib\site-packages\llama_cpp`;
  `__init__.py` reports **0.3.35**. No other private files inspected. No shared
  environment modified, no weights copied, no model discovery/download performed.

## Commands

Harness-only tests (isolated environment in this worktree):

```powershell
$root = 'C:\Users\Richard\Documents\Projects\parallel-decisions_wt\gguf-validation'
$env:PYTHONPATH = "$root\src"
$env:PYTHONDONTWRITEBYTECODE = '1'
& "$root\.venv\Scripts\python.exe" -m pytest -q -p no:cacheprovider "$root\tests\test_gguf_acceptance_contract.py"
& "$root\.venv\Scripts\python.exe" -S "$root\scripts\verify_gguf_native.py" --help
```

**After backend integration and explicit coordination of exclusive model use**,
run this acceptance command. Replace the three capitalized paths with owner-given
existing files/new report path; no model path or checksum was supplied to this
agent. `$root` must point at the integrated checkout containing this harness.
Borrowing the existing interpreter is read-only, not an installation instruction.

```powershell
$root = 'C:\Users\Richard\Documents\Projects\parallel-decisions_wt\gguf-validation'
$py = 'C:\Users\Richard\Documents\Projects\sortkasten\.venv\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $py "$root\scripts\verify_gguf_native.py" --run-native --exclusive-model-use `
  --model 'ABSOLUTE_LOCAL_MODEL.gguf' --sha256-file 'ABSOLUTE_CHECKSUM_FILE.sha256' `
  --output 'NEW_ABSOLUTE_REPORT_PATH.json' --repeats 2 --atol 0.0001
exit $LASTEXITCODE
```

Alternatively `--sha256 64_HEX_DIGITS` supplies the checksum directly. Omission
still computes and records SHA-256 but explicitly marks expected-checksum
verification false. Only the explicit model/checksum paths are read. Reports are
created exclusively: existing files are not overwritten. No mail, secrets,
external context inputs, downloads, or network requests belong in this harness.
`--exclusive-model-use` is an acknowledgement, **not a process lock or scheduler**.
Do not run it simultaneously with the engine agent or other model inference.

Exit 0 means **functional contract passed, native proof pending**, never complete
native acceptance. Exit 1 writes a failure report; CLI misuse exits 2 before
inference. The harness cannot certify exclusive GPU use or truthful telemetry.

## Functional contract covered

The script pins imports to its own checkout's `src`, records package path, source
hashes, Python/platform/binding version, model hash, checksum time, explicit lazy
`Decider.load()` wall time, per-request external wall time, engine `prefill_ms`,
`pass_ms`, `latency_ms`, chunks, compiled token sequences, distributions and raw
telemetry. `warmup=False`; the first request is identified rather than discarded.
Changing row limits may also create first-use shapes; no warmed-speedup summary
is fabricated. Model file-cache residency is uncontrolled; load is not a true
cold-machine measurement.

- One long-lived public `Decider(model_id=local_path, backend='llamacpp',
  max_fields_per_batch=8, max_collision_rows=8, warmup=False,
  config=Config(log='off'))`, then `load()`, `tokenizer`, `Schema.compile()`,
  `decide()`, and existing public result/calibrator attributes. Explicit Config
  avoids implicit calibration/config-file reads. No private runtime calls.
- Fixed synthetic four-field schema: bool, three-choice enum, three-choice multi,
  deliberately different-length field suffixes, and one collision enum: six
  expanded rows. Candidate fixture selection tries only three bounded pools and
  **fails** if actual tokenizer compilation does not expose a collision with
  distinct, nonempty, multi-token answer sequences. A common string prefix is
  removed by the compiler, so two similarly named options alone are insufficient.
- Full identical schema/prompt at `max_fields_per_batch` **8, 2, 1**; public
  `max_collision_rows` follows the same limit so exact scoring also gets a
  singleton/chunk equivalence check. Mutating these facade attributes must be
  honored by the backend; ignored row-limit/chunk telemetry is a failure.
- Two repetitions reverse configuration order. Each configuration executes
  A → changed B → longer context → A. Retained result objects are rechecked after
  subsequent calls to detect reused-logit-buffer aliasing.
- Full-schema singleton-row equivalence is the meaningful comparison. Reducing
  the schema itself changes the prompt! A separate one-field boolean call checks
  singleton schema shape, but its probability is not compared to the large schema.
- Strict schema keys/types, enum membership, boolean bool (not text/int), unique
  ordered multi values, no approximate flags, finite [0,1] scores, exact allowed
  distribution keys, normalized bool/enum distributions, selected maxima,
  distribution/confidence agreement, alternatives, raw-confidence preservation,
  result JSON helpers, row count and expected chunks.
- Multi distributions are independent inclusion marginals: **not sum-to-one**.
- Temperature-2 synthetic calibrator checks selected-value invariance, original
  confidence preservation and agreement with public Calibrator transforms.
  This is not fitted calibration and says nothing about correctness frequency.
- All distributions must agree within explicit absolute tolerance (default 1e-4),
  all selected values exactly. Near ties may fail; inspect evidence, don't hide
  flips or auto-relax tolerance. B need not select a different answer: no label
  accuracy assumption is valid here. A/B distribution sensitivity is reported,
  not enforced; zero sensitivity leaves stale-context detection inconclusive.

A serial fake intentionally passes these checks in pytest. That test protects
against mistaking an output contract or a claimed `chunks=1` for native proof.

## Installed native API findings / requirements for implementer

Read-only source references (line positions refer to installed binding 0.3.35):

1. `llama_cpp.py:3075` has `llama_batch_init(n_tokens, embd, n_seq_max)`;
   initialize all members and pair with `llama_batch_free`. Batch `n_seq_max`
   means **sequence IDs assignable per token**, whereas context params
   `n_seq_max` controls concurrent states/sequences. Allocate capacities for the
   immutable root plus active suffix/collision branches, along with sufficient
   `n_ctx`, `n_batch`, `n_ubatch`. Do not confuse these capacities.
2. `llama_get_memory(ctx)` provides the memory handle;
   `llama_memory_seq_cp(mem, src, dst, p0, p1)` at 2399 copies sequence membership
   over a position range. Root sequence 0 owns one prompt prefill; create distinct
   suffix sequence IDs from its [0, prompt_length) range and preserve the root.
   Equivalent shared-token multi-ID membership is possible but must be traced.
3. `llama_memory_seq_rm`, `llama_memory_seq_keep`, `llama_memory_clear`, and
   `llama_memory_seq_pos_min/max` support cleanup and state checks.
   `_internals.LlamaContext.kv_cache_seq_rm` clamps negative sequence IDs to 0!
   Do not assume wrapper `seq_rm(-1,...)` clears all sequences. Clear/reset between
   changed requests and delete old branch tails before reusing IDs. Check return
   values and error paths so no partially processed batch contaminates a retry.
4. `llama_decode(ctx, batch)` at 3135 operates over a **single token batch with
   explicit `token`, `pos`, `n_seq_id`, `seq_id`, `logits` per token**. It returns
   0 on success; 1 KV-slot failure, 2 abort, negative errors. Abort/fatal paths can
   retain processed ubatches. Pure KV copies followed by a serial decode loop
   still do not prove batched suffix evaluation.
5. `_internals.LlamaBatch.add_sequence` at 518 sets positions starting at **0**,
   not prompt_length. `set_batch` fixes sequence ID to **0**. Neither helper as-is
   correctly creates independent suffix branches after a shared prefill. Populate
   native arrays or deliberately adjust every suffix position to prompt_length +
   row_offset; preserve per-row ownership, causal masking and unequal lengths.
   `llama_batch_get_one` fixes sequence 0 and is not the multi-branch API.
6. Request logits at each actual decision position (not a padded final column).
   Exact collisions need logits at **every answer predecessor**, including the
   last suffix token for the first answer token; sum full-vocabulary normalized
   log-probabilities over the complete compiled continuation, then normalize
   across choices. Don't score only shared first tokens, length-average sequences,
   or sample prose/JSON. Different answer lengths and sequence-ID isolation matter.
7. `llama_get_logits` at 3262 packs only logits-enabled outputs. Native header
   comments for `llama_get_logits_ith` at 3274 describe positive **batch token
   indices mapped by output_ids**, negative reverse-output indices, and NULL for
   invalid IDs. Its Python docstring is less precise. Do not index sparse outputs
   as a dense `[n_tokens, vocab]` matrix; copy requested rows before the next decode.
8. `llama_synchronize(ctx)` at 3245 waits for GPU completion; logit getters also
   synchronize. Prefill timing often has no logit read, so explicitly synchronize
   before ending prefill time. Suffix time must include collision scoring and its
   synchronization. Do not measure merely enqueued CUDA work.

These are **source/API observations, not successful DLL symbol calls or proof
that the implementation uses them**. Binding version string alone does not prove
CUDA offload, DLL compatibility, wheel provenance, or native GPU execution.

## Separate native architecture + instrumentation acceptance gate

An independent reviewer must inspect the integrated backend and capture native
boundary evidence during the public Decider workload before promoting acceptance:

- Record exact engine commit/source hashes, binding/DLL build and device/offload
  facts. Inspect that there is no full-prompt-per-field or high-level
  `create_completion` / `generate` / sampling loop. Candidate-only outputs can
  still be produced serially; no-free-text assembly alone proves nothing native.
- For every request, trace native decode **phase**, request/chunk IDs, token
  count, positions, sequence memberships, logits-enabled token indices, root
  length, and return code. Trace sequence copy/remove/clear ranges and IDs.
  Synthetic inputs only; never turn this into production-mail logging.
- Show one logical prompt prefill (it may span multiple n_batch decode calls),
  root tokens processed once, not once per field. Record prompt token IDs/hashes
  from the actual tokenizer and verify prefix boundaries/BOS exactly once.
- Show a suffix `llama_decode` batch containing **at least two distinct branch
  sequence IDs** at prompt-relative suffix positions for the 8-row-limit case.
  With available token capacity, the small six-row fixture should fit one logical
  suffix batch. Multiple native ubatches are not repeated full prompt inference.
- Show the branch copies/membership inherit only the root prompt, and no suffix
  can attend to another row's tokens. Verify root max position before/after each
  suffix/collision chunk, cleanup for changed context/length, bounded live seqs,
  and genuine singleton-row decode under limit 1.
- Trace collision candidate branches and every selected predecessor-logit index.
  Add an independent scalar teacher-forcing/full-sequence oracle or controlled
  deterministic-logit native test; row-limit equivalence alone cannot catch an
  exact-scoring algorithm that is consistently wrong in every mode.
- Record synchronized phase boundaries (including collision work), not just
  backend-reported counters. A telemetry dictionary can claim anything: review
  where the numbers come from and corroborate at native calls. A trace collector
  can wrap the native boundary without changing production code, but must not
  inject generated tokens or alter results. This collector is **not implemented**
  here because the integration architecture is absent.

## Remaining gaps / honest release boundary

No native inference, native call trace, DLL capability check, actual GPU placement,
real collision oracle, fresh-runtime changed-context oracle, model checksum
verification, measured load/prefill/pass performance, calibrated domain evaluation,
long-context capacity/OOM recovery test, prepared-prefix support, or lock/concurrency
stress test has been executed. No mail or app integration is exercised. Trace review
is needed to verify no free-text generation internally; typed output alone cannot
rule out generation followed by coercion. The harness is a necessary functional
layer, not a complete native proof or accuracy benchmark. Do not stall/poll waiting
for the concurrent implementation: merge these owned files and run the gated steps
when the owner schedules exclusive native acceptance.
