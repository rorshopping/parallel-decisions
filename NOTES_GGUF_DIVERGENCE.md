# Windows GGUF divergence investigation (2026-09-17)

## Update (2026-09-18): functional gate passed; independent native oracle pending

The loader now explicitly sets `llama_model_params.use_extra_bufts=False`.
On the same pinned 7B and 0.3.35 wheel, the original acceptance harness run
**directly against the modified source**, without monkeypatches/launcher overrides,
passed all **21/21 comparisons over 24 runs**, max distribution delta **0.0**,
at unchanged `atol=0.0001`. Changed-context sensitivity was observed; the model
checksum and imported source path were verified. Private report:
`pd-patched-native-acceptance.json` (with sibling `.log`).

The controlled full-schema diagnostic with this loader setting also showed
identical target logits at limits 8/1/2. Its native trace retained one prefill,
five-sequence ordinary and three-sequence collision batches, copy/removal calls,
and unchanged root KV state (`pd-divergence-no-extra-bufts.{json,npz,log}`).
The earlier override-only full acceptance is `pd-accept-no-extra-bufts.json`.

**Bounded conclusion:** standard buffer types resolve the tested decision-row
cross-limit failure, not arbitrary native numerical equivalence. A separate
single-token probe still differed by about 0.711 raw logit between one/eight
identical branches with extra buffer types disabled (repeats and peer branches
were identical). The precise kernel-level cause is not established. Do not claim
that repacking explains every observed difference or that a singleton reference
is inherently correct. No serial fallback, rounding, tolerance change or prompt
change was made. `test_llamacpp_loader.py` checks the loader argument before model
creation; it does not replace native numerical tests.

The acceptance harness deliberately retains
`FUNCTIONAL_CONTRACT_PASS_NATIVE_PROOF_PENDING`. Self-reported telemetry alone
is not independent native proof. The independent serial native teacher-forcing
collision oracle below remains outstanding; full integration acceptance is not
yet declared. Earlier statements in this file record the pre-fix investigation.

## Historical status (2026-09-17): BLOCKED, no production fix justified

Starting HEAD `a691967`, clean `integration/windows-gguf` worktree. Native acceptance
report supplied by the owner failed at unchanged absolute tolerance **0.0001**.
This investigation reproduced the failure before proposing production changes.
No production inference code, prompt, candidate/collision semantics, calibration,
row limits, tolerances, or environment were changed. No full acceptance rerun:
none of the three focused native configurations improved the failing comparison.
No downloads, mailbox access, credentials, installations or shared-environment
writes. The local official split Qwen2.5-7B-Instruct Q4_K_M model was read-only.

## Reproduction and controls

`scripts/diagnose_gguf_divergence.py` requires explicit `--run-native` and
`--exclusive-model-use`, a local model and a new output path. It contains only the
acceptance harness's synthetic fixture; it cannot accept arbitrary mail/context.
It invokes the public Decider at limits 8, 1, 2 with the **same full schema** and
383-token long prompt. Reducing the schema would change the prompt and invalidate
the comparison. It focuses numerical analysis on one expanded row, `tags[1]`.
Native trace/JSON and copied full-vocabulary float32 arrays are retained privately.

Read-only interpreter: `sortkasten/.venv/Scripts/python.exe`; binding **0.3.35**.
Imports pinned to this worktree's `src`, bytecode writes disabled. CPU-only,
`n_ctx=4096`, `n_batch=n_ubatch=512`, 9 sequences, unified KV, 8 threads.
System info: SSE3, SSSE3, AVX, AVX2, F16C, FMA, LLAMAFILE, OPENMP, REPACK enabled.
This identifies advertised build capabilities, **not which kernel executed**.

| Configuration (three total) | KV storage | Actual attention | max target probability delta | max target raw-logit delta |
|---|---|---|---:|---:|
| Unchanged baseline | f16 K/V, 224 MiB | auto resolves enabled | 0.009533555247378644 | 0.7090544700622559 |
| Flash off only | f16 K/V | disabled | 0.030477694519114973 | 1.15985107421875 |
| fp32 KV only | f32 K/V, 448 MiB | auto resolves enabled | 0.009533555247378644 | 0.7090544700622559 |

Each script exited **1**, explicitly reporting `strict_pass=false`. No crashes.
The last variant's context log confirms f32 allocation, so it was not a no-op
parameter request. It nevertheless produced identical target results to baseline.
Disabling flash worsened the discrepancy. These controls do not prove that all
attention/precision effects are excluded, only that neither isolated setting fixes
this fixture. No further native variants were run under the bounded task.

## What was actually established

- Prompt IDs saved separately from suffix IDs; every native-tokenized full
  prompt+suffix equals prompt IDs plus compiled suffix IDs. No prompt truncation.
- `tags[1]` suffix is `[220,330,14082,58,16,60,788,220]`, candidates are
  `[[1866],[3849]]`. Its scored token is 220 at absolute position **390**.
- Baseline target output maps to batch index **33 / seq 4** (limit 8),
  **7 / seq 1** (limit 1), **15 / seq 2** (limit 2). Native requested-index
  `llama_get_logits_ith` arrays equal corresponding compact `llama_get_logits`
  outputs byte-for-byte; no dense/compact output indexing mistake was observed.
- Full native trace proves exactly **one 383-token root decode per request**, a
  **35-token ragged ordinary suffix decode with 5 independent nonzero IDs** at
  limit 8, and a **26-token collision decode with 3 IDs**. No serial re-prefill,
  high-level completion, padding or generation. Limits 1/2 trace real smaller
  independent branch batches. Sequence copies are [0,383); removes succeed.
- Serialized root sequence state is 21,968,044 bytes with SHA256
  `ff4fd6a509888be92c6a7503537eea15f9b2efe307acf626cbd0409722e1aede`
  after each baseline prefill, identically across limits. Its byte hash and
  min/max (0/382) remain unchanged during every suffix/collision decode.
  Clears pass `data=True`; empty-root max is -1; removed branch max is -1.
  Sequence serializations include sequence metadata: different branch IDs need
  not hash identically. Raw physical unused KV memory was **not** inspected;
  serialized root stability is not a proof that every unused cell is zero.
- Returned logits have shape `(152064,)`, dtype float32. Candidate indexing uses
  the same IDs. Independent scalar `math.fsum` full-vocabulary log-softmax matches
  production float64 `_log_probability` within 1e-12 for the target's two tokens.
  Candidate renormalization reconstructs the public marginal exactly. The native
  logit differences precede Python slicing, conversion and normalization.

Baseline detailed target values:

| limit | logit true (1866) | logit false (3849) | logit margin | P(true), allowed-only |
|---|---:|---:|---:|---:|
| 8 | 17.17156982421875 | 13.862398147583008 | 3.309171676635742 | 0.9647421164129862 |
| 1 / 2 | 16.745254516601562 | 13.685341835021973 | 3.05991268157959 | 0.9552085611656076 |

Offline inspection of saved arrays also confirms lossless float32→float64→float32
conversion. Candidate float32 bit patterns are `41895f60 / 415dcc62` (limit 8)
versus `4185f648 / 415af729` (limit 1). These already differ before any float64
conversion; casting does not fix the underlying native result.

Full-vocab log probabilities for true/false at limit 8:
`-8.594943835523503 / -11.904115512159246`; at 1/2:
`-8.916412019066836 / -11.976324700646426`.
Thus the difference is not simply a common logit offset. Limits 1/2 full target
vectors are bit-identical. Largest difference is vocabulary ID **86229**; first
unequal vocabulary index is 0 (this is NOT the first divergent model operation).
Full-vocab top-five IDs at all limits: `16,830,15,895,330` — the engine still uses
only the unchanged compiled true/false candidates, preserving reference semantics.
Top full-vocab token 16 probability is 0.8345630343620629 (8) versus
0.8881472949749206 (1/2). Full-vocab and allowed-only probabilities differ by design.

## Regression coverage and evidence gaps

`tests/test_gguf_divergence.py` freezes the actual 383-token synthetic prompt and
five Qwen suffix rows. A deterministic fake validates independent causal history,
sparse output indexing, ragged packing, changed order, limits 8/2/1, native token
capacities 8/512, copied output lifetime, and candidate/full-vocab normalization.
It is **not** evidence about native kernel agreement. Existing exact-collision
fake oracle remains in `tests/test_llamacpp.py`. New CLI help/opt-in tests run with
`python -S`, proving help/refusal do not need native/model imports.

Precise native root cause remains **unresolved**. The observed association is with
batch shape; it is not enough to assert 'float reassociation', a specific CPU
kernel dispatch, fp16 corruption, or cross-sequence attention. The current trace
captures requested decision/answer-predecessor outputs, not every intermediate
suffix position or layer. A subsequent separately authorized diagnostic should
find the first divergent token/layer and inspect CPU dispatch (including quantized
matmul paths) while preserving the real independent-sequence batch. Do not assume
that a dtype or attention toggle is a solution.

A native **independent scalar teacher-forcing collision oracle** is still missing;
the fake collision oracle and native collision trace do not replace it. The full
native architecture/acceptance gate is therefore NOT PASS. Neither Sortkasten
integration nor Gmail accuracy/readiness can be inferred. No serial fallback,
rounding, tolerance relaxation or prompt tuning was introduced.

## Verification

Focused existing/new backend and acceptance tests: **46 passed** before adding
CLI gate coverage. Final complete offline suite: **256 passed, 12 skipped** in
8.72 s with the owner-supplied interpreter and this worktree's `PYTHONPATH`.
Native diagnostic strict tests remain **3 failures / 3 configurations** as above;
the green offline suite is not a native acceptance pass.

## Private synthetic artifacts

All beneath `C:/Users/Richard/Documents/Projects/sortkasten/private/provisioning/`:

- `pd-divergence-baseline.{json,npz,log}`
- `pd-divergence-flash-off.{json,npz,log}`
- `pd-divergence-fp32kv.{json,npz,log}`

The original `pd-merged-native-acceptance.json` was read only. `.npz` contains raw
synthetic native logits; JSON contains actual token/position/sequence traces and
serialized-state hashes (not KV contents). None of these private artifacts are
committed. Notes and tests are synthetic only.
