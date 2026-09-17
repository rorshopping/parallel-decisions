# GGUF native acceptance execution (2026-09-17)

## Current release gate: functional PASS, expanded oracle FAILED

Final bounded run 3 completed with repacking disabled (`use_extra_bufts=False`):
**all 21 original harness comparisons passed, maximum delta 0.0**, all typed,
calibration, retained-object, singleton and context checks passed. No tolerance
relaxation. The boundary wrapper subsequently exited **1** on its additional
full-row scalar oracle: tags differ by **0.0017988534479707052** when the full
prompt and suffix are decoded together rather than in prefill/suffix phases.
Collision oracle delta **4.349700010175539e-7** passes at 1e-4; bool and enum
ordinary deltas are below 6e-9. Root cause of the remaining full-sequence shape
sensitivity is NOT established. This is not an overall green acceptance claim.
The diagnostic's legacy JSON key/error says `scalar_collision_oracle` even though
run 3 expanded it to ordinary rows too; the script now calls it full-row correctly.
No output values/tolerance were changed to make the functional checks pass.

Run 3 load: 737.65 ms (file cache uncontrolled). Traced median phase times in ms:

| limit | contexts | prefill | suffix + collision | request wall |
|---|---|---:|---:|---:|
| 8 | short A/B | 4372.00 | 2399.38 | 6724.24 |
| 8 | long | 14684.24 | 2362.49 | 17059.31 |
| 2 | short A/B | 4541.08 | 2606.52 | 7156.46 |
| 2 | long | 14409.35 | 2628.95 | 17051.79 |
| 1 | short A/B | 5024.71 | 2634.11 | 7596.50 |
| 1 | long | 16188.09 | 3430.41 | 19632.79 |

Repacking-off is retained because it fixes the requested cross-row-limit contract,
with a CPU latency cost. Three native runs total; no endless retries. Full
model-free suite after changes: **257 passed, 10 skipped**. Evidence remains
local in this worktree, with exact source hashes recorded inside each report.

## Earlier runs (historical failures, before repacking fix)

The merged harness commits 1b1621b and 7ddb83d were cherry-picked as c84520c
and 3a9ac0b. The shared `_assemble_multi` now uses the P(true) marginal.
Two completed full native suites (2 repetitions each, limits 8/2/1) failed
strict cross-row-limit numerical equivalence at the unchanged `atol=0.0001`.
All selected values stayed identical. Of 21 comparisons, 5 passed and 16 failed.
The maximum delta was **0.009533555247378644**, in `tags` for the long context.
For context A tags the maximum was 0.002318128842347944; for B, 0.00027005101122823394.
These are not explained away as rounding. Root cause is not yet established.

Reports are local synthetic-only files `native-acceptance-1.json`,
`native-acceptance-2.json`, and `native-acceptance-2.trace.json` in the owned
worktree. No mailbox data, downloads, shared-env installs or GPU offload occurred.
A third bounded diagnostic run disables model `use_extra_bufts` (runtime CPU
weight repacking); its result will be appended below. No tolerance relaxation.

## Native boundary evidence (run 2)

The `verify_gguf_boundary.py` wrapper calls the unmodified acceptance suite,
wraps actual native calls without injecting tokens/results, and records inputs,
sequence copies/removals/clears, root positions, sparse output-token indices,
return codes and synchronized decode wall times. It also runs an independent
full-sequence scalar teacher-forcing oracle on sequence 0 after the public suite.
That validation-only oracle deliberately re-prefills candidate sequences and is
NOT a production fallback. Later versions also cover ordinary rows.

- 26 public requests, exactly one root prefill each. Full-schema prompt token
  lengths A/B/long: 121/119/383; singleton schema 47. First token 151644
  (ChatML), no extra BOS inserted. The trace preserves every actual prompt ID.
- First request: root decode 121 tokens; ordinary suffix decode 35 tokens,
  distinct IDs 1..5; collision decode 26 tokens, IDs 1..3. Every suffix position
  starts at root length plus row offset. Unequal rows are ragged, not padded.
- 201 root copies and 201 successful removals. Copy ranges [0,prompt_length),
  source 0, destination max prompt_length-1. Root max unchanged through every
  suffix/collision decode. Removed branch max -1. Clears verified root max -1.
- All native decode return codes 0. Limit 1 really uses one branch per batch.
- Independent scalar **collision** oracle versus first public A result:
  max distribution delta **3.42992312662993e-7**, PASS at 1e-4;
  oracle wall 9065.1041 ms. This only validates the collision fixture, not accuracy.
- Production path uses no completion/chat/generate/sampling call. JSON is assembled
  from candidates. Native attention implementation itself is not instrumented.
- Evidence still requires independent coordinator/reviewer inspection; don't
  promote backend counters alone to proof. Overall acceptance remains failed.

## Provenance and phase times (run 2)

- Python 3.12, llama-cpp-python 0.3.35, read-only Sortkasten `.venv` interpreter.
- Model first shard SHA256:
  `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db`.
  No owner expected checksum was provided; this is recorded, NOT verified
  against upstream. Second shard hash is not collected by the base harness.
- llama.dll SHA256:
  `fadb7f4ffa452cf0702a9a27938562afcddc2538bcd71977e3ff584b071746ad`.
- Native system info: CPU SSE3 SSSE3 AVX AVX2 F16C FMA LLAMAFILE OPENMP REPACK.
  Engine sets n_gpu_layers=0, offload_kqv=False, op_offload=False, 8 CPU threads.
- Checksum 2798.15 ms, model load 3889.57 ms, no warmup. File cache uncontrolled.
- Prefill now explicitly calls llama_synchronize even without requested logits;
  native wrapper additionally synchronizes every decode during this trace.

Medians in ms over this synthetic traced run, not throughput/speedup claims:

| limit | contexts | prefill | suffix + collision | request wall |
|---|---|---:|---:|---:|
| 8 | short A/B | 3130.22 | 1715.80 | 4817.01 |
| 8 | long | 9772.50 | 1804.96 | 11592.49 |
| 2 | short A/B | 2948.51 | 1802.35 | 4724.50 |
| 2 | long | 9225.23 | 1856.41 | 11096.41 |
| 1 | short A/B | 2979.43 | 1974.07 | 4979.39 |
| 1 | long | 9104.69 | 2266.28 | 11385.11 |

## Rerun, only after exclusive model scheduling

PowerShell, with a new output path (files are exclusive-created):

```powershell
$root = 'C:\Users\Richard\Documents\Projects\parallel-decisions_wt\gguf-engine'
$py = 'C:\Users\Richard\Documents\Projects\sortkasten\.venv\Scripts\python.exe'
& $py "$root\scripts\verify_gguf_boundary.py" --run-native --exclusive-model-use `
  --model 'C:\Users\Richard\Documents\Projects\sortkasten\private\models\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf' `
  --output "$root\native-coordinator-rerun.json" --repeats 2 --atol 0.0001
exit $LASTEXITCODE
```

Use `verify_gguf_native.py` instead for the original functional-only harness.
The boundary script emits a sibling `.trace.json`. These are synthetic-only
instruments; never point them at user mail or add private contexts.

## Implemented facade signature

`Decider(model_id=<existing local GGUF>, backend='llamacpp',
max_fields_per_batch=8, max_collision_rows=8, n_ctx=4096, n_batch=512,
n_threads=8, warmup=False, calibration=None, lock_timeout_s=0,
config=Config(log='off'), verbose=False)`.

`n_threads=None` selects min(8,cpu_count). Config/env support PD_N_CTX,
PD_N_BATCH, PD_N_THREADS. Calibration accepts the existing path/object/mapping
forms. `warmup` accepted without hidden inference. `cuda_graph=False` accepted;
true, torch_device/torch_dtype, and memory_budget_gb fail closed.
No n_gpu_layers, device, model_path, n_ubatch or llama_kwargs facade arguments.
No prepared prefix/shared-prefix support. Changing row limits downwards after
load is honored; exceeding allocated sequence capacity fails, never resizes
or silently serializes. CPU Qwen2 full-attention GGUF only, binding pinned 0.3.35.
