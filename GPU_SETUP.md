# GPU setup and what the speedups actually mean

## Bottom line

This is **LLM inference**, not CUDA video rendering. The GPU answers a fixed schema
of questions. CUDA Graphs make repeated inference calls cheaper to launch; they do
not render pages or make Playwright clicks themselves faster.

On this Windows PC (RTX 2060 SUPER, 8 GB), using **Qwen2.5-0.5B-Instruct**:

| Change being measured | Before → after | What improved | Evidence |
|---|---|---|---|
| CPU → ordinary CUDA | 2,887 → about 355–365 ms/request | About **8× faster**, on an 816-token, 3-field workload | Historical session measurements; research doc 14 |
| CUDA full prefill → shared schema prefix | 88.5 → 61.9 ms/request, means | **1.43× throughput equivalent / 30% less latency**; 8 description-heavy fields, 405 prefix tokens, 10 pairs, fp16 | `benchmarks/prefix-results.json` |
| Ordinary CUDA → CUDA Graph replay | 60.1 → 51.8 ms/request, medians | **1.16× / 14% less latency**; 8 synthetic boolean fields, 10 pairs, fp16 | `benchmarks/cuda-graph-results.json` |
| Ordinary CUDA → CUDA Graph replay | 240.9 → 83.8 ms/request, medians | **2.88× / 65% less latency**; custom 27-field schema, 6 pairs, fp16 | `benchmarks/cuda-graph-27-results.json` |
| Ordinary CUDA → CUDA Graph replay (retry) | 190.7 → 202.8 ms/request, medians | **6.3% MORE latency**; synthetic 27-field schema, 10 pairs, fp16 | `benchmarks/cuda-graph-retry-diagnostic.json` |

**Integration default: leave CUDA Graphs off.** They are an optional experiment,
not a guaranteed acceleration. The latest synthetic workload was slower with
graphs; enable only after paired measurements on your app's actual inputs.

**The 2.88× is an extra gain over an already GPU-accelerated path, not a CPU
comparison.** It saves about 157 ms on that particular request. At one request
at a time, reciprocals of those medians are about 4.2 versus 11.9 requests/second;
these are illustrative rates, not measured multi-user service throughput.

Do **not multiply these speedups**: the rows use different schemas, lengths,
precision settings, summary statistics and runs. There is no verified combined
"8 × 1.43 × 2.88" gain. The CPU run used fp32; the early CUDA run auto-selected
bf16, so that comparison includes a precision change. The 0.5B latency results
also do not inherit the older 7B model's accuracy score.

### Why the result varies

A request has two main parts:

1. **Prefill:** read the schema and context. Prefix reuse saves the repeated schema
   part; it cannot skip reading a new context.
2. **Suffix/decision evaluation:** evaluate allowed answers in parallel, including
   extra collision-scoring passes where needed. CUDA Graphs replay this part using
   fixed-shape buffers, reducing CPU/kernel launch overhead. Prefill stays eager.

In the 27-field run, suffix median fell **199.8 → 42.1 ms**, while prefill stayed
about **40 ms**. In the 8-field graph run, suffix median fell **24.1 → 16.0 ms**.
Graphs help more when the suffix stage is a large share of total latency.
A profiler showing much time in matrix multiplication does not prove a specific
memory-bandwidth bottleneck; graphs do not remove the required matrix math.

### Warm results, not startup time

The table excludes model loading/downloads, graph capture and warmup. Graph
capture took about **187 ms** for the 8-field shape and **283 ms** for the custom
27-field shape. Prefix preparation took **89 ms** in its separate experiment.
Approximate capture-only break-even is 23 repeated 8-field requests or 2 repeated
27-field requests; warmup/startup can push the real break-even later.

The first occurrence of a graph shape runs eagerly; the second attempts capture;
subsequent calls replay it. Keep one model process alive to amortize these costs.

## Windows installation (explicit, reproducible route)

Prerequisites: NVIDIA driver, Python 3.12, Git and `uv`. These are the versions
used in the measurements: PyTorch **2.6.0+cu124**, Transformers **5.17.0**.
A driver reporting CUDA 13.x can still run the CUDA 12.4 PyTorch wheel; a separate
CUDA toolkit is not needed for this Python-wheel setup.

In PowerShell, from the release checkout:

```powershell
cd C:\Users\Richard\Documents\Projects\parallel-decisions
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv\Scripts\python.exe -e ".[torch]" transformers==5.17.0 "pytest>=8"
& .venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Install **`.[torch]`**, not just `.` for Torch inference. The extra supplies Torch,
Transformers and Accelerate; MLX dependencies are restricted to Apple Silicon
macOS. No `--no-deps` workaround is needed. Installing the CUDA wheel first selects
the desired PyTorch build; the extra alone does not promise a specific CUDA build.
For CPU-only integration, replace the first install command's index with
`https://download.pytorch.org/whl/cpu`. The base package on Windows remains useful
for schema/config/calibration utilities but does not install an inference backend.

**Measured acceptance baseline for this release:** one CPU wheel installation with
torch **2.6.0+cpu** and transformers **5.17.0** passed real inference, prefix reuse
and `decide_many` with automatic defaults; one CUDA session used torch
**2.6.0+cu124** with the same transformers. Other versions within the extra's
bounds are untested.

### Recommended starting configuration

Create `pd.toml` in the directory from which you run the app:

```toml
backend = "torch"
model = "Qwen/Qwen2.5-0.5B-Instruct"
torch_device = "cuda"
torch_dtype = "float16"
cuda_graph = false  # opt in only after measuring your app's workload
max_fields_per_batch = 8
lock_timeout_s = 60
```

**Pin the model for reproducible integrations.** `backend="auto"` selects Torch
on Windows with `Qwen/Qwen2.5-0.5B-Instruct` as its fallback model. Apple Silicon
retains the MLX 7B 4-bit default. Explicit/configured IDs take precedence and are
not automatically converted between backends. Explicit fp16 is the tested
starting point on this Turing GPU; reported bf16 availability is not proof of
native bf16 performance.

Start with 8 rows, then measure your actual schema. An unquantized 7B model needs
roughly 14 GB for fp16 weights alone and does **not** fit this 8 GB card. No Torch
4-bit loading option is exposed here. The Torch path currently chunks by
`max_fields_per_batch`, not an automatically enforced VRAM budget. Reduce this
limit when needed; eager inference can still run out of memory.

### Python and CLI usage

```python
from parallel_decisions import Decider, Schema

# Load once at application startup, not once per request.
decider = Decider(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    backend="torch", torch_device="cuda", torch_dtype="float16",
    cuda_graph=False, max_fields_per_batch=8,
)
schema = Schema({
    "category": {"type": "enum", "choices": ["BILLING", "SUPPORT"],
                 "description": "Ticket category"},
    "needs_review": {"type": "boolean", "description": "Needs human review"},
})
prefix = decider.prepare(schema)
try:
    for ticket in ["Duplicate charge on invoice", "Cannot sign in"]:
        result = decider.decide_with_prefix(prefix, ticket)
        print(result.json(), result.latency_ms, result.telemetry)
finally:
    prefix.release()
```

Without repeated schemas, use `decider.decide(context, schema)`. To disable graph
capture, pass `cuda_graph=False`; to compare without prefix reuse, use `decide`
instead of `decide_with_prefix`. `decide_many(shared_prefix=True)` is a convenience
loop with shared prefill, **not cross-context GPU batching**.

With `pd.toml` present:

```powershell
& .venv\Scripts\pd.exe config
& .venv\Scripts\pd.exe decide --schema examples/fraud.json --context "Customer reports a duplicate charge."
& .venv\Scripts\python.exe examples\serve.py --port 8000
```

The long-lived localhost HTTP example reuses one model and can amortize graphs;
it does not automatically prepare a schema prefix. Each standalone CLI invocation
loads a new model, so it is not the way to get warmed replay performance.

## Reproduce the comparisons

```powershell
& .venv\Scripts\python.exe scripts\measure_cuda_graph.py --output graph-local.json --runs 10
& .venv\Scripts\python.exe scripts\measure_prefix_speedup.py --output prefix-local.json --runs 10
& .venv\Scripts\python.exe scripts\measure_prefix_speedup.py --cuda-graph --output prefix-with-graphs.json --runs 10
```

`measure_cuda_graph.py` also accepts `--schema path.json --context path.txt`.
Currently its schema argument expects a **flat field mapping** (not a `fields`
wrapper); use UTF-8 without BOM. Inspect output for `graph_unavailable`, captured
shapes, replay counts, capture cost, probability differences and answer agreement.
A successful call alone does not prove a graph was used: fallback is intentional.
Compare matching workload, model, precision and device settings; include cold
startup if your application starts a new process for each job.

## Graph scope, memory and browser caveats

- Graphs are experimental and off by default. Supported path: Qwen2 full-attention
  models in eval mode, eager/SDPA attention. CPU/MLX do not use them.
- StaticCache buffers are keyed by rows, suffix length and prefix-length bucket.
  Up to four captured shapes are retained; more shapes can remain eager.
- Capture/replay failures disable graphs for that runtime with one warning.
  Capture refuses estimates above 2 GiB or half the currently free VRAM. This is a
  heuristic, not an OOM guarantee. Other desktop applications change free VRAM.
- The original 27-row attempt was refused for memory pressure. A later **custom**
  27-field run succeeded; it does not replace that recorded refusal or establish
  performance for every 27-field schema. A fresh documentation verification run
  refused **both** 8- and 27-field captures with CUDA reporting zero free bytes:
  [`benchmarks/cuda-graph-doc-verification.json`](benchmarks/cuda-graph-doc-verification.json).
  Eager fallback completed, but no new replay timings were obtained. The successful
  historical measurements above are not a guarantee under current desktop load.
  The small, no-download CUDA regression test still passed: one graph captured,
  three replays with changed inputs/lengths, eager logits matched within `1e-5`.
  That is a correctness check, not a representative-model speed measurement.
- The subsequent diagnostic retry captured the **synthetic 27-field** shape:
  eager **190.7 ms** vs graph **202.8 ms** p50 over 10 pairs (6.3% slower),
  capture ~490 ms, identical answers and probabilities. The 8-field shape was
  refused when CUDA reported zero free bytes, then the 27-field check reported
  5,664 MiB free moments later without closing applications. These Windows/WDDM
  readings do not establish an actual OOM or fragmentation. We did not change
  the guard or stop processes. Raw evidence:
  [`benchmarks/cuda-graph-retry-diagnostic.json`](benchmarks/cuda-graph-retry-diagnostic.json).
  Leave graphs off unless your app's paired measurements show a benefit.
- The earlier Chrome/Playwright lab fired 10 local HTTP requests and displayed
  model recommendations in the DOM: about **2.0 seconds** with one model process,
  **2.5 seconds** with two. It did not execute ten model-selected Playwright
  clicks or real email actions. It predates the prefix/graph measurements.
- Two/four model-process experiments were slower on this shared card. Their
  process-wall throughput includes startup and is not steady-state throughput.
  Prefer one long-lived queued model; benchmark actual browser round trips next
  before claiming faster real-world automation.
- Fast, valid outputs can still be wrong. The toy email output even recommended
  payment for suspicious messages. Keep this advisory; do not execute payments,
  blocks or other consequential actions from unvalidated model confidence.

## Project layout on this machine

All paths are under `C:\Users\Richard\Documents\Projects\`:

| Folder | Purpose |
|---|---|
| `parallel-decisions` | Public release library, primary checkout; run/install this |
| `jev-on-a-laptop` | Public research mirror: docs, evals, local browser lab |
| `rlcd-research` | Private research repository; not a source for public raw-data uploads |
| `rlcd-upstream` | Clone of `harshatheg/Qwen-2.5-1B-RLCD`; method reference |
| `parallel-decisions_wt/gpu` | Earlier implementation worktree; currently holds the reusable test venv |
| `parallel-decisions_wt/cuda-graphs` | Isolated graph-development worktree |

Worktrees share Git history but not working files. When borrowing the GPU
worktree's venv, pin `PYTHONPATH` to the checkout you mean to test; its editable
install otherwise imports the older worktree. See `AGENTS.md` for the exact check.
