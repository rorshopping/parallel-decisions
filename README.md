# parallel-decisions

**Turn a local LLM into a typed decision engine.** Give it a context and a small
schema of questions; get back typed answers with probabilities — as one batched
forward pass, with JSON that cannot be malformed.

Built on the "parallel constrained decoding" idea behind Jev / TypeSafe AI, packaged
as a clean, dependency-light library for Apple Silicon.

```python
from parallel_decisions import Decider, Schema

decider = Decider(calibration="calibration.json")  # optional: thresholdable numbers

schema = Schema({
    "category":  {"type": "enum", "choices": ["billing", "technical", "account"], "description": "Primary support category"},
    "priority":  {"type": "enum", "choices": ["P1", "P2", "P3"], "description": "Urgency tier"},
    "needs_human": {"type": "boolean", "description": "Must a person look at this"},
})

result = decider.decide("Ticket: charged twice for my subscription, angry, wants a refund.", schema)

result["category"].value       # "billing"
result["category"].probability # 0.94  (calibrated, if a calibrator is loaded)
print(result.json())           # {"category": "billing", "priority": "P2", "needs_human": true}
```

Two commands to a working call:

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .
.venv/bin/pd decide --schema examples/fraud.json \
  --context "wire transfer to Cyprus, new device, Tor exit node"
```

## Why this exists

- **Typed by construction.** The model never writes JSON. Field values are selected
  from allowed answers and assembled in code, so keys and types are always valid.
- **One pass for all questions.** The context is prefilled once; every field is
  evaluated in a single batched forward pass. Adding fields adds little latency.
- **Local and free.** Runs on Apple Silicon with MLX. No API keys, no network
  after the model download.
- **Calibratable probabilities.** Each answer carries a softmax over its allowed
  choices. That is a ranking signal, not a probability of being right — the
  calibrator (`pd calibrate`) turns it into one you can threshold on, with the
  routing table to prove it.

## Measured accuracy

On all public example cases of TypeSafe's four eval workflows (20 cases, 373
reference question-pairs), scored against their own reference — the consensus of
GPT-6 Astra and Fable 5.1 — this package's default model reached **73.8%**
(253/343 on the strict like-for-like subset), versus Jev at 86.6% and frontier
models at 89–90%.

That headline hides as much as it says, and the details are more useful than the
total (all measured, see `evals/analysis/REPORT.md` in the research tree):

| slice | result |
|---|---|
| `noul` (yes/no) questions | 83.1% — but the true rate is 18.6%, and a constant "false" scores 92.7% on invoices |
| Invoice `noul` | 90.8% vs a 92.7% constant-answer baseline (0 of 8 positives found) |
| Score (0–3) questions | 25.9% exact, 85.2% within one level |
| Choice questions | 63.6%, and 96.7% when the correct option happens to be listed first (24.5% otherwise) |

So: it is not frontier-accurate, it is free, private, offline, and the reliable
slices are ones where the answer is written down in the context rather than
inferred. Full write-up: `evals/analysis/REPORT.md` and `docs/12-full-head-to-head.md`
in the research repo.

## Install

```bash
# editable, from a checkout
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

# or plain pip
python3.12 -m venv .venv && .venv/bin/pip install -e .
```

First run downloads the model (~4.3 GB) from Hugging Face. Apple Silicon (arm64)
is the supported platform; on anything else you get a clear error rather than a
crash (`PD_ALLOW_NON_ARM=1` to try anyway).

## Python API

### Define a schema

```python
from parallel_decisions import Schema

schema = Schema({
    "verdict": {"type": "enum", "choices": ["APPROVE", "REVIEW", "BLOCK"], "description": "What to do with this payment"},
    "is_fraud": {"type": "boolean", "description": "Whether this looks fraudulent"},
    "actions":  {"type": "multi", "choices": ["retry", "escalate", "refund"], "description": "Actions that apply"},
})
```

Supported field types: `boolean`, `enum` (2–255 choices) and `multi` (any subset of
the choices). Free text and numbers are not supported by the decoding method.

Answers can carry their definitions, which measurably helps the model — this is the
single highest-leverage thing you can do to a schema:

```python
schema = Schema({
    "context_explains_activity": {
        "type": "boolean",
        "description": "Do the records account for the flagged activity?",
        "choices": {
            "true":  "A specific record covers this specific activity",
            "false": "No record covers it, or the records are about something adjacent",
        },
    },
})
```

### Decide

```python
from parallel_decisions import Decider

decider = Decider()                      # default model
decider = Decider("mlx-community/Qwen2.5-1.5B-Instruct-4bit")  # faster, less accurate
decider = Decider(calibration="calibration.json")              # calibrated confidences

result = decider.decide(context, schema)

result["verdict"].value          # "BLOCK"
result["verdict"].probability    # 0.97  (calibrated if a calibrator is loaded)
result["verdict"].raw_probability# 0.99  (the softmax, kept for comparison)
result["verdict"].distribution   # {"APPROVE": 0.01, "REVIEW": 0.02, "BLOCK": 0.97}
result["verdict"].alternatives   # [("REVIEW", 0.02), ("APPROVE", 0.01)]
result["actions"].value          # ["retry", "escalate"] for a multi field
result.json()                    # plain dict of values
result.full_json()               # values + probabilities per field
result.calibrated                # was a calibrator applied
result.latency_ms                # wall time for this call
result.prefill_ms, result.pass_ms, result.chunks
```

### Calibrate and route

```python
from parallel_decisions import Calibrator, fit_calibration, load_records, risk_coverage

records = load_records("labelled.jsonl")          # distribution + correct/target per row
fit = fit_calibration(records, method="auto")     # compares raw/temperature/platt/isotonic
fit.summary()                                     # cross-validated comparison
fit.calibrator.to_json("calibration.json")

cal = fit.calibrator
cal.transform({"true": 0.97, "false": 0.03})      # {"true": 0.88, "false": 0.12}
```

Or from the command line, which prints the comparison, the before/after reliability
table and a routing sweep:

```bash
.venv/bin/pd calibrate --data labelled.jsonl --out calibration.json
.venv/bin/python examples/routing.py --data labelled.jsonl --max-error 0.01
```

`examples/routing.py` is the point of the whole exercise: it answers "if I only act
on answers at this confidence, what error rate do I inherit and how much work do I
skip" — measured out of sample, with Wilson intervals on the error rate.

**Read the interval, not the point estimate.** Routing 100 decisions at a 1% budget
will typically show a 0% measured error rate with a 95% upper bound near 20%: that
is not evidence of a 1% error rate. The act bucket only becomes trustworthy with a
few hundred labelled rows from the domain it will run in — which is why the routing
demo is a measurement tool, not a policy you can copy.

### Batch many contexts

```python
results = decider.decide_many([ctx1, ctx2, ctx3], schema)
```

Sequential calls that reuse the loaded model and the compiled schema. (True
cross-context batching is not implemented; `decide_many` is a loop.)

### Save and load schemas

```python
schema.to_json("fraud.json")
schema = Schema.from_json("fraud.json")
```

## Configuration

Optional `pd.toml` in the working directory or `~/.config/parallel-decisions/`, and
`PD_*` environment variables that override it. Precedence: explicit argument > env >
file > default. `pd config` prints what is in effect.

```toml
model = "mlx-community/Qwen2.5-7B-Instruct-4bit"
calibration = "~/.config/parallel-decisions/calibration.json"
max_fields_per_batch = 6      # lower = less memory, more passes
memory_budget_gb = 6.0        # KV broadcast budget; also clamped to 65% of RAM
max_collision_rows = 8        # rows per pass when scoring colliding choices exactly
warmup = true                 # compile Metal shaders once at load
lock_timeout_s = 0            # 0 = fail fast if another thread is mid-call
log = "json"                  # one JSON line per call on stderr
```

### Observability

`PD_LOG=json` (or `log = "json"` in `pd.toml`) emits one line per model load and per
decision:

```json
{"event": "decide", "model": "...", "fields": 12, "rows": 12, "chunks": 1,
 "context_chars": 4210, "prompt_tokens": 1180, "prefill_ms": 640.2, "pass_ms": 121.9,
 "latency_ms": 812.4, "calibrated": true, "calibration_kind": "platt"}
```

### Concurrency

One model, one call at a time. `decide()` takes a lock; a second thread waits (or
raises `ConcurrencyError` after `lock_timeout_s`). `examples/serve.py` queues
requests and returns 503 rather than crashing when it cannot take one.

## Using it from another project

```bash
uv pip install --python /path/to/other-project/.venv/bin/python -e /path/to/parallel-decisions
```

```python
from parallel_decisions import Decider, Schema

# create ONCE at process start (model load takes a few seconds) and reuse
_decider = Decider(calibration="/path/calibration.json")

def classify(ticket: str) -> dict:
    return _decider.decide(ticket, TICKET_SCHEMA).json()
```

Swap the model per use case:

| model id | notes |
|---|---|
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | default, best measured accuracy (73.8%) |
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | ~4x faster, noticeably weaker decisions |

## CLI

```bash
# validate a schema (types, choice counts, token collisions with rename advice)
.venv/bin/pd validate examples/fraud.json --check-tokens

# run one decision
.venv/bin/pd decide --schema examples/fraud.json --context "transfer of $49,500 to an offshore escrow, new device, Tor exit node"

# with calibrated confidences, from files, JSON output
.venv/bin/pd decide --schema examples/fraud.json --context-file ticket.txt \
  --calibration calibration.json --json

# fit a calibrator from labelled rows, print the routing table
.venv/bin/pd calibrate --data labelled.jsonl --out calibration.json

# show the settings in effect
.venv/bin/pd config
```

## HTTP server (optional, for non-Python callers)

```bash
.venv/bin/python examples/serve.py --port 8000 --calibration calibration.json
curl -s localhost:8000/decide -H 'content-type: application/json' \
  -d '{"context": "...", "schema": {"fields": {"risk": {"type": "enum", "choices": ["LOW","HIGH"], "description": "risk level"}}}}'
curl -s localhost:8000/health   # busy flag + served/queued/rejected counters
```

Standard library only.

## Performance on an M5 MacBook Air (16 GB)

| Context size | Fields | Latency |
|---|---|---|
| ~1k tokens | 3–14 | 0.5–1.5 s |
| ~1k tokens | 28 | 1.5–2 s |
| ~38k tokens | 48 | ~10 min (prefill dominates) |

Two costs dominate:

- **Prefill scales with context length** and is most of the time for large
  documents. Measured on the invoice eval: 84.6% of case wall time, ~620 ms per
  1k prompt tokens.
- **Memory scales with `fields × context`** because the KV cache is broadcast once
  per decision row. The chunk size is derived from a *measured* per-token cache
  growth, clamped so the weights plus the broadcast stay under 65% of physical RAM
  (set `memory_budget_gb` lower if you want a smaller footprint). If a pass still
  fails with a memory error the chunk size halves and retries instead of failing the
  call. Chunking costs one extra cache copy per chunk, not another prefill.

## Accuracy notes

- **Probabilities are softmax over allowed answers**, not calibrated frequencies.
  Treat them as relative confidence until you fit a calibrator on your own labelled
  data. On the default model, 28 of 40 wrong answers carried ≥0.90 confidence before
  calibration, and the raw confidence ranks better than it calibrates (AUROC 0.74
  overall: 0.91 on choice fields, 0.66 on yes/no). Filtering to the most confident
  10% of answers cut the error rate from 35% to 17% on the data we measured.
  See [`CALIBRATION.md`](CALIBRATION.md) for the measured tables and, importantly,
  the sample sizes — **calibration is not a substitute for labelled domain data, and
  on ~100 rows no post-hoc method beat raw softmax out of sample.**
- **The model has a listing-order preference.** It picks the first listed option in
  82% of choice fields, and is right 96.7% of the time when the correct option is
  first versus 24.5% when it is not. For choice fields, order your options by what
  you believe is most likely — or treat low-confidence answers as "position, not
  evidence".
- **Choice collisions.** If two choices start with the same token, a single logit
  cannot separate them. The library detects this exactly (the candidate token is the
  one the model actually emits after the field's prefix) and scores the full
  sequences in one extra batched pass — correct, but slower. `pd validate
  --check-tokens` reports collisions with concrete rename suggestions.
- **One answer per field** unless the field is `multi`, which is one yes/no decision
  per choice. No free-form text or numeric output.

## Project layout

```
parallel-decisions/
├── src/parallel_decisions/
│   ├── schema.py       # Schema/Field, validation, token compilation, collisions
│   ├── engine.py       # Decider: prefill, broadcast, batched passes, calibration
│   ├── calibration.py  # Calibrator (temperature/Platt/isotonic), metrics, CV fitting
│   ├── config.py       # pd.toml + PD_* environment variables
│   ├── lint.py         # collision lint with rename advice
│   ├── prompts.py      # prompt construction
│   └── cli.py          # `pd validate` / `decide` / `calibrate` / `config`
├── examples/
│   ├── basic.py, fraud.json, support.json
│   ├── routing.py      # act / review / refuse policy from labelled data
│   └── serve.py        # stdlib HTTP server, queued
├── tools/
│   └── fit_calibration.py
├── tests/              # 87 tests, no model needed
├── CHANGELOG.md
└── AGENTS.md           # notes for AI coding agents working in this repo
```

## License

MIT (this package). The parallel-constrained decoding technique originates from
the community artifact `harshatheg/Qwen-2.5-1B-RLCD` (Apache-2.0); models are
Apache-2.0 by their respective publishers.
