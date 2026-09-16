# parallel-decisions

**Turn a local LLM into a typed decision engine.** Give it a context and a small
schema of questions; get back typed answers with probabilities — as one batched
forward pass, with JSON that cannot be malformed.

Built on the "parallel constrained decoding" idea behind Jev / TypeSafe AI, packaged
as a clean, dependency-light library for Apple Silicon.

```python
from parallel_decisions import Decider, Schema

decider = Decider()  # Qwen2.5-7B-Instruct-4bit, runs on your Mac

schema = Schema({
    "category":  {"type": "enum", "choices": ["billing", "technical", "account"], "description": "Primary support category"},
    "priority":  {"type": "enum", "choices": ["P1", "P2", "P3"], "description": "Urgency tier"},
    "needs_human": {"type": "boolean", "description": "Must a person look at this"},
})

result = decider.decide("Ticket: charged twice for my subscription, angry, wants a refund.", schema)

result["category"].value       # "billing"
result["category"].probability # 0.94
print(result.json())           # {"category": "billing", "priority": "P2", "needs_human": true}
```

## Why this exists

- **Typed by construction.** The model never writes JSON. Field values are selected
  from allowed answers and assembled in code, so keys and types are always valid.
- **One pass for all questions.** The context is prefilled once; every field is
  evaluated in a single batched forward pass. Adding fields adds little latency.
- **Local and free.** Runs on Apple Silicon with MLX. No API keys, no network
  after the model download.
- **Honest probabilities.** Each answer carries a softmax over its allowed choices.
  It is a confidence signal, not a calibrated probability — see below.

## Measured accuracy

On all public example cases of TypeSafe's four eval workflows (20 cases, 373
reference question-pairs), scored against their own reference — the consensus of
GPT-6 Astra and Fable 5.1 — this package's default model reached **73.8%**
(253/343 on the strict like-for-like subset), versus Jev at 86.6% and frontier
models at 89–90%.

It is not frontier-accurate. It is free, private, offline, and good enough for
many routing and extraction jobs. Full write-up:
`docs/12-full-head-to-head.md` in the parent research repo.

## Install

```bash
cd parallel-decisions
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
```

Or with plain pip:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

First run downloads the model (~4.3 GB) from Hugging Face.

## Python API

### Define a schema

```python
from parallel_decisions import Schema

schema = Schema({
    "verdict": {"type": "enum", "choices": ["APPROVE", "REVIEW", "BLOCK"], "description": "What to do with this payment"},
    "is_fraud": {"type": "boolean", "description": "Whether this looks fraudulent"},
    "confidence_note": ...  # not supported: only boolean and enum
})
```

Supported field types: `boolean` and `enum` (2–255 choices). Free text and numbers
are not supported by the decoding method.

### Decide

```python
from parallel_decisions import Decider

decider = Decider()                      # default model
decider = Decider("mlx-community/Qwen2.5-1.5B-Instruct-4bit")  # faster, less accurate

result = decider.decide(context, schema)

result["verdict"].value          # "BLOCK"
result["verdict"].probability    # 0.97
result["verdict"].alternatives   # [("REVIEW", 0.02), ("APPROVE", 0.01)]
result.json()                    # plain dict of values
result.full_json()               # values + probabilities per field
result.latency_ms                # wall time for this call
result.prefill_ms, result.pass_ms
```

### Batch many contexts

```python
results = decider.decide_many([ctx1, ctx2, ctx3], schema)
```

Each context is prefilled once; calls are sequential but the model stays loaded.

### Save and load schemas

```python
schema.to_json("fraud.json")
schema = Schema.from_json("fraud.json")
```

## Using it from another project

Install this folder as an editable dependency of the other project's environment:

```bash
uv pip install --python /path/to/other-project/.venv/bin/python -e /Users/richardbaecker/Documents/projects/parallel-decisions
```

Then in that project:

```python
from parallel_decisions import Decider, Schema

# create ONCE at process start (model load takes a few seconds) and reuse
_decider = Decider()

def classify(ticket: str) -> dict:
    return _decider.decide(ticket, TICKET_SCHEMA).json()
```

For services, keep one `Decider` per process and call it from a thread pool — the
model is small enough for one request at a time, and batching is the caller's job.
If you cannot install the package, copying `src/parallel_decisions/` into the other
project works: it has no intra-package dependencies outside the standard library.

Swap the model per use case:

| model id | notes |
|---|---|
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | default, best measured accuracy (73.8%) |
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | ~4x faster, noticeably weaker decisions |
| `mlx-community/Qwen3-8B-4bit` | fits, no measured advantage over the 7B here |

## CLI

```bash
# validate a schema (checks types, choice counts, token collisions)
.venv/bin/pd validate examples/fraud.json

# run one decision
.venv/bin/pd decide --schema examples/fraud.json --context "transfer of $49,500 to an offshore escrow, new device, Tor exit node"

# from files, JSON output
.venv/bin/pd decide --schema examples/fraud.json --context-file ticket.txt --json
```

## HTTP server (optional, for non-Python callers)

```bash
.venv/bin/python examples/serve.py --port 8000
curl -s localhost:8000/decide -H 'content-type: application/json' \
  -d '{"context": "...", "schema": {"fields": {"risk": {"type": "enum", "choices": ["LOW","HIGH"], "description": "risk level"}}}}'
```

Uses only the standard library.

## Performance on an M5 MacBook Air (16 GB)

| Context size | Fields | Latency |
|---|---|---|
| ~1k tokens | 3–14 | 0.5–1.5 s |
| ~1k tokens | 28 | 1.5–2 s |
| ~11k tokens | 10–48 | 10 s–7 min (see below) |

Two costs dominate:

- **Prefill scales with context length.** A long context is read once; that read is
  most of the time for big documents.
- **Memory scales with `fields × context`.** The KV cache is broadcast once per
  field. For very long contexts with many fields this can exceed 16 GB, so the
  library automatically evaluates fields in chunks when needed (`max_fields_per_batch`,
  default 24). Chunking costs one extra prefill per chunk.

## Accuracy notes

- **Probabilities are softmax over allowed answers**, not calibrated frequencies.
  Treat them as relative confidence. In our eval, a high probability did not
  reliably flag errors.
- **Choice collisions.** If two choices start with the same token, a single logit
  cannot separate them. This happens when a field's choices share a leading word but
  the set as a whole has no common prefix — e.g. `extra_approved` / `extra_unapproved`
  alongside `unsure` (both begin with the token ` extra`). The library detects this
  and scores those fields with one extra batched pass over the full choice sequences,
  which is exact but slower for that field. `pd validate --check-tokens` reports
  collisions so you can rename choices to avoid them when latency matters.
- Only **one decision per field**; no free-form text, numbers, or multi-select.

## Project layout

```
parallel-decisions/
├── src/parallel_decisions/
│   ├── schema.py     # Schema/Field, validation, token compilation
│   ├── engine.py     # Decider: prefill, broadcast, batched passes
│   ├── prompts.py    # prompt construction
│   └── cli.py        # `pd` command
├── examples/
│   ├── basic.py
│   ├── fraud.json
│   ├── support.json
│   └── serve.py
├── tests/
└── AGENTS.md         # notes for AI coding agents working in this repo
```

## License

MIT (this package). The parallel-constrained decoding technique originates from
the community artifact `harshatheg/Qwen-2.5-1B-RLCD` (Apache-2.0); models are
Apache-2.0 by their respective publishers.
