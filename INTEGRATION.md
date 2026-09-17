# Integrating parallel-decisions

This is a synchronous, local inference library, not a hosted API. Inference can
run offline after model assets are cached. Install the appropriate backend first;
see [GPU_SETUP.md](GPU_SETUP.md) for the Windows/Torch setup and memory limits.
Torch defaults to `Qwen/Qwen2.5-0.5B-Instruct`; pin an explicit model ID for
reproducible deployments.
The HTTP and MCP scripts below are checkout examples, not installed `pd` commands.

## Python: one model per application lifespan

Create one `Decider` at application startup and reuse it. Construction resolves
configuration; `load()` explicitly loads/warmups the model. Without `load()`, the
first inference call loads it lazily. Startup can fail due to unavailable weights,
backend dependencies, device configuration or insufficient memory.

```python
import json
from parallel_decisions import Decider, Schema, ConcurrencyError

schema = Schema({
    "needs_human": {"type": "boolean", "description": "Requires human review"},
    "category": {"type": "enum", "choices": ["HARDWARE", "SOFTWARE"]},
})
decider = Decider(
    model_id="Qwen/Qwen2.5-0.5B-Instruct", backend="torch",
    torch_device="cuda", torch_dtype="float16", cuda_graph=False,
    lock_timeout_s=5,
)
decider.load()  # startup, not once per request

try:
    result = decider.decide("The laptop will not turn on.", schema)
except ConcurrencyError:
    # Reject, defer or retry with a bounded policy in your application.
    raise
else:
    values = result.json()             # dict: bool / str / list[str] values
    details = result.full_json()       # dict: value + probability for each field
    wire_json = json.dumps(details)    # the helpers return dicts, NOT JSON strings
```

Schema construction takes `{name: spec}` or `Field` objects. To load a serialized
`{"fields": ...}` wrapper, use `Schema.from_json(path_or_json_text)` instead.
Supported types are `boolean`, `enum` and `multi`; enum/multi allow 2–255 choices.
Keep schema objects unchanged after construction and reuse them across calls.
The JSON is assembled from allowed values, not generated as free-form text.

For many contexts with one schema, opt into shared-prefix reuse:

```python
prefix = decider.prepare(schema)
try:
    results = [decider.decide_with_prefix(prefix, text)
               for text in ("The screen is cracked.", "The app crashes.")]
finally:
    prefix.release()  # coordinate cleanup after in-flight work finishes
```

A prefix belongs to its creating `Decider`, holds additional KV memory, and must
be recreated when the schema changes. `prefix.reusable` reports whether a cache
was prepared; incompatible tokenization can fall back to full prefill.
`decide_many(contexts, schema, shared_prefix=True)` manages this lifetime for one
call. It is a sequential loop, **not** cross-context inference batching. There is
no public `Decider.close()` or context-manager lifecycle; stop accepting work,
wait for active calls, release prefixes, then let the owner process exit.

Calls block. Use your framework's worker/executor rather than blocking an async
event loop. One Decider serializes model use with a lock; `lock_timeout_s=0`
fails fast, a positive value waits before `ConcurrencyError`. This is a lock-wait
limit, not an inference deadline or a fair/bounded job queue. A caller's timeout
or cancelled async task does not cancel an already running forward pass.

## Configuration

Precedence: explicit Python arguments/CLI flags > `PD_*` environment > TOML >
defaults. Choose a file with `config="path/pd.toml"`, `--config`, or `PD_CONFIG`;
otherwise lookup checks the current directory, then the user config directory.
A directly supplied `Config` object is already resolved (no environment overlay).
For example, a CUDA starting configuration with experimental graphs **off**:

```toml
model = "Qwen/Qwen2.5-0.5B-Instruct"
backend = "torch"
torch_device = "cuda"
torch_dtype = "float16"
cuda_graph = false
max_fields_per_batch = 8
lock_timeout_s = 5
```

Equivalent environment keys include `PD_MODEL`, `PD_BACKEND`, `PD_TORCH_DEVICE`,
`PD_TORCH_DTYPE`, `PD_CUDA_GRAPH`, `PD_LOCK_TIMEOUT_S`, `PD_CALIBRATION` and
`PD_LOG=json`. Graphs are experimental and default off: baseline correctness and
capacity should be established without them. They do not make the engine concurrent.

## Local HTTP example

From a checkout with the package/backend available in the chosen interpreter:

```sh
python examples/serve.py --config pd.toml --port 8000 --queue-timeout 5
```

The default bind is **127.0.0.1**. It loads one model before listening. Only these
routes are implemented (no `/v1`, streaming, batch or prefix endpoints):

- `GET /health`: HTTP 200, JSON snapshot of model/busy state and counters.
- `POST /decide`: JSON object containing non-empty string `context` and `schema`.

Example `request.json`:

```json
{
  "context": "The laptop will not turn on after a power cut.",
  "schema": {
    "fields": {
      "category": {
        "type": "enum",
        "choices": ["HARDWARE", "SOFTWARE"],
        "description": "Issue type"
      }
    }
  }
}
```

```sh
curl --fail-with-body -sS http://127.0.0.1:8000/decide -H "Content-Type: application/json" --data-binary @request.json
curl -sS http://127.0.0.1:8000/health
```

On Windows PowerShell use `curl.exe` to avoid the legacy `curl` alias. HTTP also
accepts an unwrapped schema mapping or the `{"fields": [{"name": ..., ...}]}`
form produced by `Schema.to_json()`. A successful response has this shape
(**illustrative values**, not an inference measurement):

```json
{
  "decisions": {"category": {"value": "HARDWARE", "probability": 0.8123}},
  "calibrated": false,
  "latency_ms": 12.3,
  "model": "Qwen/Qwen2.5-0.5B-Instruct"
}
```

`latency_ms` is engine telemetry, not end-to-end HTTP time including queueing.
With calibration, fields also include `raw_probability`. Full distributions and
alternatives are available on Python `FieldValue` objects, not this HTTP response.
Health resembles `{"ok":true,"model":"...","busy":false,"served":1,
"rejected":0,"queued":0}`. `queued` counts requests observed arriving while the
lock was busy, not current queue depth; it is an approximate contention counter.

| Status | Meaning / body |
| --- | --- |
| 200 | Successful decision, or health snapshot |
| 400 | Invalid JSON, missing/invalid context or schema, invalid content length: `{"error":"bad request: ..."}` |
| 404 | Unknown GET/POST route: `{"error":"use POST /decide"}` |
| 503 | Lock wait expired, or no Decider available: `{"error":"...","retry":true}` |
| 500 | Inference/runtime failure: `{"error":"..."}` |

The request body limit is 1,000,000 bytes. Send a known `Content-Length` (curl does
this for the file example); chunked request bodies are not supported. Other HTTP
methods use the standard-library handler's error response, not the JSON contract.
`--queue-timeout` overrides config/env; without any configured value the example
waits up to 600 seconds for the lock. `0` fails fast with 503. Use a client timeout
appropriate to queue wait plus inference; disconnection does not cancel the model
call. Retry 503 with bounded backoff, not an unlimited retry loop. Treat 400 as an
input correction and investigate 500/device failures before retrying.

## CLI

Global `--model` and `--config` options go **before** the subcommand:

```sh
pd --config pd.toml config --json
pd validate examples/support.json
pd --config pd.toml decide --schema examples/support.json --context-file ticket.txt --json
pd --model Qwen/Qwen2.5-0.5B-Instruct decide --schema examples/support.json --context "My laptop failed" --json
pd calibrate --data labelled.jsonl --out calibration.json
```

`python -m parallel_decisions.cli` is equivalent to `pd`. Context comes from
`--context-file`, then `--context`, otherwise stdin; empty input is an error.
`--json` emits only a JSON document on stdout, including when `--verbose` is used
(logs go to stderr). Its body is `{"decisions": {...}}`; `"calibrated": true` is
added only when a calibrator was applied. CLI JSON does not include HTTP telemetry.
Use exit status as well as stdout: 0 success; 2 input/schema/config/calibration
errors; 1 caught model-busy/runtime/dependency errors. Failures report on stderr,
not as a success-shaped JSON object. Unexpected exceptions may still traceback.
Each CLI invocation owns a new Decider; use Python/HTTP for repeated low-latency
calls. `validate` is model-free unless `--check-tokens` is specified; token checking
can currently load model weights on Torch.

## MCP and larger applications

`python examples/mcp_server.py` exposes `decide_tool` and `validate_schema_tool`
over stdio using the optional MCP SDK (not a required package dependency). Set
`PD_*` / `PD_CONFIG` in the child process and use absolute interpreter/script paths
when registering it. The first call loads lazily. Use dictionary schemas (plain or
wrapped in `fields`); the MCP example currently does **not** accept the serialized
list-of-fields form. Validation errors return JSON error strings; model-busy and
runtime failures propagate to the SDK rather than using the HTTP status contract.
Its token-validation path can load weights on Torch. No confidence is calibrated
unless a calibrator is actually configured; the returned `calibrated` flag is the
source of truth, despite older MCP docstrings saying "calibrated".

The HTTP script is a **local demonstration, not a production service**. It has no
authentication, TLS, bounded admission queue, request-read deadline, inference
cancellation, tenant isolation or process supervision. Keep its localhost default;
do not expose it directly to an untrusted network. An application service needs its
own lifecycle, request validation/limits, bounded scheduling, overload policy,
monitoring and failure recovery. Multiple worker processes each load another model
and consume another memory budget. The Torch row limit is not an automatic VRAM
budget: choose model size, context/field limits and concurrency for actual capacity.

Raw softmax confidence is **not accuracy or probability of correctness**. Validate
against representative labelled application data and measure abstention/review
policies. A fitted calibrator and its metadata describe a particular evaluation
scope, not a guarantee on new domains. Multi-select aggregate confidence is not a
joint probability that the entire set is correct. Do not reuse the older 7B model's
accuracy claims for the 0.5B latency model, or treat a high number as permission to
automate consequential decisions without application-level checks.
