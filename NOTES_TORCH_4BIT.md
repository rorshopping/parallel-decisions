# Torch 4-bit loading (`torch_quant`): NF4 / FP4

Status: **built, offline-tested; no real CUDA/bitsandbytes load or acceptance run
was performed for this change.** The GPU acceptance run (RTX 2060 SUPER) is a
separate, explicitly staged step. Do not read this note as a measured result.

## Public contract

`backend="torch"` can now load a Hugging Face causal LM in 4-bit via
bitsandbytes. There is one new setting, plumbed like `torch_dtype` /
`torch_device` (explicit kwarg > `PD_TORCH_QUANT` env > `pd.toml` > default
`None`):

```toml
# pd.toml
backend = "torch"
torch_device = "cuda"
torch_dtype = "float16"
torch_quant = "nf4"        # "nf4" | "fp4"; omit for full-precision weights
```

```python
from parallel_decisions import Decider

decider = Decider(
    model_id="Qwen/Qwen2.5-7B-Instruct",   # a local HF model directory works too
    backend="torch",
    torch_device="cuda",
    torch_dtype="float16",                 # required on Turing; see below
    torch_quant="nf4",
)
```

What the loader does when `torch_quant` is set:

- requires the resolved device to be CUDA (CPU 4-bit is rejected);
- imports `bitsandbytes` and `transformers.BitsAndBytesConfig` lazily and fails
  closed with a clear `ImportError` if either is missing;
- builds `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4"|"fp4",
  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=<resolved torch dtype>)`;
- passes `quantization_config=...` and `device_map={"": "cuda"}` (the device
  string) to `AutoModelForCausalLM.from_pretrained`, keeping
  `low_cpu_mem_usage=True`;
- does **not** call `model.to(device)` afterwards — bitsandbytes modules are
  placed by `device_map`, and moving them afterwards is unsupported;
- still requires `accelerate` for `device_map` and fails closed with a clear
  `ImportError` when it is absent.

Unknown values (`torch_quant="int8"`, `PD_TORCH_QUANT=q4`, a bad `pd.toml`
entry) raise `ValueError` at `Decider` construction, before any model load.
`torch_quant=None` keeps the previous loader byte-for-byte: same kwargs, same
`.to()` call.

Install the two optional packages alongside the `torch` extra:

```bash
pip install bitsandbytes accelerate
```

They are **not** added to declared package extras by this change; the
`.[torch]` extra is unchanged.

## Turing / RTX 2060 SUPER (SM75)

Turing has fp16 tensor cores but no native bf16. Pass `torch_dtype="float16"`
explicitly (as above); the resolved dtype also becomes the 4-bit compute dtype.
Turing reports bf16 "availability" in some PyTorch builds via emulation, and
`engine.py`'s auto-dtype logic can therefore select bf16 on CUDA — that is why
the 4-bit contract is documented with an explicit fp16 dtype.

## VRAM and latency caveats

- Target is roughly 4.5 GB of weights+overhead for a 7B model: about 3.5 GB of
  4-bit weights plus double-quant constants, and the embedding table typically
  stays fp16 (it is not a `nn.Linear`), so budget headroom above the raw 4-bit
  weight size. **No VRAM measurement was made here**; treat ~4.5 GB as a design
  target, not a verified number.
- KV cache, activations and logits are unchanged by weight quantization; long
  prompts still grow memory with context length and row broadcast, and the
  Torch backend still chunks only by `max_fields_per_batch`.
- 4-bit compute dequantizes on the fly; expect **slower** per-token compute than
  fp16 on Turing. No latency benchmark was run for this change. Do not quote
  MLX or 0.5B PyTorch timings as 4-bit 7B latency.
- `device_map={"": ...}` with a single device does not offload; nothing here
  supports multi-GPU sharding, and `bnb_4bit_compute_dtype` is fp16/fp32
  depending on the configured `torch_dtype`.

## Verified vs. not

Verified by offline unit tests (`tests/test_torch_quant.py`, no GPU, faked
`from_pretrained` and stubbed `bitsandbytes`/`accelerate`):

- the exact `quantization_config` fields (`load_in_4bit`, quant type,
  double quant, compute dtype) for both NF4 and FP4;
- `device_map={"": <device>}`, `low_cpu_mem_usage=True`, dtype passthrough;
- `model.to()` is not called when quantized, and is unchanged when not;
- config precedence kwarg > env > `pd.toml`, and case normalization;
- fail-closed behavior for a non-CUDA device, unknown quant values, and missing
  `bitsandbytes` / `accelerate`.

Not verified (no GPU run was performed here):

- an actual 7B NF4 load on the RTX 2060 SUPER, VRAM in use, or whether the
  ~4.5 GB target is met;
- bitsandbytes kernels, dequantization correctness, throughput or latency;
- any change in model answers, probabilities or calibration from 4-bit weights.

## No accuracy claim

4-bit quantization changes weights and can change probabilities and near-tied
answers. Raw softmax output is not a probability of correctness. This change
makes **no accuracy claim** for NF4, FP4, or any model — neither against the
MLX 7B numbers nor the PyTorch 0.5B measurements. Accuracy requires a labelled
evaluation on the quantized model and calibrated confidence, which has not been
run.
