"""`torch_quant` (bitsandbytes NF4/FP4) plumbing: config, load kwargs, fail-closed.

No GPU and no real weights: `from_pretrained` is faked, CUDA presence is
monkeypatched and bitsandbytes/accelerate are injected as stub modules. A real
CUDA load on the RTX 2060 SUPER is an acceptance step, not a unit test.
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from parallel_decisions import Config, Decider  # noqa: E402
from parallel_decisions.config import ENV_NAMES, load_config  # noqa: E402
from parallel_decisions.engine_torch import (  # noqa: E402
    TorchRuntime,
    normalize_torch_quant,
)


class _FakeModel:
    """Records `to()` calls instead of touching a device."""

    def __init__(self):
        self.to_calls: list = []
        self.eval_called = False

    def to(self, device):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_called = True
        return self


@pytest.fixture(autouse=True)
def clean_pd_env(monkeypatch):
    for name in ENV_NAMES.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("PD_CONFIG", raising=False)


@pytest.fixture
def fake_pretrained(monkeypatch):
    """Fake tokenizer/model loading; records `from_pretrained` kwargs."""
    calls: dict = {}

    def tokenizer_from_pretrained(model_id, **kwargs):
        return types.SimpleNamespace(pad_token_id=0, eos_token_id=0)

    def model_from_pretrained(model_id, **kwargs):
        model = _FakeModel()
        calls["model_id"] = model_id
        calls["kwargs"] = kwargs
        calls["model"] = model
        return model

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        tokenizer_from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        model_from_pretrained)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    return calls


@pytest.fixture
def stub_modules(monkeypatch):
    """bitsandbytes/accelerate presence without importing either for real."""
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))
    monkeypatch.setitem(sys.modules, "accelerate", types.ModuleType("accelerate"))


def test_normalize_torch_quant_accepts_the_two_types_and_none():
    assert normalize_torch_quant(None) is None
    assert normalize_torch_quant("nf4") == "nf4"
    assert normalize_torch_quant("FP4") == "fp4"
    assert normalize_torch_quant(" nf4 ") == "nf4"


@pytest.mark.parametrize("bad", ["int8", "q4", "fp8x"])
def test_normalize_torch_quant_rejects_unknown_values(bad):
    with pytest.raises(ValueError, match="torch_quant"):
        normalize_torch_quant(bad)


@pytest.mark.parametrize("quant", ["nf4", "fp4"])
def test_quantized_load_passes_bitsandbytes_config_and_device_map(
        fake_pretrained, stub_modules, quant):
    rt = TorchRuntime("tiny-local", dtype="float16", device="cuda", quant=quant)
    rt.load()

    kwargs = fake_pretrained["kwargs"]
    config = kwargs["quantization_config"]
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == quant
    assert config.bnb_4bit_use_double_quant is True
    assert config.bnb_4bit_compute_dtype is torch.float16
    assert kwargs["device_map"] == {"": "cuda"}
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["dtype"] is torch.float16
    # Quantized weights are dispatched by device_map; .to() must not run.
    assert fake_pretrained["model"].to_calls == []
    assert fake_pretrained["model"].eval_called


def test_no_quant_leaves_load_kwargs_and_to_untouched(fake_pretrained):
    """torch_quant=None is the pre-existing path: no new kwargs, plain .to()."""
    rt = TorchRuntime("tiny-local", dtype="float16", device="cuda", quant=None)
    rt.load()
    assert rt.quant is None
    assert fake_pretrained["kwargs"] == {"low_cpu_mem_usage": True, "dtype": torch.float16}
    assert fake_pretrained["model"].to_calls == ["cuda"]


def test_quant_requires_a_cuda_device(fake_pretrained):
    rt = TorchRuntime("tiny-local", dtype="float16", device="cpu", quant="nf4")
    with pytest.raises(RuntimeError, match="CUDA"):
        rt.load()
    assert "kwargs" not in fake_pretrained  # failed before any model load


def test_missing_bitsandbytes_fails_closed(monkeypatch, fake_pretrained):
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    rt = TorchRuntime("tiny-local", dtype="float16", device="cuda", quant="nf4")
    with pytest.raises(ImportError, match="bitsandbytes"):
        rt.load()
    assert "kwargs" not in fake_pretrained


def test_missing_accelerate_fails_closed(monkeypatch, fake_pretrained, stub_modules):
    monkeypatch.setitem(sys.modules, "accelerate", None)
    rt = TorchRuntime("tiny-local", dtype="float16", device="cuda", quant="nf4")
    with pytest.raises(ImportError, match="accelerate"):
        rt.load()
    assert "kwargs" not in fake_pretrained


@pytest.mark.parametrize("bad", ["int8", "q4", "fp8"])
def test_decider_rejects_unknown_quant_values(bad):
    with pytest.raises(ValueError, match="torch_quant"):
        Decider("unused", backend="torch", config=Config(), warmup=False,
                torch_quant=bad)
    with pytest.raises(ValueError, match="torch_quant"):
        Decider("unused", backend="torch", config=Config(torch_quant=bad),
                warmup=False)


def test_toml_unknown_quant_fails_when_the_decider_resolves_it(tmp_path):
    path = tmp_path / "pd.toml"
    path.write_text('torch_quant = "int8"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="torch_quant"):
        Decider("unused", backend="torch", config=str(path), warmup=False)


def test_quant_precedence_argument_then_env_then_toml(monkeypatch, tmp_path):
    path = tmp_path / "pd.toml"
    path.write_text('torch_quant = "nf4"\n', encoding="utf-8")
    assert load_config(str(path)).torch_quant == "nf4"
    assert Decider("unused", backend="torch", config=str(path),
                   warmup=False).torch_quant == "nf4"

    monkeypatch.setenv("PD_TORCH_QUANT", "fp4")
    assert load_config(str(path)).torch_quant == "fp4"
    assert Decider("unused", backend="torch", config=str(path),
                   warmup=False).torch_quant == "fp4"

    assert Decider("unused", backend="torch", config=str(path), warmup=False,
                   torch_quant="nf4").torch_quant == "nf4"


def test_decider_quant_plumbing_reaches_from_pretrained(fake_pretrained, stub_modules):
    decider = Decider("tiny-local", backend="torch", config=Config(), warmup=False,
                      torch_device="cuda", torch_dtype="float16", torch_quant="nf4")
    assert decider.torch_quant == "nf4"
    decider.load()

    kwargs = fake_pretrained["kwargs"]
    assert kwargs["quantization_config"].bnb_4bit_quant_type == "nf4"
    assert kwargs["quantization_config"].bnb_4bit_compute_dtype is torch.float16
    assert kwargs["device_map"] == {"": "cuda"}
    assert fake_pretrained["model"].to_calls == []
