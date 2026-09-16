"""Prefix regression tests with a tiny random CPU model; no downloads or MLX."""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from parallel_decisions import Config, Decider, Schema
from parallel_decisions.engine import ConcurrencyError
from parallel_decisions.engine_torch import TorchRuntime
from parallel_decisions.prompts import build_prompt


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(c) % 255 + 1 for c in text]


@pytest.fixture
def decider():
    torch.manual_seed(7)
    config = transformers.Qwen2Config(
        vocab_size=256, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=4096, attention_dropout=0.0,
    )
    rt = TorchRuntime("tiny-local", device="cpu")
    rt.torch = torch
    rt.device = "cpu"
    rt.dtype = torch.float32
    rt.tokenizer = Tokenizer()
    rt.model = transformers.Qwen2ForCausalLM(config).eval()
    d = Decider("tiny-local", backend="torch", config=Config(), warmup=False,
                max_fields_per_batch=1)
    d._torch_rt = rt
    d._model = rt.model
    d._tokenizer = rt.tokenizer
    return d


@pytest.fixture
def schema():
    # Char tokenizer: AB/AC collide on A (no common prefix across all choices).
    return Schema({
        "kind": {"type": "enum", "choices": ["AB", "AC", "Z"]},
        "flag": {"type": "boolean", "description": "a boolean"},
    })


def test_prefix_matches_full_and_preserves_cache(decider, schema):
    prefix = decider.prepare(schema)
    assert prefix.reusable
    original = copy.deepcopy(prefix.cache)
    for context in ("one", " another\nline", "日本語"):
        full = decider.decide(context, schema)
        reused = decider.decide_with_prefix(prefix, context)
        assert reused.json() == full.json()
        assert reused.chunks == 2
        assert reused.telemetry["shared_prefix"] is True
        for name in schema:
            assert reused[name].distribution == pytest.approx(
                full[name].distribution, abs=1e-6)
        assert prefix.cache.get_seq_length() == original.get_seq_length()
        for before, after in zip(original.layers, prefix.cache.layers):
            assert torch.equal(before.keys, after.keys)
            assert torch.equal(before.values, after.values)


def test_prepare_holds_model_lock(decider, schema):
    with decider._acquire_lock("test"):
        with pytest.raises(ConcurrencyError):
            decider.prepare(schema)


def test_wrong_owner_rejected_and_release_falls_back(decider, schema):
    prefix = decider.prepare(schema)
    other = Decider("tiny-local", backend="torch", config=Config())
    with pytest.raises(ValueError, match="different Decider"):
        other.decide_with_prefix(prefix, "hello")
    prefix.release()
    prefix.release()
    assert not prefix.reusable
    full = decider.decide("hello", schema)
    reused = decider.decide_with_prefix(prefix, "hello")
    assert full.json() == reused.json()
    assert reused.telemetry.get("shared_prefix", False) is False


def test_context_specific_boundary_merge_falls_back(decider, schema):
    prefix = decider.prepare(schema)
    context = "boundary merge"
    full_text = build_prompt(context, schema)
    encode = decider._tokenizer.encode

    def merging(text, **kwargs):
        result = encode(text, **kwargs)
        if text == full_text:
            result[len(prefix.prefix_tokens) - 1] = 255
        return result

    decider._tokenizer.encode = merging
    full = decider.decide(context, schema)
    reused = decider.decide_with_prefix(prefix, context)
    assert reused.json() == full.json()
    assert reused.telemetry["shared_prefix"] is False
    assert reused.telemetry["prompt_tokens"] == len(merging(full_text))


def test_prepare_rejects_probe_boundary_merge(decider, schema):
    probe = build_prompt("x", schema)
    encode = decider._tokenizer.encode

    def merging(text, **kwargs):
        result = encode(text, **kwargs)
        if text == probe:
            result[0] = 255
        return result

    decider._tokenizer.encode = merging
    assert not decider.prepare(schema).reusable


def test_bos_is_not_duplicated(decider, schema):
    encode = decider._tokenizer.encode
    decider._tokenizer.encode = lambda text, **kw: [254] + encode(text, **kw)
    prefix = decider.prepare(schema)
    full = decider.decide("hello", schema)
    reused = decider.decide_with_prefix(prefix, "hello")
    assert reused.telemetry["shared_prefix"]
    assert reused.telemetry["prompt_tokens"] == full.telemetry["prompt_tokens"]
    for name in schema:
        assert reused[name].distribution == pytest.approx(full[name].distribution, abs=1e-6)


def test_release_and_use_obey_lock(decider, schema):
    prefix = decider.prepare(schema)
    with decider._acquire_lock("test"):
        with pytest.raises(ConcurrencyError):
            prefix.release()
        with pytest.raises(ConcurrencyError):
            decider.decide_with_prefix(prefix, "hello")
    assert prefix.reusable


def test_unknown_cache_copy_does_not_alias_mutable_state():
    from types import SimpleNamespace
    state = torch.ones(2)
    cache = SimpleNamespace(layers=[SimpleNamespace(states={"r": state})])
    copied = TorchRuntime._copy_torch_cache(cache)
    copied.layers[0].states["r"].zero_()
    assert torch.equal(state, torch.ones(2))


def test_prefix_calibration_matches_full(decider, schema):
    from parallel_decisions import Calibrator
    decider.calibrator = Calibrator(kind="temperature", temperature=2.0)
    prefix = decider.prepare(schema)
    full = decider.decide("hello", schema)
    reused = decider.decide_with_prefix(prefix, "hello")
    assert reused.calibrated and reused.json() == full.json()
    for name in schema:
        assert reused[name].probability == pytest.approx(full[name].probability, abs=1e-6)


def test_schema_prefill_runs_only_once(decider, schema, monkeypatch):
    """Spy on actual tiny CPU-model prefills; never reload the shared schema."""
    rt = decider._torch_rt
    prefill = rt.prefill
    calls = []

    def record(tokens, cache=None):
        calls.append((list(tokens), cache is not None))
        return prefill(tokens, cache=cache)

    monkeypatch.setattr(rt, "prefill", record)
    prefix = decider.prepare(schema)
    assert calls == [(list(prefix.prefix_tokens), False)]
    for context in ("first", "second"):
        decider.decide_with_prefix(prefix, context)
        whole = rt.tokenizer.encode(build_prompt(context, schema))
        assert calls[-1] == (whole[len(prefix.prefix_tokens):], True)
    assert len(calls) == 3  # schema once, then two context-only extensions
    assert sum(not extending for _, extending in calls) == 1
    prefix.release()


def test_decide_many_uses_prefix(decider, schema):
    results = decider.decide_many(["one", "two"], schema, shared_prefix=True)
    assert len(results) == 2
    assert all(r.telemetry["shared_prefix"] for r in results)
