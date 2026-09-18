"""Torch backend unit tests: no model download needed (synthetic tensors)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from transformers.cache_utils import DynamicCache  # noqa: E402

from parallel_decisions.engine_torch import (  # noqa: E402
    TorchRuntime,
    resolve_collision_rows,
    slice_decision,
)
from parallel_decisions.schema import CompiledField, Field  # noqa: E402


def _runtime() -> TorchRuntime:
    rt = TorchRuntime.__new__(TorchRuntime)   # bypass load(); broadcast is pure
    rt.torch = torch
    return rt


def _cache_with_tokens(n_tokens: int) -> DynamicCache:
    cache = DynamicCache()
    keys = torch.randn(1, 2, n_tokens, 6)
    values = torch.randn(1, 2, n_tokens, 6)
    cache.update(keys, values, 0)
    return cache


def test_broadcast_does_not_mutate_the_callers_cache():
    """The bug the email-triage lab caught: a shallow copy shares the layer objects,
    and HF's in-place repeat corrupted the prompt cache, so any second pass over it
    (collision rows, chunking, prefix reuse) crashed in torch.cat."""
    rt = _runtime()
    cache = _cache_with_tokens(8)
    repeated = rt.broadcast(cache, 3)
    assert tuple(repeated.layers[0].keys.shape) == (3, 2, 8, 6)
    assert tuple(cache.layers[0].keys.shape) == (1, 2, 8, 6)
    cache.update(torch.randn(1, 2, 1, 6), torch.randn(1, 2, 1, 6), 0)
    assert tuple(cache.layers[0].keys.shape) == (1, 2, 9, 6)


def test_broadcast_with_n_one_returns_a_usable_cache():
    rt = _runtime()
    cache = _cache_with_tokens(5)
    repeated = rt.broadcast(cache, 1)
    assert tuple(repeated.layers[0].keys.shape) == (1, 2, 5, 6)


def test_broadcast_repeats_values_too():
    rt = _runtime()
    cache = _cache_with_tokens(7)
    repeated = rt.broadcast(cache, 4)
    assert tuple(repeated.layers[0].values.shape) == (4, 2, 7, 6)


# ---- chunked prefill ------------------------------------------------------ #

class _RecordingModel:
    """Stand-in for a HF causal LM: records segment sizes and threads the cache."""

    def __init__(self) -> None:
        self.segments: list[int] = []
        self.caches: list[object] = []

    def __call__(self, inputs, use_cache=True, past_key_values=None):
        self.segments.append(int(inputs.shape[1]))
        self.caches.append(past_key_values)
        return SimpleNamespace(past_key_values=f"cache-{len(self.segments)}")


def _prefill_runtime(chunk):
    rt = TorchRuntime.__new__(TorchRuntime)  # bypass load(); prefill path is pure
    rt.torch = torch
    rt.device = "cpu"
    rt.verbose = False
    rt.model = _RecordingModel()
    rt.prefill_chunk = chunk
    return rt


def test_prefill_chunks_long_prompts_and_threads_the_cache():
    rt = _prefill_runtime(4)
    cache, count = rt.prefill(list(range(10)))
    assert rt.model.segments == [4, 4, 2]
    assert rt.model.caches == [None, "cache-1", "cache-2"]
    assert cache == "cache-3"
    assert count == 10


def test_prefill_chunk_zero_is_a_single_pass():
    rt = _prefill_runtime(0)
    cache, count = rt.prefill(list(range(10)))
    assert rt.model.segments == [10]
    assert count == 10


def test_prefill_at_chunk_size_is_a_single_pass():
    rt = _prefill_runtime(4)
    rt.prefill(list(range(4)))
    assert rt.model.segments == [4]


def test_prefill_appends_to_an_existing_cache():
    rt = _prefill_runtime(2)
    cache, count = rt.prefill(list(range(3)), cache="existing")
    assert rt.model.caches == ["existing", "cache-1"]
    assert cache == "cache-2"
    assert count == 3


# ---- decision slicing and collision scoring ------------------------------- #

def test_slice_decision_matches_the_scalar_reference():
    import torch.nn.functional as F

    torch.manual_seed(5)
    row = torch.randn(64, dtype=torch.float32)
    cf = SimpleNamespace(candidate_ids=[[3], [7], [7, 9], []])
    scores = [max(float(row[i]) for i in ids) if ids else -1e9 for ids in cf.candidate_ids]
    expected = F.softmax(torch.tensor(scores, dtype=torch.float32) / 0.5, dim=-1).tolist()

    got = slice_decision(None, cf, row, 0.5)
    assert got == pytest.approx(expected, rel=1e-6)

    # fp16 logits must not disturb the fp32 softmax path.
    got16 = slice_decision(None, cf, row.half(), 0.5)
    assert got16 == pytest.approx(expected, rel=2e-3)
    assert sum(got16) == pytest.approx(1.0, abs=1e-4)


class _CharTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(c) % 60 + 1 for c in text]


def _tiny_collision_runtime() -> TorchRuntime:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(11)
    config = transformers.Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512,
        attention_dropout=0.0,
    )
    rt = TorchRuntime.__new__(TorchRuntime)
    rt.model_id = "tiny-local"
    rt.torch = torch
    rt.device = "cpu"
    rt.dtype = torch.float32
    rt.verbose = False
    rt.cuda_graph = False
    rt.prefill_chunk = 2048
    rt.tokenizer = _CharTokenizer()
    rt.model = transformers.Qwen2ForCausalLM(config).eval()
    return rt


def _reference_collision_rows(rt, cache, groups, max_collision_rows):
    """The element-wise implementation this optimization replaced."""
    import torch.nn.functional as F

    rows, spans = [], []
    for cf in groups:
        suffix_len = len(cf.suffix_tokens)
        for ci, seq in enumerate(cf.sequences):
            rows.append(cf.suffix_tokens + seq)
            spans.append((cf.row_name, ci, suffix_len, suffix_len + len(seq)))

    log_probs: dict[str, dict[int, float]] = {}
    pad = rt.pad_id()
    cursor = 0
    while cursor < len(rows):
        batch_rows = rows[cursor:cursor + max_collision_rows]
        batch_spans = spans[cursor:cursor + max_collision_rows]
        logits, arr = rt.batched_pass(cache, batch_rows, pad)
        log_softmax = F.log_softmax(logits.float(), dim=-1)
        for local, (name, ci, start, end) in enumerate(batch_spans):
            total = 0.0
            for pos in range(start, end):
                total += float(log_softmax[local, pos - 1, int(arr[local, pos])])
            log_probs.setdefault(name, {})[ci] = total
        cursor += len(batch_rows)
    return log_probs


def test_collision_rows_match_the_elementwise_reference():
    rt = _tiny_collision_runtime()
    cache, _ = rt.prefill([1, 2, 0, 3, 4, 5])
    cf = CompiledField(
        field=Field(name="kind", type="enum", choices=["AB", "AC", "AD"]),
        suffix='  "kind": "A',
        suffix_tokens=[9, 10, 11, 12],
        candidate_ids=[[13], [14], [15]],
        sequences=[[13], [14, 21], [15]],   # mixed lengths exercise the band slicing
        collision=True,
    )
    new = resolve_collision_rows(rt, cache, [cf], max_collision_rows=2)
    ref = _reference_collision_rows(rt, cache, [cf], max_collision_rows=2)

    assert set(new["kind"]) == set(ref["kind"])
    for ci in ref["kind"]:
        assert new["kind"][ci] == pytest.approx(ref["kind"][ci], abs=1e-5)
