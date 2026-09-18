"""Torch backend unit tests: no model download needed (synthetic tensors)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from transformers.cache_utils import DynamicCache  # noqa: E402

from parallel_decisions.engine_torch import TorchRuntime  # noqa: E402


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
