"""Torch backend unit tests: no model download needed (synthetic tensors)."""

from __future__ import annotations

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
