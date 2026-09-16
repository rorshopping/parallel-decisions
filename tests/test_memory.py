"""Memory-budget tests: the clamp that keeps a 16 GB Mac out of swap.

The rule under test: the budget never lets `model weights + KV broadcast` exceed a
fixed share of physical RAM, because exceeding it does not raise an error — macOS
swaps and the call gets several times slower.
"""

from __future__ import annotations

import pytest

from parallel_decisions.engine import (
    MEMORY_HEADROOM,
    Decider,
    _physical_ram_bytes,
)
from parallel_decisions.config import Config


def _decider(**kwargs) -> Decider:
    return Decider(model_id="unused", warmup=False, config=Config(), **kwargs)


class _FakeArray:
    """Stands in for mx.array where only shape/ndim/nbytes matter."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)

    @property
    def nbytes(self) -> int:
        total = 2                                 # fp16
        for dim in self.shape:
            total *= dim
        return total


def test_physical_ram_is_reported_and_plausible():
    ram = _physical_ram_bytes()
    assert ram is not None
    assert ram > 2 * 1024 ** 3          # any machine that can run this
    assert ram < 4096 * 1024 ** 3


def test_budget_is_clamped_to_the_ram_headroom():
    decider = _decider(memory_budget_gb=64.0)
    decider._model_bytes = 4 * 1024 ** 3
    decider._clamp_memory_budget()
    ram = _physical_ram_bytes()
    assert decider.memory_budget_bytes == int(ram * MEMORY_HEADROOM) - decider._model_bytes
    assert decider.memory_budget_bytes < 64 * 1024 ** 3


def test_budget_below_the_headroom_is_left_alone():
    decider = _decider(memory_budget_gb=1.0)
    decider._model_bytes = 4 * 1024 ** 3
    before = decider.memory_budget_bytes
    decider._clamp_memory_budget()
    assert decider.memory_budget_bytes == before


def test_clamp_is_a_noop_without_a_model_estimate():
    decider = _decider(memory_budget_gb=64.0)
    decider._model_bytes = None
    before = decider.memory_budget_bytes
    decider._clamp_memory_budget()
    assert decider.memory_budget_bytes == before


def test_cache_slots_walks_instance_storage_only():
    """Properties must not be walked.

    `mlx-lm`'s `KVCache.state` is a *view* over keys/values used for serialisation;
    writing a broadcast through it replaces the real cache with a truncated copy. That
    bug silently corrupted the cache during development, so this pins the rule.
    """
    from parallel_decisions.engine import _cache_slots

    class ViewLikeKVCache:
        def __init__(self):
            self.keys = _FakeArray((1, 4, 8, 8))
            self.values = _FakeArray((1, 4, 8, 8))
            self.offset = 0      # a scalar: must not be walked as a per-sequence array

        @property
        def state(self):        # a view, not storage
            return (self.keys, self.values)

    slots = list(_cache_slots(ViewLikeKVCache()))
    assert sorted(name for name, _, _ in slots) == ["keys", "values"]
    assert all(index is None for _, index, _ in slots)


def test_cache_slots_finds_list_entries_and_skips_none():
    from parallel_decisions.engine import _cache_slots

    class ArraysCache:
        def __init__(self):
            self.cache = [_FakeArray((1, 3, 8)), None, _FakeArray((1, 32, 8, 8))]
            self.lengths = None

    slots = {(name, index) for name, index, _ in _cache_slots(ArraysCache())}
    assert slots == {("cache", 0), ("cache", 2)}


def test_repeat_array_only_touches_batch_major_arrays(monkeypatch):
    """Scalars, already-batched arrays and n==1 must not be repeated."""
    from parallel_decisions import engine

    try:
        import mlx.core  # noqa: F401
    except Exception as exc:
        pytest.skip(f"mlx unavailable: {exc}")

    calls = []
    monkeypatch.setattr(engine.mx, "repeat",
                        lambda v, n, axis: calls.append((v.shape, n, axis)) or v)

    assert engine._repeat_array(None, 4) is None
    assert engine._repeat_array(_FakeArray(()), 4).shape == ()           # a scalar
    assert engine._repeat_array(_FakeArray((6, 3)), 4).shape == (6, 3)   # already batched
    assert engine._repeat_array(_FakeArray((1, 3)), 1).shape == (1, 3)   # nothing to do
    assert calls == []
    engine._repeat_array(_FakeArray((1, 3)), 4)
    assert calls == [((1, 3), 4, 0)]


def test_chunk_size_accounts_for_the_constant_per_row_state():
    """A hybrid model's constant state is copied per row, so it must count.

    Qwen3.5 keeps ~49 MB of linear-attention state per sequence. Ignoring it would
    let a 24-row chunk allocate 1.2 GB more than the budget allows.
    """
    decider = _decider(memory_budget_gb=1.0)
    decider.max_fields_per_batch = 32
    decider._kv_bytes_per_token = 1024                 # ~8 MB per row at 8k tokens
    decider._kv_constant_bytes_per_row = 0
    without = decider._auto_chunk_size(8000)
    decider._kv_constant_bytes_per_row = 49 * 1024 ** 2
    with_constant = decider._auto_chunk_size(8000)
    assert with_constant < without
    assert with_constant >= 1


def test_chunk_size_without_a_measurement_uses_max_fields():
    decider = _decider()
    decider.max_fields_per_batch = 7
    decider._kv_bytes_per_token = None
    decider._kv_constant_bytes_per_row = 0
    assert decider._auto_chunk_size(5000) == 7


def test_a_model_larger_than_the_headroom_does_not_produce_a_negative_budget():
    decider = _decider(memory_budget_gb=8.0)
    decider._model_bytes = 512 * 1024 ** 3        # absurd, but must not go negative
    before = decider.memory_budget_bytes
    decider._clamp_memory_budget()
    assert decider.memory_budget_bytes == before


def test_chunk_size_shrinks_with_the_budget():
    decider = _decider(memory_budget_gb=1.0)
    decider._kv_bytes_per_token = 57344            # Qwen2.5-7B shape
    decider.max_fields_per_batch = 32
    small = decider._auto_chunk_size(8_000)         # ~0.43 GB per row
    decider.memory_budget_bytes *= 4                # 4 GB budget
    bigger = decider._auto_chunk_size(8_000)
    assert small == 2
    assert bigger == 9
    assert bigger <= 32                             # never above max_fields_per_batch


def test_max_fields_per_batch_is_an_upper_bound():
    decider = _decider(memory_budget_gb=64.0)
    decider._kv_bytes_per_token = 1024              # tiny model
    decider.max_fields_per_batch = 7
    assert decider._auto_chunk_size(1_000) == 7


def test_chunk_size_is_at_least_one_when_the_context_alone_overflows():
    decider = _decider(memory_budget_gb=1.0)
    decider._kv_bytes_per_token = 57344
    assert decider._auto_chunk_size(1_000_000) == 1
