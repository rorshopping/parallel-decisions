"""Memory-budget tests: the clamp that keeps a 16 GB Mac out of swap.

The rule under test: the budget never lets `model weights + KV broadcast` exceed a
fixed share of physical RAM, because exceeding it does not raise an error — macOS
swaps and the call gets several times slower.
"""

from __future__ import annotations

from parallel_decisions.engine import (
    MEMORY_HEADROOM,
    Decider,
    _physical_ram_bytes,
)
from parallel_decisions.config import Config


def _decider(**kwargs) -> Decider:
    return Decider(model_id="unused", warmup=False, config=Config(), **kwargs)


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


def test_cache_arrays_finds_any_cache_shape():
    """Config-based KV estimates break on hybrid models, so the engine measures the
    cache; that only works if every array in any cache class is found."""
    from parallel_decisions.engine import _cache_arrays

    class Array:
        def __init__(self, nbytes):
            self.nbytes = nbytes

    class KVCache:
        keys = Array(100)
        values = Array(120)

    class HybridCache:          # linear-attention layers add a constant-size state
        keys = Array(50)
        state = Array(1000)

    class NestedCache:          # some caches keep a list in `.cache`
        cache = [Array(10), Array(20)]

    class Empty:
        pass

    assert sorted(a.nbytes for a in _cache_arrays(KVCache())) == [100, 120]
    assert sorted(a.nbytes for a in _cache_arrays(HybridCache())) == [50, 1000]
    assert sorted(a.nbytes for a in _cache_arrays(NestedCache())) == [10, 20]
    assert list(_cache_arrays(Empty())) == []


def test_cache_arrays_does_not_loop_forever_on_self_reference():
    from parallel_decisions.engine import _cache_arrays

    class Looping:
        pass

    loop = Looping()
    loop.cache = [loop]
    assert list(_cache_arrays(loop)) == []


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
