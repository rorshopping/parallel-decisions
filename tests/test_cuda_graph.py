"""No-download CUDA graph regressions; configuration/CPU checks need no GPU."""
from __future__ import annotations

import pytest

from parallel_decisions import Config, Decider, Schema
from parallel_decisions.config import load_config
from parallel_decisions.engine_torch import TorchRuntime


@pytest.fixture
def runtime(request):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    device = getattr(request, 'param', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(7)
    rt = TorchRuntime('tiny-local', device=device, cuda_graph=True)
    rt.torch, rt.device, rt.dtype = torch, device, torch.float32
    rt.model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=256, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=4096, attention_dropout=0.0,
    )).to(device).eval()

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return [ord(c) % 255 + 1 for c in text]

    rt.tokenizer = Tokenizer()
    return rt


def test_graph_replay_fresh_buffers_and_bucket_lengths(runtime):
    rt = runtime
    # First call eager; second captures. Subsequent replays change tokens, KV,
    # prompt length (within a bucket), row padding, and then shrink the prompt.
    retained = []
    for i, prompt in enumerate(([1, 2, 3], [8, 9, 10, 11], [12, 13], [20] * 7)):
        cache, _ = rt.prefill(list(prompt))
        before = cache.layers[0].keys.clone()
        rows = [[4 + i, 5 + i], [6 + i]]
        eager, arr = rt._batched_pass_eager(cache, rows, 0)
        actual, actual_arr = rt.batched_pass(cache, rows, 0)
        rt.synchronize()
        rt.torch.testing.assert_close(actual, eager, atol=1e-5, rtol=0)
        assert rt.torch.equal(arr, actual_arr)
        assert rt.torch.equal(before, cache.layers[0].keys)
        assert not rt.graph_cache.disabled
        assert len(rt.graph_cache.entries) == (0 if i == 0 else 1)
        retained.append((actual, actual.clone()))
    assert rt.graph_cache.replays == 3
    assert rt.graph_cache.capture_ms > 0
    for actual, snapshot in retained:
        assert rt.torch.equal(actual, snapshot)
    # A new row/suffix shape is not captured on first use.
    rt.batched_pass(cache, [[7, 8, 9]], 0)
    assert len(rt.graph_cache.entries) == 1


@pytest.mark.parametrize('failure', ['capture', 'replay'])
def test_graph_failure_falls_back_once(runtime, monkeypatch, caplog, failure):
    rt = runtime
    cache, _ = rt.prefill([1, 2, 3])
    rows = [[4, 5], [6, 7]]
    expected, _ = rt._batched_pass_eager(cache, rows, 0)
    rt.batched_pass(cache, rows, 0)  # first sighting stays eager
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('injected graph failure')

    if failure == 'capture':
        monkeypatch.setattr(rt.graph_cache, '_capture', broken)
    else:
        rt.batched_pass(cache, rows, 0)
        entry = next(iter(rt.graph_cache.entries.values()))
        class BrokenGraph:
            replay = broken
        entry['graph'] = BrokenGraph()
    for _ in range(2):
        actual, _ = rt.batched_pass(cache, rows, 0)
        rt.torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
    assert len(calls) == 1
    assert rt.graph_cache.disabled and not rt.graph_cache.entries
    assert caplog.text.count('CUDA graph fallback') == 1


@pytest.mark.parametrize('runtime', ['cpu'], indirect=True)
def test_cpu_never_routes_to_graph(runtime, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('CPU must never attempt graphs')
    monkeypatch.setattr(runtime.graph_cache, 'run', forbidden)
    cache, _ = runtime.prefill([1, 2, 3])
    for _ in range(3):
        eager, _ = runtime._batched_pass_eager(cache, [[4, 5]], 0)
        actual, _ = runtime.batched_pass(cache, [[4, 5]], 0)
        assert runtime.torch.equal(actual, eager)


def test_cuda_graph_config(tmp_path, monkeypatch):
    monkeypatch.delenv('PD_CUDA_GRAPH', raising=False)
    path = tmp_path / 'pd.toml'
    path.write_text('cuda_graph = true\nbackend = "torch"\n', encoding='utf-8')
    assert load_config(str(path)).cuda_graph is True
    assert Decider(config=str(path)).cuda_graph is True
    monkeypatch.setenv('PD_CUDA_GRAPH', 'false')
    assert load_config(str(path)).cuda_graph is False
    assert Decider(config=str(path), cuda_graph=True).cuda_graph is True
    monkeypatch.setenv('PD_CUDA_GRAPH', 'on')
    assert load_config(str(path)).cuda_graph is True
    assert Decider(config=str(path), cuda_graph=False).cuda_graph is False
    assert Decider(config=Config()).cuda_graph is False


def test_facade_passes_graph_option(monkeypatch):
    calls = []
    def load(rt):
        calls.append(rt.cuda_graph)
        rt.model, rt.tokenizer, rt.device = object(), object(), 'cpu'
        return rt
    monkeypatch.setattr(TorchRuntime, 'load', load)
    d = Decider('tiny-local', backend='torch', cuda_graph=True, warmup=False,
                config=Config())
    d.load()
    assert calls == [True]


def test_graph_prefix_collisions_chunks(runtime):
    rt = runtime
    d = Decider('tiny-local', backend='torch', cuda_graph=True, warmup=False,
                max_fields_per_batch=1, config=Config())
    d._torch_rt, d._model, d._tokenizer = rt, rt.model, rt.tokenizer
    schema = Schema({'kind': {'type': 'enum', 'choices': ['AB', 'AC', 'Z']},
                     'flag': {'type': 'boolean'}})
    prefix = d.prepare(schema)
    before = prefix.cache.layers[0].keys.clone()
    for context in ('one', 'two', 'longer', 'short'):
        rt.cuda_graph = False
        eager = d.decide(context, schema)
        rt.cuda_graph = True
        actual = d.decide_with_prefix(prefix, context)
        assert actual.json() == eager.json()
        assert actual.telemetry['shared_prefix']
        for name in schema:
            assert actual[name].distribution == pytest.approx(
                eager[name].distribution, abs=1e-5)
        assert rt.torch.equal(prefix.cache.layers[0].keys, before)
    assert rt.graph_cache.replays > 0 and not rt.graph_cache.disabled
    prefix.release()
