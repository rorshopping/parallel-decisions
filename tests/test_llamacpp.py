"""Native-shaped fake: asserts actual sequence IDs, positions and copied logits.

No llama-cpp-python, model downloads or MLX/Torch installation needed.
"""
import ctypes
import math
from types import SimpleNamespace

import numpy as np
import pytest

from parallel_decisions import Calibrator, Config, ConcurrencyError, Decider, Schema
from parallel_decisions.engine_llamacpp import LlamaCppRuntime, decide_llamacpp
from parallel_decisions.prompts import build_prompt


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + list(text.encode())


class Native:
    def __init__(self):
        self.sequences = {}
        self.calls = []
        self.copies = []
        self.clears = 0
        self.fail = False
        self.buffer = np.zeros((4096, 256), dtype=np.float32)

    def llama_memory_clear(self, mem, data):
        assert data is True
        self.sequences.clear()
        self.clears += 1

    def llama_memory_seq_cp(self, mem, src, dst, start, end):
        assert src == 0 and dst != 0 and start == 0
        self.sequences[dst] = list(self.sequences[src][:end])
        self.copies.append((src, dst, end))

    def llama_memory_seq_rm(self, mem, seq, start, end):
        assert seq != 0 and start == end == -1
        self.sequences.pop(seq, None)
        return True

    def llama_decode(self, ctx, batch):
        if self.fail:
            self.fail = False
            return 1
        entries = []
        self.buffer.fill(np.nan)  # reuse outputs on every decode
        for i in range(batch.n_tokens):
            assert batch.n_seq_id[i] == 1
            tok, pos, seq = batch.token[i], batch.pos[i], batch.seq_id[i][0]
            entries.append((tok, pos, seq, batch.logits[i]))
            history = self.sequences.setdefault(seq, [])
            assert pos == len(history), (pos, len(history), seq)
            history.append(tok)
            if batch.logits[i]:
                # Deterministic, context- and row-sensitive causal logits.
                seed = sum((j + 1) * t for j, t in enumerate(history))
                self.buffer[i] = np.sin(np.arange(256) * .7 + seed % 997)
        self.calls.append(entries)
        return 0

    def llama_get_logits_ith(self, ctx, index):
        return self.buffer[index].ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def runtime(n_batch=4096, n_ctx=16000):
    rt = LlamaCppRuntime("unused.gguf", n_ctx=n_ctx, n_batch=n_batch, max_rows=8)
    rt.native = Native()
    rt.np = np
    rt.n_vocab = 256
    rt.n_ctx_seq = n_ctx
    rt.tokenizer = Tokenizer()
    rt.metadata = {}
    rt.ctx = SimpleNamespace(ctx=object())
    rt.memory = object()
    rt.batch = SimpleNamespace(n_tokens=0, token=[0]*n_batch, pos=[0]*n_batch,
                               n_seq_id=[0]*n_batch, seq_id=[[0] for _ in range(n_batch)],
                               logits=[False]*n_batch)
    return rt


def schema():
    return Schema({"first": {"type": "boolean"},
                   "longer_field": {"type": "enum", "choices": ["yes", "no"]},
                   "items": {"type": "multi", "choices": ["one", "two"]},
                   "collision": {"type": "enum", "choices": ["ax", "ayy", "bz"]}})


def call(rt, text="Synthetic context", s=None, **kw):
    s = s or schema()
    return decide_llamacpp(rt, text, s, s.compile(rt.tokenizer), **kw)


def test_one_prefill_true_independent_native_batch_and_clean_request():
    rt = runtime()
    result = call(rt)
    prefill = [c for c in rt.native.calls if c[0][2] == 0]
    assert len(prefill) == 1
    assert [e[0] for e in prefill[0]] == rt.tokenizer.encode(build_prompt("Synthetic context", schema()))
    assert result["telemetry"]["prefill_calls"] == 1
    assert result["telemetry"]["max_parallel_rows"] == 4
    assert any(len({e[2] for e in c}) == 4 for c in rt.native.calls)
    assert all(e[2] != 0 for c in rt.native.calls[1:] for e in c)
    assert rt.native.copies and not rt.native.sequences
    assert rt.native.clears == 2
    assert isinstance(result["values"]["first"].value, bool)
    fv = result["values"]["items"]
    assert fv.value == [c for c, p in fv.distribution.items() if p >= .5]


def test_chunk_singleton_mixed_lengths_collisions_match_and_context_reorder():
    rt = runtime(n_batch=32)
    a = call(rt, "alpha", fields_per_pass=8, max_collision_rows=3)
    b = call(rt, "beta, a different length", fields_per_pass=1, max_collision_rows=1)
    again = call(rt, "alpha", fields_per_pass=1, max_collision_rows=1)
    for name in a["values"]:
        assert a["values"][name].distribution == pytest.approx(again["values"][name].distribution)
    assert a["values"]["first"].distribution != b["values"]["first"].distribution
    assert a["telemetry"]["prefill_calls"] == again["telemetry"]["prefill_calls"] == 1
    assert a["telemetry"]["prefill_decode_calls"] > 1
    assert again["chunks"] == 5
    assert not rt.native.sequences


def test_exact_collision_scores_full_vocab_and_all_answer_tokens():
    rt = runtime()
    s = Schema({"c": {"type": "enum", "choices": ["ax", "ayy", "bz"]}})
    result = call(rt, s=s)
    cf = s.compile(rt.tokenizer)[0]
    assert cf.collision
    prefix = rt.tokenizer.encode(build_prompt("Synthetic context", s)) + cf.suffix_tokens
    lps = []
    for seq in cf.sequences:
        history = list(prefix)
        total = 0.
        for tok in seq:
            seed = sum((j + 1) * t for j, t in enumerate(history))
            scores = np.sin(np.arange(256) * .7 + seed % 997).astype(np.float32).astype(np.float64)
            total += scores[tok] - math.log(np.exp(scores).sum())
            history.append(tok)
        lps.append(total)
    expected = np.exp(lps - np.max(lps))
    expected /= expected.sum()
    assert list(result["values"]["c"].distribution.values()) == pytest.approx(expected)
    hot = call(rt, s=s, temperature=5.)
    assert hot["values"]["c"].distribution == result["values"]["c"].distribution


def test_logits_are_copied_before_native_buffer_reuse():
    rt = runtime(n_batch=8)
    result = call(rt, fields_per_pass=4, max_collision_rows=3)
    for fv in result["values"].values():
        assert all(math.isfinite(p) for p in fv.distribution.values())
    assert result["telemetry"]["parallel_decode_calls"] > 2


def test_decode_exception_cleans_then_reuses_context():
    rt = runtime()
    rt.native.fail = True
    with pytest.raises(RuntimeError, match="llama_decode failed"):
        call(rt)
    assert not rt.native.sequences
    assert call(rt)["telemetry"]["prefill_calls"] == 1


def test_suffix_exception_removes_branches_and_all_request_state(monkeypatch):
    rt = runtime()
    real = rt.native.llama_decode
    def broken(ctx, batch):
        if batch.seq_id[0][0] != 0:
            return -1
        return real(ctx, batch)
    monkeypatch.setattr(rt.native, "llama_decode", broken)
    with pytest.raises(RuntimeError):
        call(rt)
    assert not rt.native.sequences


def test_full_tokenization_boundary_mismatch_fails_before_prefill():
    rt = runtime()
    encode = rt.tokenizer.encode
    rt.tokenizer.encode = lambda text, add_special_tokens=True: (
        encode(text, add_special_tokens) + [2] if '{\n  "' in text
        else encode(text, add_special_tokens))
    with pytest.raises(ValueError, match="boundary mismatch"):
        call(rt)
    assert not rt.native.calls


def test_capacity_does_not_silently_truncate():
    rt = runtime(n_batch=32, n_ctx=64)
    with pytest.raises(ValueError, match="capacity"):
        call(rt)
    assert not rt.native.sequences


@pytest.mark.parametrize("options", [dict(cuda_graph=True), dict(torch_device="cpu"),
    dict(torch_dtype="float32"), dict(memory_budget_gb=1), dict(n_ctx=0),
    dict(n_batch=-1), dict(n_threads=True), dict(max_fields_per_batch=0)])
def test_unsupported_config_fails_closed(options):
    with pytest.raises(ValueError):
        Decider("local.gguf", backend="llamacpp", config=Config(), **options)


def test_local_model_required_and_no_prefix_pretence():
    with pytest.raises(ValueError, match="explicit local"):
        Decider(backend="llamacpp", config=Config())
    d = Decider("missing.gguf", backend="llamacpp", config=Config())
    with pytest.raises(ValueError, match="existing local"):
        d.load()
    with pytest.raises(NotImplementedError):
        d.prepare(schema())
    with pytest.raises(NotImplementedError):
        d.decide_many(["ctx"], schema(), shared_prefix=True)


def test_facade_calibration_lock_cleanup_and_config(monkeypatch):
    rt = runtime()
    d = Decider("local.gguf", backend="llamacpp", warmup=False,
                calibration=Calibrator(), config=Config(n_ctx=8192, n_batch=64, n_threads=2))
    d._llamacpp_rt = rt
    d._model, d._tokenizer = object(), rt.tokenizer
    assert (d.n_ctx, d.n_batch, d.n_threads) == (8192, 64, 2)
    result = d.decide("test context", schema())
    assert result.calibrated and result["first"].calibrated
    assert result.telemetry["backend"] == "llamacpp"
    d._lock.acquire()
    try:
        with pytest.raises(ConcurrencyError):
            d.decide("test", schema())
    finally:
        d._lock.release()
    rt.native.fail = True
    with pytest.raises(RuntimeError):
        d.decide("test", schema())
    assert not d._lock.locked() and not rt.native.sequences


def test_facade_does_not_reuse_stale_compilation_after_schema_mutation():
    rt = runtime()
    d = Decider("local.gguf", backend="llamacpp", config=Config())
    d._llamacpp_rt = rt
    d._model, d._tokenizer = object(), rt.tokenizer
    s = schema()
    d.decide("context", s)
    s.fields["extra"] = Schema({"extra": {"type": "boolean"}})["extra"]
    result = d.decide("context", s)
    assert "extra" in result
    assert result.fields_evaluated == 6


def test_env_config_precedence(monkeypatch, tmp_path):
    from parallel_decisions import load_config
    p = tmp_path / "pd.toml"
    p.write_text('backend="llamacpp"\nmodel="local.gguf"\nn_ctx=1024\nn_batch=64\nn_threads=2\n')
    monkeypatch.setenv("PD_N_CTX", "2048")
    cfg = load_config(str(p))
    assert cfg.n_ctx == 2048 and cfg.n_batch == 64
    assert Decider(config=cfg, n_ctx=4096).n_ctx == 4096
