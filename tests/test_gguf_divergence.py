"""Frozen real Qwen2 token fixture; fake proves packing, NOT native numerics."""
import ctypes
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from parallel_decisions.engine_llamacpp import LlamaCppRuntime, _log_probability, _softmax

# Synthetic acceptance 'long' prompt, captured with native tokenizer 0.3.35.
PROMPT = [
    151644, 8948, 198, 1957, 1437, 4718, 8201, 510, 220, 330, 33198, 457,
    788, 2160, 279, 29309, 44250, 5267, 220, 330, 3423, 788, 96364, 29309,
    1894, 14566, 25, 25810, 26, 55892, 26, 53559, 198, 220, 330, 14082,
    788, 96364, 29309, 9492, 508, 26268, 20458, 60, 4226, 830, 476, 895,
    369, 1817, 30581, 1376, 330, 14082, 989, 19177, 320, 2047, 600, 17180,
    304, 279, 25993, 568, 14566, 25, 9058, 11, 3100, 11, 8811, 198, 220,
    330, 4825, 71834, 70375, 5013, 788, 96364, 29309, 17133, 14566, 25,
    29309, 2518, 26, 29309, 6303, 26, 1008, 151645, 198, 151644, 872, 198,
    37134, 17556, 15444, 5185, 25, 27645, 8094, 5610, 19046, 4287, 14697, 13,
] + [74828, 15444, 5185, 25, 27645, 8094, 5610, 19046, 4287, 14697, 13] * 23 + [
    42994, 3255, 25, 44250, 902, 26, 1894, 53559, 26, 1008, 26, 9492, 9058,
    323, 8811, 13, 151645, 198, 151644, 77091, 198, 515,
]
SUFFIXES = [
    [220, 330, 33198, 457, 788, 220],
    [220, 330, 3423, 788, 330],
    [220, 330, 14082, 58, 15, 60, 788, 220],
    [220, 330, 14082, 58, 16, 60, 788, 220],
    [220, 330, 14082, 58, 17, 60, 788, 220],
]


def test_diagnostic_help_and_opt_in_gate_without_site_packages(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_gguf_divergence.py"
    result = subprocess.run([sys.executable, "-S", str(script), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    result = subprocess.run([sys.executable, "-S", str(script), "--model", "missing.gguf",
                             "--output", str(tmp_path / "out.json")],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "requires --run-native --exclusive-model-use" in result.stderr
    assert not (tmp_path / "out.json").exists()


class CompactNative:
    def __init__(self):
        self.histories = {0: list(PROMPT)}
        self.outputs = {}
        self.calls = []

    def llama_memory_seq_cp(self, mem, src, dst, start, end):
        assert (src, start, end) == (0, 0, 383)
        self.histories[dst] = list(self.histories[src])

    def llama_memory_seq_rm(self, mem, seq, start, end):
        assert start == end == -1
        del self.histories[seq]
        return True

    @staticmethod
    def logits(history):
        seed = sum((i + 1) * t for i, t in enumerate(history)) % 997
        return np.sin(np.arange(152064) * .7 + seed).astype(np.float32)

    def llama_decode(self, ctx, batch):
        for value in self.outputs.values():
            value.fill(np.nan)
        self.outputs = {}
        entries = []
        for i in range(batch.n_tokens):
            assert batch.n_seq_id[i] == 1
            seq, pos, tok = batch.seq_id[i][0], batch.pos[i], batch.token[i]
            history = self.histories[seq]
            assert pos == len(history)
            history.append(tok)
            entries.append((tok, pos, seq, bool(batch.logits[i])))
            if batch.logits[i]:
                self.outputs[i] = self.logits(history)
        self.calls.append(entries)
        return 0

    def llama_get_logits_ith(self, ctx, index):
        # Mapping uses the original sparse token index, not compact output rank.
        return self.outputs[index].ctypes.data_as(ctypes.POINTER(ctypes.c_float))


@pytest.mark.parametrize("capacity", [8, 512])
def test_frozen_qwen_rows_sparse_outputs_chunking_and_order(capacity):
    assert len(PROMPT) == 383
    rt = LlamaCppRuntime("unused.gguf", n_batch=capacity, max_rows=8)
    rt.native, rt.np, rt.n_vocab = CompactNative(), np, 152064
    rt.n_ctx_seq, rt.prompt_length = 4096, 383
    rt.ctx, rt.memory = SimpleNamespace(ctx=None), None
    rt.batch = SimpleNamespace(n_tokens=0, token=[0]*capacity, pos=[0]*capacity,
                               n_seq_id=[0]*capacity, seq_id=[[0] for _ in range(capacity)],
                               logits=[False]*capacity)
    rt.stats = dict(decode_calls=0, max_parallel_rows=0, parallel_decode_calls=0,
                    sequence_copies=0)
    retained = []
    for limit in (8, 2, 1):
        for order in (SUFFIXES, list(reversed(SUFFIXES))):
            for start in range(0, len(order), limit):
                rows = order[start:start+limit]
                out = rt.batched_pass(rows, [{len(row)-1} for row in rows])
                for tokens, result in zip(rows, out):
                    z = result[len(tokens)-1]
                    expected = CompactNative.logits(PROMPT + tokens)
                    np.testing.assert_array_equal(z, expected)
                    retained.append((z, expected))
                    # Python float conversion and float64 normalization cannot
                    # account for a native raw-logit shape discrepancy.
                    if tokens == SUFFIXES[3]:
                        lp = [_log_probability(rt, z, t) for t in (1866, 3849)]
                        assert _softmax(lp) == pytest.approx(
                            _softmax([float(z[t]) for t in (1866, 3849)]), abs=1e-12)
    for z, expected in retained:
        np.testing.assert_array_equal(z, expected)
    assert rt.native.histories == {0: PROMPT}
    assert any(len({e[2] for e in call}) == 5 for call in rt.native.calls)
    assert any(not e[3] for call in rt.native.calls for e in call)
