"""Model-free regression tests for the optional native evidence collector."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("verify_gguf_boundary", SCRIPTS / "verify_gguf_boundary.py")
boundary = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SCRIPTS))
try:
    spec.loader.exec_module(boundary)
finally:
    sys.path.remove(str(SCRIPTS))


class Native:
    def __init__(self):
        self.positions = {0: 4}
        self.synchronized = False

    def llama_memory_seq_pos_max(self, mem, seq):
        return self.positions.get(seq, -1)

    def llama_memory_seq_cp(self, mem, src, dst, start, end):
        self.positions[dst] = end - 1

    def llama_memory_seq_rm(self, mem, seq, start, end):
        self.positions.pop(seq, None)
        return True

    def llama_memory_clear(self, mem, data):
        self.positions.clear()

    def llama_decode(self, ctx, batch):
        return 0

    def llama_synchronize(self, ctx):
        self.synchronized = True


def test_boundary_records_real_arguments_without_changing_them():
    native = Native()
    events = []
    rt = SimpleNamespace(native=native, memory=object())
    wrapped = boundary.Boundary(rt, events)
    wrapped.llama_memory_seq_cp(rt.memory, 0, 1, 0, 5)
    batch = SimpleNamespace(n_tokens=2, token=[10, 20], pos=[5, 5],
                            n_seq_id=[1, 1], seq_id=[[1], [2]], logits=[False, True])
    assert wrapped.llama_decode(None, batch) == 0
    event = events[-1]
    assert event["entries"] == [[10, 5, [1], False], [20, 5, [2], True]]
    assert event["root_before"] == event["root_after"] == 4
    assert event["synchronized_ms"] >= 0 and native.synchronized
    assert wrapped.llama_memory_seq_rm(rt.memory, 1, -1, -1)
    wrapped.llama_memory_clear(rt.memory, True)
    assert [e["op"] for e in events] == ["copy", "decode", "remove", "clear"]


def test_boundary_detects_wrong_root_range():
    rt = SimpleNamespace(native=Native(), memory=object())
    with pytest.raises(AssertionError):
        boundary.Boundary(rt, []).llama_memory_seq_cp(rt.memory, 0, 1, 0, 3)
