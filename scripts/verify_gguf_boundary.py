"""Opt-in synthetic native boundary trace plus independent scalar full-row oracle.

Run with the same arguments as verify_gguf_native.py. Adds a sibling .trace.json
report. Never use this collector with private/user contexts: token IDs are logged.
The scalar oracle deliberately re-prefills each candidate on seq 0; this is
validation-only and never a fallback in the production Decider path.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import verify_gguf_native as acceptance

sys.path.insert(0, str(acceptance.ROOT / "src"))
from parallel_decisions.engine_llamacpp import LlamaCppRuntime
from parallel_decisions.prompts import build_prompt


class Boundary:
    def __init__(self, runtime, events):
        self.rt = runtime
        self.native = runtime.native
        self.events = events
        self.phase = "idle"
        self.request = 0
        self.chunk = 0

    def __getattr__(self, name):
        return getattr(self.native, name)

    def record(self, op, **data):
        self.events.append(dict(op=op, request=self.request, chunk=self.chunk,
                                phase=self.phase, **data))

    def root_max(self):
        return self.native.llama_memory_seq_pos_max(self.rt.memory, 0)

    def llama_decode(self, ctx, batch):
        entries = [[int(batch.token[i]), int(batch.pos[i]),
                    [int(batch.seq_id[i][j]) for j in range(batch.n_seq_id[i])],
                    bool(batch.logits[i])] for i in range(batch.n_tokens)]
        before = self.root_max()
        start = time.perf_counter()
        code = self.native.llama_decode(ctx, batch)
        self.native.llama_synchronize(ctx)
        self.record("decode", entries=entries, return_code=code,
                    root_before=before, root_after=self.root_max(),
                    synchronized_ms=(time.perf_counter() - start) * 1000)
        return code

    def llama_get_logits_ith(self, ctx, index):
        result = self.native.llama_get_logits_ith(ctx, index)
        self.record("get_logits_ith", batch_token_index=index, present=bool(result))
        return result

    def llama_memory_seq_cp(self, mem, src, dst, start, end):
        before = self.root_max()
        result = self.native.llama_memory_seq_cp(mem, src, dst, start, end)
        self.record("copy", src=src, dst=dst, start=start, end=end,
                    root_before=before, root_after=self.root_max(),
                    dest_max=self.native.llama_memory_seq_pos_max(mem, dst))
        assert before == end - 1 == self.root_max()
        assert self.native.llama_memory_seq_pos_max(mem, dst) == before
        return result

    def llama_memory_seq_rm(self, mem, seq, start, end):
        before = self.root_max()
        result = self.native.llama_memory_seq_rm(mem, seq, start, end)
        self.record("remove", seq=seq, start=start, end=end, result=result,
                    root_before=before, root_after=self.root_max(),
                    dest_max=self.native.llama_memory_seq_pos_max(mem, seq))
        assert before == self.root_max()
        assert self.native.llama_memory_seq_pos_max(mem, seq) == -1
        return result

    def llama_memory_clear(self, mem, data):
        result = self.native.llama_memory_clear(mem, data)
        self.record("clear", data=data, root_after=self.root_max())
        assert self.root_max() == -1
        return result

    def llama_synchronize(self, ctx):
        result = self.native.llama_synchronize(ctx)
        self.record("synchronize")
        return result


def scalar_oracle(decider, schema, compiled):
    """Independent low-level full-sequence teacher forcing, no runtime pass helpers."""
    rt = decider._llamacpp_rt
    n = rt.native
    n.phase = "scalar_oracle"
    prefix = rt.tokenizer.encode(build_prompt(acceptance.CONTEXTS["a"], schema))
    results = {}
    batch = n.llama_batch_init(rt.n_batch, 0, 1)
    try:
        for cf in compiled:
            totals = []
            ordinary_logits = None
            for answer in (cf.sequences if cf.collision else [[0]]):
                n.llama_memory_clear(rt.memory, True)
                tokens = prefix + cf.suffix_tokens + answer[:-1]
                first = len(prefix) + len(cf.suffix_tokens) - 1
                total = 0.0
                for start in range(0, len(tokens), rt.n_batch):
                    part = tokens[start:start + rt.n_batch]
                    batch.n_tokens = len(part)
                    for i, token in enumerate(part):
                        batch.token[i], batch.pos[i] = token, start + i
                        batch.n_seq_id[i], batch.seq_id[i][0] = 1, 0
                        batch.logits[i] = start + i >= first
                    assert n.llama_decode(rt.ctx.ctx, batch) == 0
                    for i in range(len(part)):
                        position = start + i
                        if position < first:
                            continue
                        ptr = n.llama_get_logits_ith(rt.ctx.ctx, i)
                        assert ptr
                        logits = rt.np.ctypeslib.as_array(ptr, (rt.n_vocab,)).astype(rt.np.float64)
                        ordinary_logits = logits
                        peak = float(logits.max())
                        total += float(logits[answer[position - first]]) - peak - math.log(
                            float(rt.np.exp(logits - peak).sum()))
                totals.append(total)
            if not cf.collision:
                totals = [max(float(ordinary_logits[i]) for i in ids)
                          for ids in cf.candidate_ids]
            weights = [math.exp(x - max(totals)) for x in totals]
            probs = [x / sum(weights) for x in weights]
            if cf.choice_index is not None:
                results.setdefault(cf.field.name, {})[cf.field.choices[cf.choice_index]] = probs[0]
            else:
                results[cf.field.name] = dict(zip(cf.field.answers, probs))
        return results
    finally:
        n.llama_batch_free(batch)
        n.llama_memory_clear(rt.memory, True)


def main():
    args = acceptance.parser().parse_args()
    if args.output is None:
        return acceptance.main()
    trace_path = args.output.with_suffix(".trace.json")
    if trace_path.exists():
        raise SystemExit("trace output already exists")
    events = []
    original_load = LlamaCppRuntime.load
    original_prefill = LlamaCppRuntime.prefill
    original_pass = LlamaCppRuntime.batched_pass
    original_suite = acceptance.run_suite

    def load(rt):
        result = original_load(rt)
        rt.native = Boundary(rt, events)
        return result

    def prefill(rt, tokens):
        rt.native.request += 1
        rt.native.chunk = 0
        rt.native.phase = "prefill"
        rt.native.record("prompt", token_ids=tokens)
        return original_prefill(rt, tokens)

    def batched_pass(rt, rows, positions):
        rt.native.chunk += 1
        rt.native.phase = "suffix_or_collision"
        rt.native.record("branch_chunk", root_length=rt.prompt_length,
                         rows=rows, scored_offsets=[sorted(p) for p in positions])
        return original_pass(rt, rows, positions)

    def suite(decider, Schema, Calibrator, **kwargs):
        report = kwargs["report"]
        native = decider._llamacpp_rt.native.native
        dll = Path(native._lib._name)
        report["native_build"] = {
            "dll_path": str(dll), "dll_sha256": acceptance.sha256(dll),
            "system_info": native.llama_print_system_info().decode("utf-8"),
            "n_gpu_layers": 0, "offload_kqv": False, "op_offload": False,
            "use_extra_bufts": False,
            "n_threads": decider._llamacpp_rt.n_threads,
            "placement_source": "engine_llamacpp.load context/model params; CPU only",
        }
        try:
            original_suite(decider, Schema, Calibrator, **kwargs)
        finally:
            schema, compiled = acceptance.select_schema(Schema, decider.tokenizer)
            start = time.perf_counter()
            oracle = scalar_oracle(decider, schema, compiled)
            report["scalar_full_row_oracle"] = oracle
            report["scalar_oracle_ms"] = (time.perf_counter() - start) * 1000
            if report.get("runs"):
                baseline = report["runs"][0]["fields"]
                delta = max(abs(p - baseline[name]["distribution"][choice])
                            for name, dist in oracle.items() for choice, p in dist.items())
                report["scalar_oracle_max_delta"] = delta
                report["scalar_oracle_pass"] = delta <= kwargs["atol"]
        acceptance.require(report["scalar_oracle_pass"], "scalar full-row oracle tolerance failed")

    LlamaCppRuntime.load = load
    LlamaCppRuntime.prefill = prefill
    LlamaCppRuntime.batched_pass = batched_pass
    acceptance.run_suite = suite
    try:
        return acceptance.main()
    finally:
        with trace_path.open("x", encoding="utf-8") as f:
            json.dump({"synthetic_only": True, "events": events}, f, indent=2)
        print(f"Native boundary evidence (review required): {trace_path}")


if __name__ == "__main__":
    raise SystemExit(main())
