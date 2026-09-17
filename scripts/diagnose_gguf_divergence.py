"""Opt-in, synthetic-only reproduction of the merged GGUF row-shape failure.

No arbitrary input contexts, downloads or production logging. Native model use
must be exclusive. Keeps the full acceptance prompt; focuses on tags[1].
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verify_gguf_native import CONTEXTS, schema_spec


def scalar_log_probability(logits, target):
    peak = float(max(logits))
    return float(logits[target]) - peak - math.log(math.fsum(
        math.exp(float(v) - peak) for v in logits))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-native", action="store_true")
    p.add_argument("--exclusive-model-use", action="store_true")
    p.add_argument("--variant", choices=("baseline", "fp32kv", "flash-off"), default="baseline")
    a = p.parse_args()
    if not (a.run_native and a.exclusive_model_use):
        p.error("requires --run-native --exclusive-model-use")
    if a.output.exists() or a.output.with_suffix(".npz").exists():
        p.error("output must be new")
    import numpy as np
    from llama_cpp import llama_cpp as n
    from parallel_decisions import Config, Decider, Schema
    from parallel_decisions.prompts import build_prompt
    from parallel_decisions.engine_llamacpp import _log_probability

    defaults = n.llama_context_default_params
    params = {}
    def context_params():
        cp = defaults()
        if a.variant == "fp32kv":
            cp.type_k = cp.type_v = 0  # GGML_TYPE_F32
        if a.variant == "flash-off":
            cp.flash_attn_type = 0
        params.update(type_k=cp.type_k, type_v=cp.type_v, flash_attn_type=cp.flash_attn_type)
        return cp
    n.llama_context_default_params = context_params
    d = Decider(str(a.model), backend="llamacpp", max_fields_per_batch=8,
                max_collision_rows=8, warmup=False, config=Config(log="off"))
    d.load()
    rt = d._llamacpp_rt
    s = Schema(schema_spec(["parcel red", "parcel blue", "other"]))
    compiled = s.compile(d.tokenizer)
    prompt = d.tokenizer.encode(build_prompt(CONTEXTS["long"], s))
    assert len(prompt) == 383
    target = compiled[3]
    assert target.row_name == "tags[1]"
    assert target.suffix_tokens == [220, 330, 14082, 58, 16, 60, 788, 220]
    assert target.candidate_ids == [[1866], [3849]]
    for cf in compiled:
        assert d.tokenizer.encode(build_prompt(CONTEXTS["long"], s) + cf.suffix) == prompt + cf.suffix_tokens

    report = dict(synthetic_only=True, variant=a.variant, params=params,
                  system_info=n.llama_print_system_info().decode(), prompt_tokens=prompt,
                  compiled=[dict(name=cf.row_name, suffix_tokens=cf.suffix_tokens,
                                 candidate_ids=cf.candidate_ids, sequences=cf.sequences)
                            for cf in compiled], runs=[], trace=[])
    arrays = {}
    label = ""
    def state(seq=0):
        size = n.llama_state_seq_get_size(rt.ctx.ctx, seq)
        buf = (ctypes.c_uint8 * size)()
        written = n.llama_state_seq_get_data(rt.ctx.ctx, buf, size, seq)
        assert written == size
        return dict(bytes=size, sha256=hashlib.sha256(bytes(buf)).hexdigest(),
                    min=n.llama_memory_seq_pos_min(rt.memory, seq),
                    max=n.llama_memory_seq_pos_max(rt.memory, seq))

    original_decode = rt._decode
    def decode(entries):
        before = state()
        out = original_decode(entries)
        n.llama_synchronize(rt.ctx.ctx)
        after = state()
        if entries[0][2] != 0:
            assert before == after, "root KV bytes/position changed during suffix decode"
        packed = np.ctypeslib.as_array(n.llama_get_logits(rt.ctx.ctx),
                                      shape=(len(out) * rt.n_vocab,)).reshape(len(out), rt.n_vocab) if out else []
        refs = {}
        for compact, (index, logits) in enumerate(out.items()):
            assert np.array_equal(logits, packed[compact]), "sparse/packed output mapping differs"
            key = f"{label}-decode{len(report['trace'])}-i{index}"
            arrays[key] = logits
            refs[index] = key
        report["trace"].append(dict(run=label, entries=entries, outputs=refs,
                                    root_before=before, root_after=after))
        return out
    rt._decode = decode
    original_clear = n.llama_memory_clear
    def clear(mem, data):
        assert data is True
        original_clear(mem, data)
        report["trace"].append(dict(run=label, clear_buffers=data, empty_root=state()))
        assert state()["max"] == -1
    n.llama_memory_clear = clear
    original_copy, original_rm = n.llama_memory_seq_cp, n.llama_memory_seq_rm
    def copy(mem, src, dst, start, end):
        original_copy(mem, src, dst, start, end)
        report["trace"].append(dict(run=label, copy=[src, dst, start, end], branch=state(dst)))
        assert state(dst)["max"] == len(prompt) - 1
    def remove(mem, seq, start, end):
        result = original_rm(mem, seq, start, end)
        report["trace"].append(dict(run=label, remove=[seq, start, end], result=result))
        assert n.llama_memory_seq_pos_max(mem, seq) == -1
        return result
    n.llama_memory_seq_cp, n.llama_memory_seq_rm = copy, remove
    original_pass = rt.batched_pass
    targets = {}
    def branch_pass(rows, positions):
        out = original_pass(rows, positions)
        for tokens, row in zip(rows, out):
            if tokens == target.suffix_tokens:
                logits = row[len(tokens)-1]
                targets[label] = logits.copy()
                lp = [scalar_log_probability(logits, i) for i in [1866, 3849]]
                for i, expected in zip([1866, 3849], lp):
                    assert abs(_log_probability(rt, logits, i) - expected) < 1e-12
        return out
    rt.batched_pass = branch_pass
    try:
        for limit in (8, 1, 2):
            label = f"limit-{limit}"
            d.max_fields_per_batch = d.max_collision_rows = limit
            result = d.decide(CONTEXTS["long"], s)
            z = targets[label]
            report["runs"].append(dict(label=label, probability=result["tags"].distribution["light"],
                logits_dtype=str(z.dtype), logits_shape=list(z.shape),
                candidate_logits=[float(z[i]) for i in [1866,3849]],
                candidate_full_vocab_logp=[scalar_log_probability(z,i) for i in [1866,3849]],
                telemetry=result.telemetry))
        report["max_probability_delta"] = max(abs(r["probability"]-report["runs"][0]["probability"]) for r in report["runs"])
        report["max_raw_logit_delta"] = float(max(np.max(np.abs(v-targets["limit-8"])) for v in targets.values()))
        report["strict_pass"] = report["max_probability_delta"] <= 0.0001
    finally:
        n.llama_memory_clear = original_clear
        n.llama_memory_seq_cp, n.llama_memory_seq_rm = original_copy, original_rm
        n.llama_context_default_params = defaults
        rt.close()
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    np.savez(a.output.with_suffix(".npz"), **arrays)
    print(json.dumps({k:v for k,v in report.items() if k not in ("trace", "prompt_tokens", "compiled")}, indent=2))
    return 0 if report["strict_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
