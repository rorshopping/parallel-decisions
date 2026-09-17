"""Opt-in synthetic tag-logit localization; no modifications to acceptance checks."""
import json
import sys
from pathlib import Path

import verify_gguf_native as acceptance


def main():
    p = acceptance.parser()
    args = p.parse_args()
    if not args.run_native or not args.exclusive_model_use or not args.model or not args.output:
        p.error("requires --run-native --exclusive-model-use --model --output")
    if args.output.exists():
        p.error("output must be new")
    sys.path.insert(0, str(acceptance.ROOT / "src"))
    import numpy as np
    from parallel_decisions import Config, Decider, Schema
    from parallel_decisions.prompts import build_prompt
    d = Decider(str(args.model), backend="llamacpp", max_fields_per_batch=8,
                max_collision_rows=8, warmup=False, config=Config(log="off"))
    d.load()
    rt = d._llamacpp_rt
    native = rt.native
    schema, compiled = acceptance.select_schema(Schema, rt.tokenizer)
    tags = [cf for cf in compiled if cf.choice_index is not None]
    prefix = rt.tokenizer.encode(build_prompt(acceptance.CONTEXTS['a'], schema))
    captured = {}
    active = {}
    original_pass, original_decode = rt.batched_pass, rt._decode

    def branch_pass(rows, positions):
        active.clear()
        for i, row in enumerate(rows):
            for cf in tags:
                if row == cf.suffix_tokens:
                    active[i + 1] = cf.row_name
        return original_pass(rows, positions)

    def decode(entries):
        out = original_decode(entries)
        for i, logits in out.items():
            seq = entries[i][2]
            if seq in active:
                captured[active[seq]] = logits.copy()
        return out

    rt.batched_pass, rt._decode = branch_pass, decode
    result = d.decide(acceptance.CONTEXTS['a'], schema)
    rt.batched_pass, rt._decode = original_pass, original_decode
    report = {"context": "a", "source_sha256": acceptance.sha256(acceptance.ROOT / 'src/parallel_decisions/engine_llamacpp.py'),
              "public_tags": result['tags'].distribution, "rows": []}
    batch = native.llama_batch_init(rt.n_batch, 0, 1)

    def seq0(tokens, start, output):
        batch.n_tokens = len(tokens)
        assert len(tokens) <= rt.n_batch
        for i, token in enumerate(tokens):
            batch.token[i] = token
            batch.pos[i] = start + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = output and i == len(tokens) - 1
        assert native.llama_decode(rt.ctx.ctx, batch) == 0
        native.llama_synchronize(rt.ctx.ctx)
        if output:
            positive = np.ctypeslib.as_array(native.llama_get_logits_ith(rt.ctx.ctx, len(tokens)-1), (rt.n_vocab,)).copy()
            negative = np.ctypeslib.as_array(native.llama_get_logits_ith(rt.ctx.ctx, -1), (rt.n_vocab,)).copy()
            assert np.array_equal(positive, negative), "sparse output mapping differs"
            return positive

    try:
        for cf in tags:
            full = rt.tokenizer.encode(build_prompt(acceptance.CONTEXTS['a'], schema) + cf.suffix)
            assert full == prefix + cf.suffix_tokens
            native.llama_memory_clear(rt.memory, True)
            seq0(prefix, 0, False)
            split = seq0(cf.suffix_tokens, len(prefix), True)
            native.llama_memory_clear(rt.memory, True)
            fused = seq0(full, 0, True)
            def selected(logits):
                scores = [max(float(logits[i]) for i in ids) for ids in cf.candidate_ids]
                w = np.exp(np.array(scores) - max(scores))
                return {'raw': scores, 'softmax': (w / w.sum()).tolist()}
            report['rows'].append({
                'row': cf.row_name, 'suffix': cf.suffix, 'suffix_tokens': cf.suffix_tokens,
                'candidate_ids': cf.candidate_ids,
                'candidate_pieces': [[rt.model.detokenize([i]).decode('utf-8', errors='replace') for i in ids] for ids in cf.candidate_ids],
                'boundary_exact': full == prefix + cf.suffix_tokens,
                'branch': selected(captured[cf.row_name]), 'scalar_split': selected(split),
                'scalar_fused': selected(fused),
                'whole_vocab_branch_split_max': float(np.max(np.abs(captured[cf.row_name] - split))),
                'whole_vocab_split_fused_max': float(np.max(np.abs(split - fused))),
                'positive_negative_output_index_equal': True,
            })
    finally:
        native.llama_batch_free(batch)
        rt.close()
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
