"""Paired full/prefix CUDA timings (model download required; not a pytest test).

python scripts/measure_prefix_speedup.py --output prefix-results.json
Reports preparation separately and checks cache integrity and answer agreement.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from parallel_decisions import Decider, Schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--cuda-graph', action='store_true', default=None)
    args = parser.parse_args()
    if args.runs < 2:
        parser.error('--runs must be at least 2')
    import torch
    import transformers

    schema = Schema({f'check_{i}': {
        'type': 'boolean',
        'description': f'Check {i}: Is a human response required for the described support incident? '
                       'Use only explicit evidence in the ticket. Requests about billing, outages, '
                       'account access or unresolved errors require review; informational notices do not.',
    } for i in range(8)})
    contexts = [
        'Ticket: customer cannot log in, reset email not arriving.',
        'Invoice disputed: double charge of 49 EUR.',
        'Weekly product newsletter. No action needed.',
        'Export job stuck at 80 percent for two hours.',
    ]
    d = Decider(args.model, backend='torch', torch_device='cuda',
                torch_dtype='float16', max_fields_per_batch=8, cuda_graph=args.cuda_graph)
    d.load()
    rt = d._torch_rt
    rt.synchronize()
    t = time.perf_counter()
    prefix = d.prepare(schema)
    rt.synchronize()
    preparation_ms = (time.perf_counter() - t) * 1000
    assert prefix.reusable
    original_length = prefix.cache.get_seq_length()
    d.decide(contexts[0], schema)
    d.decide_with_prefix(prefix, contexts[0])
    if rt.cuda_graph:
        # Prime each shape outside measured pairs (including varying context lengths).
        for context in contexts:
            d.decide(context, schema)
            d.decide_with_prefix(prefix, context)
        if rt.graph_cache.disabled or not rt.graph_cache.entries:
            raise RuntimeError('CUDA graphs requested but capture failed')
    capture_ms = rt.graph_cache.capture_ms
    rows = []
    try:
        for i in range(args.runs):
            context = contexts[i % len(contexts)]
            measured = {}
            for mode in (['full', 'prefix'] if i % 2 == 0 else ['prefix', 'full']):
                rt.synchronize()
                started = time.perf_counter()
                result = (d.decide(context, schema) if mode == 'full' else
                          d.decide_with_prefix(prefix, context))
                rt.synchronize()
                measured[mode] = {
                    'wall_ms': (time.perf_counter() - started) * 1000,
                    'prefill_ms': result.prefill_ms, 'pass_ms': result.pass_ms,
                    'values': result.json(),
                    'distributions': {k: v.distribution for k, v in result.items()},
                    'telemetry': result.telemetry,
                }
            assert prefix.cache.get_seq_length() == original_length
            measured['same_answers'] = measured['full']['values'] == measured['prefix']['values']
            rows.append(measured)
    finally:
        prefix.release()
    means = {mode: {key: statistics.mean(r[mode][key] for r in rows)
                    for key in ('wall_ms', 'prefill_ms', 'pass_ms')}
             for mode in ('full', 'prefix')}
    output = {
        'model': args.model, 'torch': torch.__version__, 'transformers': transformers.__version__,
        'platform': platform.platform(), 'gpu': torch.cuda.get_device_name(),
        'dtype': str(rt.dtype), 'runs': args.runs, 'fields': len(schema),
        'preparation_ms': preparation_ms, 'prefix_tokens': original_length,
        'cuda_graph': rt.cuda_graph, 'capture_ms': capture_ms,
        'graph_disabled': rt.graph_cache.disabled, 'graph_replays': rt.graph_cache.replays,
        'summary_ms': {mode: {key: {
            'p50': statistics.median(r[mode][key] for r in rows),
            'p95': sorted(r[mode][key] for r in rows)[
                min(len(rows) - 1, int(.95 * len(rows)))],
        } for key in ('wall_ms', 'prefill_ms', 'pass_ms')} for mode in ('full', 'prefix')}, 
        'all_answers_equal': all(r['same_answers'] for r in rows),
        'means': means,
        'wall_speedup': means['full']['wall_ms'] / means['prefix']['wall_ms'],
        'prefill_speedup': means['full']['prefill_ms'] / means['prefix']['prefill_ms'],
        'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in output.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
