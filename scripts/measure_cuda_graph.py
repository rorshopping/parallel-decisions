"""Paired synchronized CUDA eager/graph benchmark; cached model required.

python scripts/measure_cuda_graph.py --output cuda-graph-results.json
Use --schema schema.json --context packet.txt for an external probe workload.
Default workloads are explicit synthetic 8/27-row support schemas, NOT the
unavailable historical 28-field probe. Capture calls are excluded from pairs.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

from parallel_decisions import Config, Decider, Schema
from parallel_decisions.engine_torch import TorchCudaGraphCache


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summary(rows, modes=('eager', 'graph')):
    return {mode: {phase: {
        'mean': statistics.mean(r[mode][phase] for r in rows),
        'p50': percentile([r[mode][phase] for r in rows], .50),
        'p95': percentile([r[mode][phase] for r in rows], .95),
    } for phase in ('prefill_ms', 'suffix_ms', 'total_ms')} for mode in modes}


def synthetic_schema(n):
    return Schema({f'check_{i}': {
        'type': 'boolean',
        'description': f'Check {i}: Is a human response required for this support incident? '
                       'Billing, outages, account access and unresolved errors require review.',
    } for i in range(n)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--schema', type=Path)
    parser.add_argument('--context', type=Path)
    args = parser.parse_args()
    if args.runs < 2:
        parser.error('--runs must be at least 2')
    import torch
    import transformers

    context = (args.context.read_text(encoding='utf-8') if args.context else
               'Ticket: customer cannot log in, reset email not arriving.')
    workloads = ([(args.schema.stem, Schema(json.loads(args.schema.read_text(encoding='utf-8'))))]
                 if args.schema else [('synthetic_8', synthetic_schema(8)),
                                      ('synthetic_27', synthetic_schema(27))])
    d = Decider(args.model, backend='torch', torch_device='cuda', torch_dtype='float16',
                max_fields_per_batch=32, config=Config(), cuda_graph=False)
    d.load()
    rt = d._torch_rt
    output = {'model': args.model, 'torch': torch.__version__,
              'transformers': transformers.__version__, 'platform': platform.platform(),
              'gpu': torch.cuda.get_device_name(), 'dtype': str(rt.dtype),
              'runs': args.runs, 'workloads': {}}
    for name, schema in workloads:
        rt.graph_cache = TorchCudaGraphCache(rt)
        rt.cuda_graph = False
        d.decide(context, schema)
        rt.cuda_graph = True
        # First sighting, then capture, then replay warmup; none are timed pairs.
        for _ in range(3):
            d.decide(context, schema)
        if rt.graph_cache.disabled or not rt.graph_cache.entries:
            output['workloads'][name] = {
                'status': 'graph_unavailable', 'fields': len(schema),
                'decision_rows': len(d._compile(schema)),
                'capture_ms': rt.graph_cache.capture_ms,
                'free_bytes': torch.cuda.mem_get_info()[0],
                'reason': (rt.graph_cache.disable_reason
                           or ('no shapes captured (each shape must be seen at '
                               'least twice before capture; raise --runs if the '
                               'workload only ever calls each shape once)')),
            }
            print(json.dumps(output['workloads'][name], indent=2))
            continue
        capture_ms = rt.graph_cache.capture_ms
        graph_shapes = list(rt.graph_cache.entries)
        rt.synchronize()
        persistent_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for i in range(args.runs):
            measured = {}
            for mode in (['eager', 'graph'] if i % 2 == 0 else ['graph', 'eager']):
                rt.cuda_graph = mode == 'graph'
                rt.synchronize()
                started = time.perf_counter()
                result = d.decide(context, schema)
                rt.synchronize()
                measured[mode] = {
                    'total_ms': (time.perf_counter() - started) * 1000,
                    'prefill_ms': result.prefill_ms, 'suffix_ms': result.pass_ms,
                    'values': result.json(),
                    'distributions': {k: v.distribution for k, v in result.items()},
                    'telemetry': result.telemetry,
                }
            measured['same_answers'] = measured['eager']['values'] == measured['graph']['values']
            measured['max_probability_error'] = max(
                abs(p - measured['graph']['distributions'][k][answer])
                for k, dist in measured['eager']['distributions'].items()
                for answer, p in dist.items())
            rows.append(measured)
        assert not rt.graph_cache.disabled
        assert capture_ms == rt.graph_cache.capture_ms
        stats = summary(rows)
        output['workloads'][name] = {
            'fields': len(schema), 'decision_rows': len(d._compile(schema)),
            'context': context, 'schema_source': str(args.schema) if args.schema else 'synthetic',
            'capture_ms': capture_ms, 'graph_shapes': graph_shapes,
            'replays': rt.graph_cache.replays,
            'persistent_allocated_mib': persistent_allocated / 1024**2,
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
            'all_answers_equal': all(r['same_answers'] for r in rows),
            'max_probability_error': max(r['max_probability_error'] for r in rows),
            'summary_ms': stats,
            'total_p50_speedup': stats['eager']['total_ms']['p50'] / stats['graph']['total_ms']['p50'],
            'rows': rows,
        }
        print(json.dumps({k: v for k, v in output['workloads'][name].items() if k != 'rows'}, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
