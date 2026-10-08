#!/usr/bin/env python3
"""Inspect a campaign result by completion timestamp order (one-based)."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('completion', type=int, help='one-based completion number')
parser.add_argument('--config', type=Path, default=Path('configs/two_gpu_64gib.json'))
parser.add_argument('--output', type=Path, help='override results directory')
args = parser.parse_args()
if args.completion < 1:
    parser.error('completion must be positive')
out = args.output or Path(json.loads(args.config.read_text())['output_dir'])
terminal = {'success', 'error', 'timeout', 'resource_excluded', 'memory_limit', 'worker_crash'}
rows = []
for path in (out / 'cases').glob('*/*__r*.json'):
    result = json.loads(path.read_text())
    if result.get('status') in terminal:
        rows.append((result.get('recorded_at', 0), str(path), result))
rows.sort(key=lambda row: row[:2])
if args.completion > len(rows):
    parser.error(f'only {len(rows)} terminal results available')
_, path, result = rows[args.completion - 1]
print(json.dumps(dict(completion_by_timestamp=args.completion, file=path,
    status=result.get('status'), job=result.get('job'), metrics=result.get('metrics'),
    timings=result.get('timings'), error=result.get('error')),
    indent=2))
