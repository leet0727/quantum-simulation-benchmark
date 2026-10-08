#!/usr/bin/env python3
"""Run fixed balanced-study stages sequentially; restart resumes each stage."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/mnt/main/tklee/quantum-simulation-benchmark/results/balanced_simulation_study')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='study directory containing study.json')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    study = json.loads((root / 'study.json').read_text())
    lock = (root / 'study.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / 'launcher_pid.json').write_text(json.dumps({
        'launcher_pid': os.getpid(),
        'stop_command': f'kill -TERM {os.getpid()}',
    }, indent=2) + '\n')
    stopping = False
    child = None
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for stage in study['stages']:
        if stopping:
            return 130
        output = Path(stage['output_dir'])
        progress = output / 'progress.json'
        if progress.exists() and json.loads(progress.read_text())['phase'] == 'finished':
            print('Already finished:', stage['config'], flush=True)
            continue
        command = [sys.executable, str(REPO / 'run_sim.py'), '--config', str(REPO / stage['config']), '--yes']
        if (output / 'manifest.json').exists():
            command.append('--resume')
        print('Starting:', stage['config'], flush=True)
        with (output / 'campaign.log').open('a') as log:
            child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            code = child.wait()
        child = None
        if stopping or code:
            print('Study stopped; stage exit code:', code, flush=True)
            return code or 130
    print('All six study stages finished.', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
