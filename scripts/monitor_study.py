#!/usr/bin/env python3
"""Read-only balanced study dashboard. No simulators or campaign files are modified."""
import argparse
import collections
import datetime as dt
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from simbench.config import load
from simbench.schedule import expand, pilots

DEFAULT = Path('/mnt/main/tklee/quantum-simulation-benchmark/results/balanced_simulation_study')
TERMINAL = {'success', 'error', 'timeout', 'resource_excluded', 'memory_limit', 'worker_crash'}

def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

def duration(seconds):
    if seconds is None:
        return 'unknown'
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes = seconds // 60
    return (f'{days}d ' if days else '') + f'{hours:02d}h {minutes:02d}m'

def med(values):
    return st.median(values) if values else None

def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)

class Monitor:
    def __init__(self, root):
        self.root = root
        self.stages = []
        self.cache = {}
        for stage in read(root / 'study.json')['stages']:
            config = load(REPO / stage['config'])
            jobs, _ = expand(config)
            self.stages.append((stage, config, jobs))

    def records(self, output, pilot=False):
        rows = []
        for path in (output / ('pilot' if pilot else 'cases')).glob('*/*__r*.json'):
            if path.name.endswith('.partial.json'):
                continue
            try:
                stamp = path.stat().st_mtime_ns
            except OSError:
                continue
            cached = self.cache.get(path)
            if cached is None or cached[0] != stamp:
                cached = (stamp, read(path))
                self.cache[path] = cached
            record = cached[1]
            if record.get('status') in TERMINAL:
                rows.append(record)
        return rows

    def snapshot(self):
        now = time.time()
        stages, observed, all_rows = [], [], []
        for stage, config, jobs in self.stages:
            output = Path(stage['output_dir'])
            expected = {j['id'] for j in jobs}
            rows = [r for r in self.records(output) if r.get('job', {}).get('id') in expected]
            done = {r['job']['id'] for r in rows}
            pending = [j for j in jobs if j['id'] not in done]
            pilot_jobs = pilots(config) if config['pilot']['enabled'] else []
            pilot_done = {r.get('job', {}).get('id') for r in self.records(output, True)}
            pilot_left = [j for j in pilot_jobs if j['id'] not in pilot_done]
            progress = read(output / 'progress.json')
            stages.append(dict(name=Path(stage['config']).stem.replace('balanced_', ''),
                config=config, jobs=jobs, rows=rows, pending=pending, pilot_left=pilot_left,
                progress=progress, output=output))
            for row in rows:
                all_rows.append((config, row))
                wall = row.get('supervisor_wall_s', row.get('worker_wall_s'))
                if finite(wall):
                    observed.append((row['job'], max(.1, wall)))

        predictions = {}
        def predict(job, timeout, optimistic):
            key = (job['config']['id'], job['spec']['n'], job['spec']['depth'],
                   job['spec']['family'], timeout, optimistic)
            if key not in predictions:
                predictions[key] = compute_prediction(job, timeout, optimistic)
            return predictions[key]

        def compute_prediction(job, timeout, optimistic):
            c, spec = job['config'], job['spec']
            exact = [v for j, v in observed if j['config']['id'] == c['id']
                     and j['spec']['n'] == spec['n'] and j['spec']['family'] == spec['family']
                     and j['spec']['depth'] == spec['depth']]
            if exact:
                return min(timeout, st.mean(exact) if not optimistic else min(exact))
            estimates = []
            for j, wall in observed:
                old, s = j['config'], j['spec']
                if old['name'] != c['name'] or old['gpu'] != c['gpu']:
                    continue
                if s['family'] != spec['family']:
                    continue
                factor = max(.25, spec['depth'] / s['depth'])
                # Explicit heuristics; these are scenario estimates, not fitted scaling laws.
                exponent = 1 if c['name'].startswith('trajectory') else 2
                factor *= 2 ** (exponent * max(0, spec['n'] - s['n']))
                for key, power in [('shots', 1), ('chi', 2), ('rank', 2), ('terms', 1)]:
                    if key in c and key in old:
                        factor *= max(.25, c[key] / old[key]) ** power
                estimates.append(min(timeout, wall * factor))
            if estimates:
                return min(timeout, min(estimates) if optimistic else med(estimates))
            return timeout * (.25 if optimistic else 1)

        total_eta = [0., 0., 0.]
        for stage in stages:
            config = stage['config']; timeout = config['timeout_s']
            fresh = [r for r in stage['rows'] if not r.get('reuse_provenance')]
            progress = stage['progress']
            launch_times = [r.get('at', 0) for r in
                            [read(stage['output'] / 'manifest.json').get('environment', {})]]
            launch_start = max(launch_times)
            recent = [r for r in fresh if r.get('recorded_at', 0) >= max(now - 1800, launch_start)]
            rate_span = min(1800, max(0, now - launch_start)) if launch_start else 0
            rate = len(recent) / rate_span if rate_span >= 60 else 0
            stage['rate_per_hour'] = rate * 3600
            estimates = []
            for optimistic in (True, False):
                lane_work = collections.Counter()
                for job in stage['pending']:
                    lane_work['gpu' if job['config']['gpu'] else 'cpu'] += predict(job, timeout, optimistic)
                cpu = config['execution']['cpu_workers']; gpu = len(config['execution']['gpu_ids'])
                lower = max(lane_work['cpu'] / max(1, cpu), lane_work['gpu'] / max(1, gpu))
                if not optimistic:
                    # Serial case dependencies, worker restarts, scoring I/O and phase barriers.
                    lower *= 1.5
                    if rate and stage['pending']:
                        lower = max(lower, len(stage['pending']) / rate)
                lower += len(stage['pilot_left']) * timeout / max(1, cpu + gpu)
                estimates.append(lower)
            unprepared = {j['case'] for j in stage['pending']
                          if not (stage['output'] / 'cases' / j['case'] / 'prepared.json').exists()}
            pilot_cases = {j['case'] for j in stage['pilot_left']
                           if not (stage['output'] / 'pilot' / j['case'] / 'prepared.json').exists()}
            remaining = len(stage['pending']) + len(stage['pilot_left'])
            envelope = remaining * (timeout + config['startup_timeout_s'])
            envelope += (len(unprepared) + len(pilot_cases)) * (
                config['preparation_timeout_s'] + config['startup_timeout_s'])
            stage['eta_s'] = [estimates[0], max(estimates), max(envelope, *estimates)]
            for i, seconds in enumerate(stage['eta_s']):
                total_eta[i] += seconds

        groups = collections.defaultdict(list)
        unique_rows = {}
        for config, row in all_rows:
            unique_rows.setdefault(row['job']['id'], (config, row))
        for config, row in unique_rows.values():
            groups[row['job']['config']['id']].append((config, row))
        methods = []
        for name, rows in groups.items():
            success, scored, passed, sim, fidelity, obs = 0, 0, 0, [], [], []
            for config, row in rows:
                if row['status'] != 'success':
                    continue
                success += 1
                metrics = row.get('metrics', {})
                if all(finite(metrics.get(key)) for key in config['accuracy']):
                    scored += 1
                    passed += all(metrics[key] <= limit for key, limit in config['accuracy'].items())
                value = row.get('timings', {}).get('simulation_s')
                if finite(value): sim.append(value)
                if finite(metrics.get('fidelity_score_abs_error')): fidelity.append(metrics['fidelity_score_abs_error'])
                if finite(metrics.get('observable_vector_rmse')): obs.append(metrics['observable_vector_rmse'])
            methods.append(dict(method=name, completed=len(rows), success=success, scored=scored,
                passed=passed, pass_fraction=passed / len(rows), median_simulation_s=med(sim),
                median_fidelity_error=med(fidelity), median_observable_error=med(obs)))
        return stages, methods, total_eta

    def render(self):
        stages, methods, eta = self.snapshot()
        now = dt.datetime.now(dt.timezone.utc)
        total = sum(len(s['jobs']) for s in stages); done = sum(len(s['rows']) for s in stages)
        counts = collections.Counter(r['status'] for s in stages for r in s['rows'])
        reused = sum(bool(r.get('reuse_provenance')) for s in stages for r in s['rows'])
        lines = [f'BALANCED STUDY | {now:%Y-%m-%d %H:%M:%S} UTC',
            f'Main/scaling evaluations: {done:,}/{total:,} ({done/total:.1%}) | remaining {total-done:,}',
            f'Imported records: {reused:,} (excluded from recent throughput) | statuses: {dict(counts)}',
            f'Pilot checks remaining: {sum(len(s["pilot_left"]) for s in stages)}', '',
            'Finish scenarios from now (continuous operation; estimates, not confidence intervals):']
        for label, seconds in zip(['Optimistic', 'Realistic', 'Maximum*'], eta):
            finish = now + dt.timedelta(seconds=seconds)
            lines.append(f'  {label:11} {duration(seconds):>13} | {finish:%Y-%m-%d %H:%M} UTC')
        lines.extend(['', 'STAGES                      DONE/TOTAL   LEFT  CPU/GPU LEFT  runs/h  optimistic / realistic',])
        for s in stages:
            cpu = sum(not j['config']['gpu'] for j in s['pending']); gpu = len(s['pending']) - cpu
            lines.append(f'{s["name"]:26} {len(s["rows"]):4}/{len(s["jobs"]):4} {len(s["pending"]):6} '
                         f'{cpu:5}/{gpu:<5} {s["rate_per_hour"]:6.1f}  {duration(s["eta_s"][0])} / {duration(s["eta_s"][1])}')
        established = [m for m in methods if m['scored'] >= 30 and not m['method'].startswith('dense_')]
        if established:
            best = max(established, key=lambda m: (m['pass_fraction'], -(m['median_simulation_s'] or float('inf'))))
            fastest = min(established, key=lambda m: m['median_simulation_s'] or float('inf'))
            lines.extend(['', 'PROVISIONAL APPROXIMATE LEADERS (>=30 scored unique cases/settings):',
                f'  Highest pass/all: {best["method"]} ({best["passed"]}/{best["completed"]}, {best["pass_fraction"]:.1%})',
                f'  Fastest successful median: {fastest["method"]} ({fastest["median_simulation_s"]:.2f}s; '
                f'{fastest["pass_fraction"]:.1%} pass/all)'])
        lines.extend(['', 'REMAINING METHOD SETTINGS (CPU/GPU, all stages; chi/rank/shots retained):'])
        remaining = collections.Counter(j['config']['id'] for s in stages for j in s['pending'])
        lines.extend(f'  {name:49} {count:5}' for name, count in sorted(remaining.items()))
        lines.extend(['', 'METHOD RESULTS | deduplicated across stages; sorted by passes / ALL attempts; small samples provisional',
                      'Method                                             done  good/scored  pass/all  median sim   fid err   obs err'])
        for m in sorted(methods, key=lambda m: (-m['pass_fraction'], m['median_simulation_s'] or float('inf'))):
            def number(value): return f'{value:.3g}' if value is not None else '-'
            lines.append(f'{m["method"]:49} {m["completed"]:5} {m["passed"]:4}/{m["scored"]:<5} '
                         f'{m["pass_fraction"]:7.1%} {number(m["median_simulation_s"]):>10}s '
                         f'{number(m["median_fidelity_error"]):>9} {number(m["median_observable_error"]):>9}')
        lines.extend(['', 'FAMILY LEADERS | approximate settings only, >=10 scored attempts; coverage differs'])
        family_groups = collections.defaultdict(list)
        seen = set()
        for s in stages:
            for r in s['rows']:
                if r['job']['id'] in seen: continue
                seen.add(r['job']['id'])
                c = r['job']['config']
                if c['name'] != 'dense': family_groups[(r['job']['spec']['family'], c['id'])].append((s['config'], r))
        leaders = collections.defaultdict(list)
        for (family, name), rows in family_groups.items():
            scored = [(c, r) for c, r in rows if r['status'] == 'success' and
                      all(finite(r.get('metrics', {}).get(k)) for k in c['accuracy'])]
            if len(scored) < 10: continue
            passed = sum(all(r['metrics'][k] <= v for k, v in c['accuracy'].items()) for c, r in scored)
            times = [r['timings']['simulation_s'] for _, r in scored if finite(r.get('timings', {}).get('simulation_s'))]
            leaders[family].append((passed / len(rows), med(times) or float('inf'), name, passed, len(rows)))
        for family, candidates in sorted(leaders.items()):
            best = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]
            lines.append(f'  {family:12} {best[2]:49} {best[3]}/{best[4]} joint passes, median {best[1]:.2f}s')
        lines.extend(['', 'ACTIVE WORKERS'])
        for s in stages:
            p = s['progress']; age = time.time() - p.get('updated_at', 0)
            if p.get('phase') not in ('campaign', 'pilot'): continue
            lines.append(f'  {s["name"]}: {p["phase"]}, progress age {age:.0f}s' + (' [STALE: check controller]' if age > 60 else ''))
            for w in p.get('workers', []):
                if 'method' not in w: continue
                wp = w.get('worker_phase', {})
                lines.append(f'    lane {w["index"]} GPU={w.get("gpu")} {w.get("case")} {w.get("method")} '
                             f'{w.get("elapsed_s", 0):.0f}s | {wp.get("phase", w.get("phase"))} '
                             f'{wp.get("completed", "?")}/{wp.get("total", "?")} | RSS {w.get("rss_bytes", 0)/1024**3:.2f} GiB')
        lines.extend(['', 'Interpretation:',
            '  Joint pass uses configured absolute fidelity-score and observable-error thresholds (usually 0.01 each).',
            '  These are score agreements, not full-state fidelity or proof of preserved circuit rankings.',
            '  Rankings pool unequal coverage and imported timings; compare matched cases before drawing conclusions.',
            '  Median times use successful simulations only; timeouts are censored and excluded from those medians.',
            '  Near-mixed noisy states can make absolute-error passes easy; a trivial baseline is still needed.',
            '  ETA: observed same-setting runtimes; otherwise depth/width/budget extrapolation capped at run timeout.',
            '  Width heuristic: 2^delta-n for trajectories, 4^delta-n otherwise; chi/rank squared, shots/terms linear.',
            '  Realistic adds 50% scheduling overhead and a recent 30-minute throughput floor for active stages.',
            '  *Maximum is serial timeout envelope: all remaining runs + per-run startup + missing preparation.',
            '  It assumes no pauses, stalls, retries or external failures; it is NOT a guaranteed wall-clock bound.'])
        return '\n'.join(lines), dict(updated_at=now.isoformat(), completed=done, total=total,
            remaining=total-done, statuses=dict(counts), eta_seconds=dict(zip(['optimistic', 'realistic', 'maximum_envelope'], eta)),
            methods=methods, stages=[dict(name=s['name'], completed=len(s['rows']), total=len(s['jobs']),
                remaining=len(s['pending']), pilot_remaining=len(s['pilot_left']), eta_seconds=s['eta_s']) for s in stages])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--watch', action='store_true', help='refresh until Ctrl-C')
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--json', action='store_true', help='print machine-readable snapshot')
    args = parser.parse_args()
    if args.interval < 5: parser.error('--interval must be at least 5 seconds')
    monitor = Monitor(args.root)
    try:
        while True:
            rendered, snapshot = monitor.render()
            if args.watch and sys.stdout.isatty(): print('\033[2J\033[H', end='')
            print(json.dumps(snapshot, indent=2) if args.json else rendered, flush=True)
            if not args.watch: break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
