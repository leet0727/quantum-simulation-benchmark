"""Single-controller CLI with resumable, case-safe CPU/GPU scheduling."""
import argparse,collections,fcntl,hashlib,json,os,platform,shutil,signal,sys,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .config import load,fingerprint
from .schedule import expand,pilots

def source_hash():
    digest=hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):digest.update(p.name.encode());digest.update(p.read_bytes())
    return digest.hexdigest()

def result_path(out,j,pilot=False):
    return out/('pilot' if pilot else 'cases')/j['case']/(j['config']['id']+f"__r{j['repeat']}.json")

def execute(config,jobs,out,resume):
    from .runtime import Lane,TERMINAL
    from .common import atomic_json
    from .report import write as report
    from rich.progress import Progress,BarColumn,TextColumn,TimeElapsedColumn,TimeRemainingColumn
    import importlib.metadata
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('another controller owns this output directory')
    manifest=out/'manifest.json';digest=source_hash();fp=fingerprint(config)
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if not resume:raise ValueError('output already exists; use --resume or another output_dir')
        if old['fingerprint']!=fp or old['source_hash']!=digest:raise ValueError('resume refused: scientific configuration or source changed; use a new output_dir')
    elif resume:raise ValueError('--resume requires an existing manifest')
    packages={}
    for name in ['numpy','scipy','qiskit','qiskit-aer','qiskit-aer-gpu','qiskit-ibm-runtime','cupy-cuda12x','cuquantum-python-cu12','opt_einsum']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    environment=dict(python=sys.version,platform=platform.platform(),packages=packages,execution=config['execution'],at=time.time())
    if not manifest.exists():atomic_json(manifest,dict(config=config,fingerprint=fp,source_hash=digest,environment=environment,jobs=len(jobs)))
    with (out/'launches.jsonl').open('a') as f:f.write(json.dumps(environment)+'\n')
    stop=threading.Event();previous={sig:signal.getsignal(sig) for sig in [signal.SIGINT,signal.SIGTERM]}
    for sig in previous:signal.signal(sig,lambda *_:stop.set())
    lanes=[Lane(i,g,out,config,stop) for i,g in enumerate(config['execution']['gpu_ids']+[None]*config['execution']['cpu_workers'])]
    has_cpu=any(l.gpu is None for l in lanes);started=time.monotonic();fatal=None
    def eligible(l,j):return l.gpu is not None if j['config']['gpu'] else (l.gpu is None or not has_cpu)
    def evict(active):
        paths=sorted((out/'cases').glob('*/reference.npy'),key=lambda p:p.stat().st_mtime)
        total=sum(p.stat().st_size for p in paths);limit=config['resources']['reference_cache_gib']*1024**3
        for p in paths:
            if total<=limit:break
            if p.parent.name in active:continue
            total-=p.stat().st_size;p.unlink()
    class SafePool(ThreadPoolExecutor):
        def __exit__(self,kind,value,tb):
            if kind is not None:
                stop.set()
                for lane in lanes:lane.stop()
            return super().__exit__(kind,value,tb)
    def run_batch(batch,is_pilot):
        counts=collections.Counter();pending=[]
        for j in batch:
            p=result_path(out,j,is_pilot)
            status=json.loads(p.read_text()).get('status') if p.exists() else None
            if status in TERMINAL:counts[status]+=1
            else:pending.append(j)
        if is_pilot and any(n for status,n in counts.items() if status!='success'):raise RuntimeError('existing pilot contains failed runs; fix the problem and use a fresh output directory')
        completed=sum(counts.values());last_print=0.;last_report=completed
        progress=Progress(TextColumn('{task.description}'),BarColumn(),TextColumn('{task.completed}/{task.total}'),TimeElapsedColumn(),TimeRemainingColumn(),disable=not sys.stdout.isatty())
        with progress,SafePool(max_workers=len(lanes)) as pool:
            task=progress.add_task('Pilot' if is_pilot else 'Campaign',total=len(batch),completed=completed);running={}
            while (pending or running) and not stop.is_set():
                active={j['case'] for _,j in running.values()};occupied={l.index for l,_ in running.values()}
                # Only advance to the next scheduling phase when the prior one finishes.
                phase=next(iter(running.values()))[1]['phase'] if running else pending[0]['phase']
                for lane in lanes:
                    if lane.index in occupied:continue
                    chosen=None
                    for i,j in enumerate(pending):
                        if j['phase']!=phase:break
                        if j['case'] not in active and eligible(lane,j):
                            # Keep a case's jobs in schedule order even across different device lanes.
                            if any(x['case']==j['case'] for x in pending[:i]):continue
                            chosen=i;break
                    if chosen is not None:
                        if shutil.disk_usage(out).free<config['resources']['min_free_disk_gib']*1024**3:raise RuntimeError('minimum free disk threshold reached')
                        j=pending.pop(chosen);active.add(j['case']);running[pool.submit(lane.job,j,is_pilot)]=(lane,j)
                if pending and not running:raise RuntimeError('scheduler has no eligible worker for pending jobs')
                for future in list(running):
                    if not future.done():continue
                    lane,j=running.pop(future)
                    try:r=future.result()
                    except InterruptedError:
                        stop.set();break
                    counts[r['status']]+=1;completed+=1;progress.update(task,completed=completed)
                    if is_pilot and r['status']!='success':raise RuntimeError(f"pilot failed ({r['status']}): {j['id']}; inspect its result and worker log")
                now=time.monotonic()
                if now-last_print>=1:
                    status=dict(pid=os.getpid(),updated_at=time.time(),phase='pilot' if is_pilot else 'campaign',completed=completed,
                        total=len(batch),counts=dict(counts),elapsed_s=now-started,workers=[dict(index=l.index,gpu=l.gpu,**l.current) for l in lanes])
                    atomic_json(out/'progress.json',status)
                    if not sys.stdout.isatty() and (now-last_print>=10 or last_print==0):
                        width=24;filled=int(width*completed/max(1,len(batch)))
                        print(f"{status['phase']} [{'#'*filled}{'.'*(width-filled)}] {completed}/{len(batch)} {dict(counts)} elapsed {now-started:.1f}s",flush=True)
                        last_print=now
                    elif sys.stdout.isatty():
                        progress.update(task,description=f"{status['phase']} ok={counts['success']} timeout={counts['timeout']} active={len(running)}");last_print=now
                if not is_pilot and completed-last_report>=20:report(out,config);evict({j['case'] for _,j in running.values()});last_report=completed
                time.sleep(.1)
            if stop.is_set():
                for lane in lanes:lane.stop()
        return counts
    try:
        if config['pilot']['enabled']:
            run_batch(pilots(config),True)
            if not stop.is_set():atomic_json(out/'pilot_passed.json',dict(at=time.time(),source_hash=digest))
        if not stop.is_set():run_batch(jobs,False)
    except BaseException as e:
        fatal=e;stop.set()
    finally:
        stop.set()
        for lane in lanes:lane.stop()
        report(out,config);evict(set())
        for sig,handler in previous.items():signal.signal(sig,handler)
        # Determine completion from durable records, including terminal failures.
        done=sum(result_path(out,j).exists() and json.loads(result_path(out,j).read_text()).get('status') in TERMINAL for j in jobs)
        atomic_json(out/'progress.json',dict(pid=os.getpid(),updated_at=time.time(),phase='finished' if done==len(jobs) else 'stopped',
            completed=done,total=len(jobs),elapsed_s=time.monotonic()-started,error=str(fatal) if fatal else None))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    if fatal:raise fatal
    print(f"Saved {done}/{len(jobs)} terminal records to {out}. See results.csv and summary.json.")
    return 0 if done==len(jobs) else 130

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='input.json',type=Path)
    p.add_argument('--plan',action='store_true',help='validate and preview without initializing simulators')
    p.add_argument('--yes',action='store_true',help='explicitly skip interactive confirmation')
    p.add_argument('--resume',action='store_true');p.add_argument('--status',type=Path,help='read progress from an output directory')
    args=p.parse_args()
    if args.status:
        record=json.loads((args.status/'progress.json').read_text());print(json.dumps(record,indent=2));return 0
    try:
        config=load(args.config);jobs,methods=expand(config)
        out=Path(config['output_dir']).expanduser().resolve()
        print('Expanded experiment settings:\n'+json.dumps(config,indent=2))
        print(f"\n{len({j['case'] for j in jobs})} circuits × {len(methods)} method settings; {len(jobs)} scheduled runs.")
        print(f"Pilot: {len(pilots(config)) if config['pilot']['enabled'] else 0} runs. Output: {out}")
        print('First 12 runs:\n'+'\n'.join(j['id'] for j in jobs[:12]))
        print('Dense output alone at largest size: %.3f GiB' % (16*4**max(config['circuits']['qubits'])/1024**3))
        if args.plan:return 0
        if not args.yes:
            try:answer=input('Launch this experiment? [Y/N] ').strip().lower()
            except EOFError:answer='n'
            if answer not in ['y','yes']:print('Cancelled.');return 0
        return execute(config,jobs,out,args.resume)
    except (ValueError,RuntimeError,FileNotFoundError,ImportError) as e:
        print(f'Error: {e}',file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
