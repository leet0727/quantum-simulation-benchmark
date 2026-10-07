"""A persistent worker; the parent owns deadlines, resource limits and signals."""
import argparse,hashlib,json,os,pickle,sys,time,traceback
from pathlib import Path
from . import common
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--ipc',type=Path,required=True);p.add_argument('--gpu',action='store_true');a=p.parse_args()
    a.ipc.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
    from . import engine
    from .prepare import prepare
    import qiskit_aer
    if a.gpu:
        import cupy as cp
        cp.linalg.svd(cp.eye(4));cp.linalg.eigh(cp.eye(4));engine.sync()
    common.atomic_json(a.ipc/'ready.json',dict(pid=os.getpid(),startup_s=time.perf_counter()-begin,
        gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),aer_devices=list(qiskit_aer.AerSimulator().available_devices())))
    cached=None;job=None
    for line in sys.stdin:
        req=json.loads(line);result_path=Path(req['result']);folder=Path(req['folder']);folder.mkdir(parents=True,exist_ok=True)
        config=req['settings'];start=time.perf_counter();cpu_start=time.process_time();last=[0.]
        def progress(phase,completed,total,extra):
            now=time.monotonic()
            if now-last[0]<.5 and completed not in [0,total]:return
            last[0]=now
            common.atomic_json(a.ipc/'phase.json',dict(request_id=req['request_id'],phase=phase,completed=completed,
                total=total,elapsed_s=time.perf_counter()-start,at=time.time(),**extra))
        try:
            if req['operation']=='prepare':
                dimension_bytes=16*4**req['spec']['n']
                if dimension_bytes>config['resources']['max_output_gib']*1024**3:
                    raise MemoryError(f'dense output needs {dimension_bytes/1024**3:g} GiB, above configured max_output_gib')
                result=prepare(req['spec'],folder,progress,config)
            elif req['operation']=='sleep_test':
                time.sleep(req['sleep_s']);result=dict(status='success')
            else:
                tick=time.perf_counter()
                if cached!=folder:
                    with (folder/'input.pkl').open('rb') as f:job=pickle.load(f)
                    cached=folder
                load_s=time.perf_counter()-tick;cfg=dict(req['job']['config'])
                cfg.update(threads=config['execution']['threads_per_worker'],
                    workspace_bytes=int(config['resources']['workspace_gib']*1024**3),
                    tn_intermediate_bytes=int(config['resources']['tn_intermediate_gib']*1024**3))
                progress('simulation',0,1,{})
                tick=time.perf_counter();rho,diagnostics,stages=engine.evaluate(job['variant'],cfg,progress)
                simulation_s=time.perf_counter()-tick
                if not np.all(np.isfinite(rho)):raise ValueError('non-finite output')
                if a.gpu:diagnostics['cupy_pool_reserved_after_simulation_bytes']=cp.get_default_memory_pool().total_bytes()
                partial=dict(status='incomplete',config=cfg,case=folder.name,spec=job['spec'],diagnostics=diagnostics,
                    timings=dict(load_s=load_s,simulation_s=simulation_s,**stages))
                common.atomic_json(result_path.with_suffix('.partial.json'),partial)
                progress('scoring',0,1,{})
                tick=time.perf_counter();scoring_gpu=a.gpu and config['execution']['scoring_device']!='cpu'
                metrics=engine.scores(rho,job['psi'],job['words'],eigensolver_gpu=scoring_gpu)
                metrics['observable_rmse']=float(np.sqrt(np.mean((np.array(metrics['observables'])-job['ideal_obs'])**2)))
                scoring_s=time.perf_counter()-tick;tick=time.perf_counter()
                if a.gpu:diagnostics['cupy_pool_reserved_after_scoring_bytes']=cp.get_default_memory_pool().total_bytes()
                ref_path=folder/'reference.npy';ref_json=folder/'reference.json';reference=None
                if ref_json.exists():
                    reference=json.loads(ref_json.read_text());rm=reference['metrics']
                    metrics['fidelity_score_abs_error']=abs(metrics['state_fidelity']-rm['state_fidelity'])
                    metrics['observable_vector_rmse']=float(np.sqrt(np.mean((np.array(metrics['observables'])-rm['observables'])**2)))
                    if ref_path.exists():metrics.update(engine.reference_errors(rho,metrics,ref_path,rm))
                exact=cfg['family'] in ['dense','tensor_network']
                if exact and reference is None:
                    progress('write_reference',0,1,{})
                    np.save(ref_path,rho);reference=dict(method=cfg['id'],metrics=metrics)
                    common.atomic_json(ref_json,reference)
                    metrics.update(fidelity_score_abs_error=0.,observable_vector_rmse=0.,full_density_frobenius_error=0.)
                if req.get('pilot') and exact:
                    from .kernels import aer
                    independent,_=aer(job['variant'],dict(aer_method='density_matrix',gpu=False))
                    err=float(np.linalg.norm(rho-independent));diagnostics['independent_aer_error']=err
                    if err>1e-7:raise ValueError(f'pilot exact/Aer disagreement {err}')
                result=dict(status='success',**{k:v for k,v in partial.items() if k not in ['status','timings']},metrics=metrics,
                    reference_method=reference['method'] if reference else None,
                    timings=partial['timings']|dict(scoring_s=scoring_s,reference_compare_io_s=time.perf_counter()-tick))
                del rho
        except Exception as e:
            partial=result_path.with_suffix('.partial.json')
            result=json.loads(partial.read_text()) if partial.exists() else {}
            result.update(status='resource_excluded' if isinstance(e,MemoryError) or 'OutOfMemory' in type(e).__name__ else 'error',
                error=str(e),traceback=traceback.format_exc())
        result.update(request_id=req['request_id'],worker_pid=os.getpid(),finished_at=time.time(),
            worker_wall_s=time.perf_counter()-start,worker_cpu_s=time.process_time()-cpu_start)
        common.atomic_json(result_path,result)
        try:
            if a.gpu:engine.cleanup()
            else:
                import gc;gc.collect()
        except Exception:traceback.print_exc()
        common.atomic_json(a.ipc/'done.json',dict(request_id=req['request_id']))

if __name__=='__main__':main()
