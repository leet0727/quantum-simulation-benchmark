"""One supervised subprocess per lane, warmed once and restarted on timeout."""
import json,os,signal,subprocess,sys,threading,time,uuid
from pathlib import Path
import psutil
from .common import atomic_json

TERMINAL={'success','timeout','error','resource_excluded','memory_limit','worker_crash'}

class Lane:
    def __init__(self,index,gpu,out,config,stop_event):
        self.index=index;self.gpu=gpu;self.out=out;self.config=config;self.stop_event=stop_event
        self.ipc=out/'workers'/str(index);self.ipc.mkdir(parents=True,exist_ok=True)
        self.proc=None;self.log=None;self.current={};self.telemetry={};self.restarts=0
        if gpu is not None and config['telemetry']['enabled']:
            threading.Thread(target=self.sample_gpu,daemon=True).start()

    def sample_gpu(self):
        try:
            import pynvml as nv
            nv.nvmlInit()
            h=nv.nvmlDeviceGetHandleByUUID(self.gpu.encode()) if isinstance(self.gpu,str) else nv.nvmlDeviceGetHandleByIndex(self.gpu)
            getters=dict(memory_bytes=lambda:nv.nvmlDeviceGetMemoryInfo(h).used,
                utilization_pct=lambda:nv.nvmlDeviceGetUtilizationRates(h).gpu,
                sm_clock_mhz=lambda:nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_SM),
                memory_clock_mhz=lambda:nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_MEM),
                power_w=lambda:nv.nvmlDeviceGetPowerUsage(h)/1000,
                energy_mj=lambda:nv.nvmlDeviceGetTotalEnergyConsumption(h),
                temperature_c=lambda:nv.nvmlDeviceGetTemperature(h,nv.NVML_TEMPERATURE_GPU))
            while not self.stop_event.is_set():
                row=dict(started_at=time.time())
                for name,get in getters.items():
                    try:row[name]=get()
                    except Exception as e:row[name]=None;row[name+'_error']=type(e).__name__
                row['finished_at']=time.time();self.telemetry=row
                self.stop_event.wait(self.config['telemetry']['interval_s'])
        except Exception as e:self.telemetry=dict(error=str(e))

    def stop(self):
        proc=self.proc
        if proc and proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()
        if proc and proc.stdin and not proc.stdin.closed:proc.stdin.close()
        if self.log and not self.log.closed:self.log.close()

    def start(self):
        if self.proc and self.proc.poll() is None:return
        self.stop()
        for name in ['ready.json','phase.json','done.json']:(self.ipc/name).unlink(missing_ok=True)
        if self.log:self.log.close()
        self.log=(self.ipc/'worker.log').open('a')
        threads=str(self.config['execution']['threads_per_worker'])
        env=dict(os.environ,OPENBLAS_NUM_THREADS=threads,OMP_NUM_THREADS=threads,MKL_NUM_THREADS=threads,
            RAYON_NUM_THREADS='1',QISKIT_PARALLEL='FALSE',PYTHONUNBUFFERED='1',
            CUDA_VISIBLE_DEVICES=str(self.gpu) if self.gpu is not None else '')
        cmd=[sys.executable,'-m','simbench.worker','--ipc',str(self.ipc)]
        if self.gpu is not None:cmd.append('--gpu')
        self.proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=self.log,stderr=subprocess.STDOUT,
            text=True,bufsize=1,env=env,start_new_session=True)
        start=time.monotonic();self.current=dict(phase='initializing',pid=self.proc.pid)
        while not (self.ipc/'ready.json').exists():
            if self.stop_event.is_set():self.stop();raise InterruptedError('stopped')
            if self.proc.poll() is not None:raise RuntimeError(f'worker {self.index} startup failed; inspect {self.ipc}/worker.log')
            if time.monotonic()-start>self.config['startup_timeout_s']:self.stop();raise RuntimeError('worker startup timed out')
            time.sleep(.1)
        with (self.ipc/'startups.jsonl').open('a') as f:f.write(json.dumps(json.loads((self.ipc/'ready.json').read_text()))+'\n')

    def invoke(self,request,timeout):
        self.start();request=dict(request,request_id=uuid.uuid4().hex,settings=self.config)
        result_path=Path(request['result']);result_path.unlink(missing_ok=True);partial=result_path.with_suffix('.partial.json');partial.unlink(missing_ok=True)
        for name in ['phase.json','done.json']:(self.ipc/name).unlink(missing_ok=True)
        self.proc.stdin.write(json.dumps(request)+'\n');self.proc.stdin.flush()
        process=psutil.Process(self.proc.pid);cpu0=sum(process.cpu_times()[:2]);start=time.monotonic()
        peak=0;cpu=0.;gpu_peak=None;energy=0.;last_power=None;fresh=0;samples=0;status=None;phase={}
        interval=self.config['telemetry']['interval_s'];last_sample=-interval
        self.current=dict(case=Path(request['folder']).name,method=request.get('job',{}).get('config',{}).get('id',request['operation']),
            pid=self.proc.pid,timeout_s=timeout,phase='starting')
        with result_path.with_suffix('.resources.jsonl').open('w') as trace:
            while True:
                elapsed=time.monotonic()-start
                done=self.ipc/'done.json'
                if done.exists() and json.loads(done.read_text()).get('request_id')==request['request_id']:break
                if self.stop_event.is_set():status='interrupted';self.stop();break
                if self.proc.poll() is not None:status='worker_crash';break
                if elapsed>=timeout:status='timeout';self.stop();self.restarts+=1;break
                try:
                    rss=process.memory_info().rss;peak=max(peak,rss);cpu=max(cpu,sum(process.cpu_times()[:2])-cpu0)
                    if rss>self.config['resources']['host_rss_gib']*1024**3:status='memory_limit';self.stop();self.restarts+=1;break
                except psutil.Error:continue
                phase_path=self.ipc/'phase.json'
                if phase_path.exists():phase=json.loads(phase_path.read_text())
                self.current.update(elapsed_s=elapsed,rss_bytes=rss,worker_phase=phase)
                if elapsed-last_sample>=interval:
                    last_sample=elapsed;g=dict(self.telemetry);age=time.time()-g.get('finished_at',0)
                    io=process.io_counters();ctx=process.num_ctx_switches()
                    row=dict(at=time.time(),elapsed_s=elapsed,rss_bytes=rss,cpu_s=cpu,threads=process.num_threads(),
                        read_bytes=io.read_bytes,write_bytes=io.write_bytes,context_switches=ctx._asdict(),gpu=g,gpu_age_s=age)
                    trace.write(json.dumps(row)+'\n');trace.flush();samples+=1
                    if age<max(5,3*interval):
                        fresh+=1
                        if g.get('memory_bytes') is not None:
                            gpu_peak=max(gpu_peak or 0,g['memory_bytes'])
                            if gpu_peak>self.config['resources']['gpu_memory_gib']*1024**3:status='memory_limit';self.stop();self.restarts+=1;break
                        if g.get('power_w') is not None:
                            if last_power:energy+=(last_power[1]+g['power_w'])*.5*(elapsed-last_power[0])
                            last_power=(elapsed,g['power_w'])
                time.sleep(.05)
        r=json.loads(result_path.read_text()) if result_path.exists() else (json.loads(partial.read_text()) if partial.exists() else {})
        if status:r.update(status=status,last_phase=phase)
        r.update(supervisor_wall_s=time.monotonic()-start,sampled_peak_rss_bytes=peak,process_cpu_s=cpu,
            gpu_peak_memory_bytes=gpu_peak,gpu_fresh_samples=fresh,resource_samples=samples,
            gpu_energy_j_device=energy if last_power and fresh>1 else None,worker_lane=self.index,gpu_assignment=self.gpu,
            gpu_cycles=None,gpu_cycles_reason='kernel hardware counters not collected; clocks are not cycle counts',
            telemetry_scope='device-wide; concurrent load may affect readings',recorded_at=time.time())
        atomic_json(result_path,r);return r

    def job(self,job,pilot=False):
        base=self.out/('pilot' if pilot else 'cases');folder=base/job['case'];folder.mkdir(parents=True,exist_ok=True)
        result=folder/(job['config']['id']+f"__r{job['repeat']}.json")
        if result.exists():
            r=json.loads(result.read_text())
            if r.get('status') in TERMINAL:return r
        prepared=folder/'prepared.json'
        if not prepared.exists():
            prep=self.invoke(dict(operation='prepare',spec=job['spec'],folder=str(folder),result=str(folder/'preparation.json')),
                             self.config['preparation_timeout_s'])
            if prep['status']!='success':
                r=dict(status=prep['status'],error='preparation failed',preparation=prep,job=job)
                atomic_json(result,r);return r
        r=self.invoke(dict(operation='simulate',job=job,pilot=pilot,folder=str(folder),result=str(result)),self.config['timeout_s'])
        r['job']=job;atomic_json(result,r);self.current={};return r
