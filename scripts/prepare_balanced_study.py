"""One-time local migration from the stopped broad study; never run on an initialized study."""
import copy,json,shutil,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from simbench.config import load
from simbench.schedule import expand
from simbench.runner import source_hash,result_path
root=Path('/mnt/main/tklee/quantum-simulation-benchmark/results/balanced_simulation_study')
# Check BEFORE rewriting any configuration or importing any records.
if (root/'study.json').exists() or any(root.glob('*/n*/manifest.json')):
 raise SystemExit('Study already initialized; use run_balanced_study.py to resume. No files changed.')
oldpath=Path('/mnt/main/tklee/quantum-simulation-benchmark/results/two_gpu_campaign_64gib')
old=json.loads((oldpath/'manifest.json').read_text())
assert old['source_hash']==source_hash()
assert json.loads((oldpath/'progress.json').read_text())['phase']=='stopped'
base=json.loads((REPO/'configs/two_gpu_64gib.json').read_text())
root.mkdir(exist_ok=True)
entries=[]
for stage in ['main','scaling']:
 for n in [10,12,14]:
  c=copy.deepcopy(base);c['output_dir']=str(root/stage/f'n{n}');c['circuits']['qubits']=[n]
  c['timeout_s']=300 if stage=='main' else 900
  if stage=='scaling':c['circuits']['layers']=[16,64];c['circuits']['seed_count']=2
  methods=[]
  for m in c['methods']:
   name=m['name']
   if stage=='scaling' and name in ['dense','tensor_network']:continue
   if name in ['density_mps','dmrg']:m['bond_dimensions']=[16,64,256] if stage=='main' else [512,1024]
   if name=='low_rank':m['ranks']=[8,32,128] if stage=='main' else [256,1024]
   if name=='trajectory_dense':m['shots']=[256,1024] if stage=='main' else [4096,16384]
   if name=='trajectory_mps':
    m['bond_dimensions']={10:[16,32],12:[16,64],14:[16,64,128]}[n] if stage=='main' else [2**(n//2)]
    m['shots']=[256,1024] if stage=='main' else [256,4096,16384]
    m['sweep']='cartesian';m.pop('fixed_shots',None)
   if name=='sparse_pauli':
    m['terms']=[2048,8192,32768] if stage=='main' else [131072,524288]
    m['cutoffs']=[1e-5,1e-6,1e-7] if stage=='main' else [1e-8,1e-9]
   methods.append(m)
  c['methods']=methods
  relative=Path(f'configs/balanced_{stage}_n{n}.json')
  path=REPO/relative;path.write_text(json.dumps(c,indent=2)+'\n')
  validated=load(path);jobs,_=expand(validated)
  for key in ['noise','resources','reference','trajectory_repeats','accuracy','execution']:
   assert validated[key]==old['config'][key],key
  out=Path(c['output_dir']);assert not (out/'manifest.json').exists()
  reused=[]
  for j in jobs:
   src=result_path(oldpath,j)
   if not src.exists():continue
   r=json.loads(src.read_text())
   if r['status']!='success' or max(r.get('supervisor_wall_s',float('inf')),r.get('worker_wall_s',float('inf'))) > c['timeout_s']:continue
   def numerical(cfg):return {k:v for k,v in cfg.items() if k!='budget'}
   assert numerical(r['job']['config'])==numerical(j['config'])
   dst=result_path(out,j);dst.parent.mkdir(parents=True,exist_ok=True)
   provenance=dict(original_result=str(src),original_fingerprint=old['fingerprint'],source_hash=old['source_hash'],original_timeout_s=old['config']['timeout_s'],new_timeout_s=c['timeout_s'],original_job=r['job'],timing_origin='original campaign; not remeasured')
   r['reuse_provenance']=provenance;r['job']=j;r['config']['budget']=j['config']['budget']
   dst.write_text(json.dumps(r,indent=2)+'\n');reused.append(str(src))
   # Reuse immutable prepared inputs and exact oracle; never copy other old terminal records.
   for artifact in src.parent.iterdir():
    if artifact.name in ['input.pkl','parent.qpy','compiled.qpy','backend_snapshot.json','prepared.json','preparation.json','reference.json','reference.npy'] and not (dst.parent/artifact.name).exists():
     shutil.copy2(artifact,dst.parent/artifact.name)
  entry=dict(config=str(relative),output_dir=str(out),runs=len(jobs),reused_successes=len(reused),timeout_s=c['timeout_s'])
  entries.append(entry)
  out.mkdir(parents=True,exist_ok=True);(out/'reuse_manifest.json').write_text(json.dumps(dict(stage=entry,original_results=reused),indent=2)+'\n')
  print(entry)
assert sum(e['runs'] for e in entries[:3])==10680
assert sum(e['runs'] for e in entries[3:])==1920
(root/'study.json').write_text(json.dumps(dict(name='balanced_simulation_study',stages=entries,original_output=str(oldpath)),indent=2)+'\n')
shutil.copy2(oldpath/'RESEARCH_REVIEW_AND_QUESTIONS.md',root/'RESEARCH_REVIEW_AND_QUESTIONS.md')
