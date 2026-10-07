"""Strict, dependency-free input validation and deterministic experiment expansion."""
import copy, hashlib, itertools, json, math
from pathlib import Path

FAMILIES=['chain','brickwork','clifford','clusters','all_to_all','qaoa','grid2d','qft']
METHODS=['dense','density_mps','dmrg','low_rank','trajectory_dense','trajectory_mps','sparse_pauli','tensor_network']
DEFAULTS=dict(output_dir='results/experiment',timeout_s=7200,preparation_timeout_s=7200,startup_timeout_s=300,
    trajectory_repeats=1,noise=dict(backend='FakeMarrakesh',optimization_level=1),
    execution=dict(mode='isolated',gpu_ids=[0],cpu_workers=0,threads_per_worker=2,scoring_device='auto'),
    resources=dict(host_rss_gib=96,gpu_memory_gib=42,workspace_gib=40,tn_intermediate_gib=16,
                   max_output_gib=8,reference_cache_gib=48,min_free_disk_gib=20),
    scheduling=dict(mode='qubits_first',seed=2718,method_order=METHODS,
        preliminary=dict(enabled=True,methods=['dense:gpu','density_mps:cpu','density_mps:gpu',
            'low_rank:gpu','trajectory_dense:cpu','trajectory_dense:gpu','trajectory_mps:cpu','sparse_pauli:cpu'])),
    telemetry=dict(interval_s=1.,enabled=True),reference=dict(method='auto'),
    pilot=dict(enabled=True,qubits=4,layers=2,seed=22001),
    accuracy=dict(fidelity_score_abs_error=.01,observable_vector_rmse=.01))

def merge(default, supplied, path='config'):
    if not isinstance(supplied,dict):raise ValueError(f'{path} must be an object')
    unknown=set(supplied)-set(default)
    if unknown:raise ValueError(f'{path}: unknown keys {sorted(unknown)}')
    out=copy.deepcopy(default)
    for key,value in supplied.items():
        out[key]=merge(out[key],value,path+'.'+key) if isinstance(out[key],dict) else value
    return out

def positive(x,label):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0:raise ValueError(f'{label} must be positive')

def ints(xs,label):
    if not isinstance(xs,list) or not xs or any(type(x)!=int or x<1 for x in xs):raise ValueError(f'{label} must be a nonempty array of positive integers')
    if len(set(xs))!=len(xs):raise ValueError(f'{label} contains duplicates')
    return xs

def ladder(value,label):
    if isinstance(value,list):return sorted(ints(value,label))
    if not isinstance(value,dict) or set(value)!={'start','max','factor'}:raise ValueError(f'{label}: use an array or {{start,max,factor}}')
    a,b,f=value['start'],value['max'],value['factor']
    if any(type(x)!=int for x in (a,b,f)) or a<1 or b<a or f<2:raise ValueError(f'{label}: invalid geometric ladder')
    out=[a]
    while out[-1]<b:out.append(min(b,out[-1]*f))
    return out

def load(path):
    raw=json.loads(Path(path).read_text())
    if not isinstance(raw,dict):raise ValueError('input must be a JSON object')
    required={'circuits','methods'}
    if not required<=set(raw):raise ValueError('input needs circuits and methods')
    c=merge(DEFAULTS,{k:v for k,v in raw.items() if k not in required})
    circ=raw['circuits'];allowed={'families','qubits','layers','seeds','seed_count','seed_start'}
    if not isinstance(circ,dict) or set(circ)-allowed:raise ValueError('unknown circuit options')
    families=circ.get('families',FAMILIES)
    if not isinstance(families,list) or not families or len(set(families))!=len(families) or set(families)-set(FAMILIES):raise ValueError('invalid circuit families')
    qubits=sorted(ints(circ.get('qubits',[]),'qubits'));layers=sorted(ints(circ.get('layers',[]),'layers'))
    if min(qubits)<2:raise ValueError('this benchmark requires at least 2 qubits')
    if 'seeds' in circ:
        if {'seed_count','seed_start'}&set(circ):raise ValueError('use explicit seeds OR seed_count/seed_start')
        seeds=ints(circ['seeds'],'seeds')
    else:
        count=circ.get('seed_count',5);start=circ.get('seed_start',22101)
        ints([count],'seed_count');ints([start],'seed_start');seeds=list(range(start,start+count))
    c['circuits']=dict(families=families,qubits=qubits,layers=layers,seeds=seeds)
    if not isinstance(raw['methods'],list) or not raw['methods']:raise ValueError('methods must be a nonempty array')
    c['methods']=raw['methods'];expand_methods(c) # validate before any work
    for key in ['timeout_s','preparation_timeout_s','startup_timeout_s']:positive(c[key],key)
    ints([c['trajectory_repeats']],'trajectory_repeats')
    ex=c['execution']
    if ex['mode'] not in ['isolated','parallel']:raise ValueError('execution.mode must be isolated or parallel')
    if ex['scoring_device'] not in ['auto','cpu','gpu']:raise ValueError('invalid scoring_device')
    if type(ex['cpu_workers'])!=int or ex['cpu_workers']<0:raise ValueError('cpu_workers must be >=0')
    if not isinstance(ex['gpu_ids'],list) or any(not (type(x)==int and x>=0 or isinstance(x,str) and x.startswith(('GPU-','MIG-'))) for x in ex['gpu_ids']):raise ValueError('gpu_ids: use physical ordinals or GPU/MIG UUIDs')
    if len(set(ex['gpu_ids']))!=len(ex['gpu_ids']):raise ValueError('duplicate GPU assignments are not allowed')
    ints([ex['threads_per_worker']],'threads_per_worker')
    if not ex['gpu_ids'] and ex['cpu_workers']==0:raise ValueError('configure at least one worker')
    if ex['mode']=='isolated' and len(ex['gpu_ids'])+ex['cpu_workers']!=1:raise ValueError('isolated mode needs exactly one worker')
    if any(m['gpu'] for m in expand_methods(c)) and not ex['gpu_ids']:raise ValueError('GPU methods need gpu_ids')
    if ex['scoring_device']=='gpu' and ex['cpu_workers']:raise ValueError('GPU scoring requires GPU workers; use auto or cpu for CPU workers')
    for key,val in c['resources'].items():positive(val,'resources.'+key)
    positive(c['telemetry']['interval_s'],'telemetry.interval_s')
    if c['noise']['backend'] not in ['FakeMarrakesh','none']:raise ValueError('noise.backend must be FakeMarrakesh or none')
    if type(c['noise']['optimization_level'])!=int or c['noise']['optimization_level'] not in range(4):raise ValueError('optimization_level must be 0..3')
    if c['reference']['method'] not in ['auto','dense_cpu','dense_gpu','none']:raise ValueError('invalid reference.method')
    if c['reference']['method']=='dense_gpu' and not ex['gpu_ids']:raise ValueError('dense_gpu reference needs a GPU worker')
    sched=c['scheduling']
    if sched['mode'] not in ['qubits_first','budgets_first','case_first','custom']:raise ValueError('invalid scheduling.mode')
    if set(sched['method_order'])!=set(METHODS) or len(sched['method_order'])!=len(METHODS):raise ValueError('method_order must list every method family once')
    if 'order_by' in sched:validate_order(sched['order_by'])
    for key in ['qubits','layers','seed']:ints([c['pilot'][key]],'pilot.'+key)
    if c['pilot']['qubits']<2:raise ValueError('pilot qubits >=2')
    for x in c['accuracy'].values():positive(x,'accuracy threshold')
    if not isinstance(c['output_dir'],str) or not c['output_dir']:raise ValueError('output_dir must be a path')
    return c

def validate_order(keys):
    allowed={'qubits','layers','budget','family','seed','method','device','repeat'}
    if not isinstance(keys,list) or not keys or len(set(keys))!=len(keys) or set(keys)-allowed:raise ValueError('invalid custom scheduling.order_by')

DEFAULTS['scheduling']['order_by']=['qubits','budget','layers','seed','family','method','device','repeat']

def expand_methods(config):
    result=[];ids=set()
    for spec in config['methods']:
        if not isinstance(spec,dict):raise ValueError('each method must be an object')
        name=spec.get('name')
        if name not in METHODS:raise ValueError(f'unknown method {name!r}')
        allowed={'name','devices'}
        if name in ['density_mps','dmrg','trajectory_mps']:allowed.add('bond_dimensions')
        if name=='dmrg':allowed.add('sweeps')
        if name=='low_rank':allowed.add('ranks')
        if name in ['trajectory_dense','trajectory_mps']:allowed.add('shots')
        if name=='trajectory_mps':allowed.update(['sweep','fixed_shots'])
        if name=='sparse_pauli':allowed.update(['terms','cutoffs'])
        if set(spec)-allowed:raise ValueError(f'{name}: unsupported options {set(spec)-allowed}')
        devices=spec.get('devices',['gpu'] if name=='dmrg' else ['cpu'])
        if not isinstance(devices,list) or not devices or set(devices)-{'cpu','gpu'} or len(set(devices))!=len(devices):raise ValueError('devices must be cpu and/or gpu')
        if name in ['trajectory_mps','sparse_pauli'] and devices!=['cpu']:raise ValueError(f'{name} only supports CPU in this suite')
        if name=='dmrg' and devices!=['gpu']:raise ValueError('projection DMRG requires GPU')
        budgets=[{}]
        if name in ['density_mps','dmrg']:budgets=[dict(chi=x) for x in ladder(spec.get('bond_dimensions',[16]),name+'.bond_dimensions')]
        if name=='low_rank':budgets=[dict(rank=x) for x in ladder(spec.get('ranks',[8]),name+'.ranks')]
        if name=='trajectory_dense':budgets=[dict(shots=x) for x in ladder(spec.get('shots',[256]),name+'.shots')]
        if name=='sparse_pauli':
            terms=ladder(spec.get('terms',[2048]),name+'.terms');cuts=spec.get('cutoffs',[1e-7]*len(terms))
            if not isinstance(cuts,list) or len(cuts)!=len(terms) or any(isinstance(x,bool) or not isinstance(x,(int,float)) or x<0 for x in cuts):raise ValueError('cutoffs must be nonnegative and match terms')
            budgets=[dict(terms=t,cutoff=x) for t,x in zip(terms,cuts)]
        if name=='trajectory_mps':
            chis=ladder(spec.get('bond_dimensions',[16]),name+'.bond_dimensions');shots=ladder(spec.get('shots',[256]),name+'.shots')
            sweep=spec.get('sweep','paired_with_fixed_shots')
            if sweep=='cartesian':budgets=[dict(chi=b,shots=s) for b,s in itertools.product(chis,shots)]
            elif sweep in ['paired','paired_with_fixed_shots']:
                if len(chis)!=len(shots) and min(len(chis),len(shots))!=1:raise ValueError('paired MPS ladders need equal lengths or one fixed value')
                budgets=[dict(chi=chis[min(i,len(chis)-1)],shots=shots[min(i,len(shots)-1)]) for i in range(max(len(chis),len(shots)))]
                if sweep=='paired_with_fixed_shots':
                    fixed=spec.get('fixed_shots',shots[0]);ints([fixed],'fixed_shots')
                    budgets.extend(dict(chi=x,shots=fixed) for x in chis if dict(chi=x,shots=fixed) not in budgets)
            else:raise ValueError('invalid trajectory_mps.sweep')
            budgets.sort(key=lambda x:(x['chi'],x['shots']))
        for i,budget in enumerate(budgets):
            for device in devices:
                family='aer' if name.startswith('trajectory_') else name
                cfg=dict(name=name,family=family,gpu=device=='gpu',budget=i,**budget)
                if name.startswith('trajectory_'):cfg['aer_method']='statevector' if name=='trajectory_dense' else 'matrix_product_state'
                if name=='dmrg':cfg['sweeps']=spec.get('sweeps',4);ints([cfg['sweeps']],'sweeps')
                suffix=''.join(f'_{k}{v}' for k,v in budget.items())
                cfg['id']=f'{name}_{device}{suffix}'
                if cfg['id'] in ids:raise ValueError(f'duplicate method setting {cfg["id"]}')
                ids.add(cfg['id']);result.append(cfg)
    return result

def fingerprint(config):
    # Hardware/output location may change on resume; scientific conditions may not.
    scientific={k:v for k,v in config.items() if k not in ['output_dir','execution','telemetry']}
    return hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest()
