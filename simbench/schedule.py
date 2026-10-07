"""Declarative scheduling; phase barriers preserve easier/low-qubit first order."""
import hashlib,itertools
from .config import expand_methods

def case_id(s):return f"{s['family']}_n{s['n']}_d{s['depth']}_s{s['seed']}"

def expand(config):
    methods=expand_methods(config);reference=config['reference']['method']
    if reference=='auto':reference='dense_gpu' if config['execution']['gpu_ids'] else 'dense_cpu'
    if reference!='none' and not any(m['id']==reference for m in methods):
        methods.insert(0,dict(id=reference,name='dense',family='dense',gpu=reference.endswith('gpu'),budget=0))
    circ=config['circuits'];cases=[dict(family=f,n=n,depth=d,seed=s) for n,d,s,f in itertools.product(circ['qubits'],circ['layers'],circ['seeds'],circ['families'])]
    sched=config['scheduling'];mode=sched['mode'];pre=sched['preliminary'];priority={m:i for i,m in enumerate(sched['method_order'])}
    orders=dict(qubits_first=['qubits','budget','layers','seed','family','method','device','repeat'],
                budgets_first=['budget','qubits','layers','seed','family','method','device','repeat'],
                case_first=['qubits','layers','seed','family','budget','method','device','repeat'])
    order=sched['order_by'] if mode=='custom' else orders[mode]
    jobs=[]
    for spec in cases:
        for method in methods:
            for repeat in range(config['trajectory_repeats'] if method['family']=='aer' else 1):
                small=spec['n']==min(circ['qubits']) and spec['depth']==min(circ['layers'])
                early=pre['enabled'] and small and method['budget']==0 and (method['id']==reference or f"{method['name']}:{'gpu' if method['gpu'] else 'cpu'}" in pre['methods'])
                vals=dict(qubits=spec['n'],layers=spec['depth'],budget=method['budget'],seed=circ['seeds'].index(spec['seed']),
                    family=circ['families'].index(spec['family']),method=priority[method['name']],device=not method['gpu'],repeat=repeat)
                # Exact reference should precede approximations on the same case.
                if method['id']==reference:vals['budget']=-1;vals['method']=-1
                phase=(0,) if early else (1,vals[order[0]])
                cfg=dict(method);cfg['seed']=int(hashlib.sha256(f"{case_id(spec)}:{method['id']}:{repeat}:{sched['seed']}".encode()).hexdigest()[:8],16)
                jobs.append(dict(id=f'{case_id(spec)}__{method["id"]}__r{repeat}',case=case_id(spec),spec=spec,
                    config=cfg,repeat=repeat,phase=list(phase),sort=(0 if early else 1,)+tuple(vals[x] for x in order)))
    jobs.sort(key=lambda j:j['sort'])
    for i,j in enumerate(jobs):j['index']=i;j.pop('sort')
    return jobs,methods

def pilots(config):
    from copy import deepcopy
    c=deepcopy(config);c['circuits']=dict(families=config['circuits']['families'],qubits=[config['pilot']['qubits']],
        layers=[config['pilot']['layers']],seeds=[config['pilot']['seed']]);c['trajectory_repeats']=1
    jobs,_=expand(c)
    return [j for j in jobs if j['config']['budget']==0]
