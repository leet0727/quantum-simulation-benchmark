"""Independent representations, all consuming the same frozen quantum channels."""
from __future__ import annotations
import time, itertools, math
from . import common
import numpy as np
from .common import PAULI, synchronize
from .density_mps import DensityMPS, interleaved_channel

def host(x): return x.get() if hasattr(x,'get') else np.asarray(x)

def density_tensor_reduced(tensor,n,support,xp=np):
    # tensor axes are ascending sites, each site's digit is (row,column).
    t=tensor.reshape((2,2)*n); current=list(range(n))
    for site in reversed(range(n)):
        if site not in support:
            k=current.index(site); t=xp.trace(t,axis1=2*k,axis2=2*k+1); current.remove(site)
    if not support: return host(t).reshape(1,1)
    order=[2*current.index(q) for q in reversed(support)]+[2*current.index(q)+1 for q in reversed(support)]
    return host(t.transpose(order).reshape(2**len(support),2**len(support)))

def dense(v,cfg,ideal=False):
    xp=__import__('cupy') if cfg.get('gpu') else np
    state=xp.zeros((4,)*v['n'],dtype=xp.complex128); state[(0,)*v['n']]=1
    stream=v['ideal_stream'] if ideal else v['stream']
    for sites,ch in stream:
        mat=xp.asarray(interleaved_channel(ch,len(sites))).reshape((4,)*(2*len(sites)))
        state=xp.tensordot(mat,state,axes=(list(range(len(sites),2*len(sites))),list(sites)))
        state=xp.ascontiguousarray(xp.moveaxis(state,list(range(len(sites))),list(sites)))
    synchronize()
    return density_tensor_reduced(state,v['n'],v['support'],xp),{}

def density_mps(v,cfg):
    state=DensityMPS(v['n'],cfg['chi'],gpu=cfg.get('gpu',False))
    for sites,ch in v['stream']: state.apply(sites,ch)
    rho=host(state.reduced(v['support'])); synchronize()
    return rho,state.diagnostics()

def dmrg(v,cfg):
    from .dmrg_engine import ProjectionMPSEngine
    engine=ProjectionMPSEngine(bond_dim=cfg['chi'],max_sweeps=cfg.get('sweeps',4),scratch_fraction=.15,seed=cfg.get('seed',1729))
    try:
        ops=[(sites,interleaved_channel(ch,len(sites)),False) for sites,ch in v['stream']]
        ts,meta=engine.run([4]*v['n'],ops)
        # Preserve raw density scale; no trace normalization before signed reconstruction.
        state=DensityMPS(v['n'],cfg['chi'],gpu=False)
        state.tensors=[t[None,:,:] if t.ndim==2 and i==0 else t[:,:,None] if t.ndim==2 else t for i,t in enumerate(ts)]
        return state.reduced(v['support']),meta
    finally: engine.close()

def tensor_network(v,cfg):
    import opt_einsum as oe
    xp=__import__('cupy') if cfg.get('gpu') else np
    args=[]; legs=list(range(v['n'])); counter=v['n']
    zero=xp.array([1,0,0,0],dtype=complex)
    for leg in legs: args.extend([zero,[leg]])
    for sites,ch in v['stream']:
        ins=[legs[s] for s in sites]; outs=list(range(counter,counter+len(sites))); counter+=len(sites)
        args.extend([xp.asarray(interleaved_channel(ch,len(sites))).reshape((4,)*(2*len(sites))),outs+ins])
        for s,l in zip(sites,outs): legs[s]=l
    for q in range(v['n']):
        if q not in v['support']:args.extend([xp.array([1,0,0,1],complex),[legs[q]]])
    ordered=sorted(v['support']); args.append([legs[q] for q in ordered])
    tick=time.perf_counter(); path,info=oe.contract_path(*args,optimize='greedy')
    planning=time.perf_counter()-tick
    if int(info.largest_intermediate)*16>8*1024**3: raise MemoryError('TN intermediate exceeds 8 GiB')
    value=oe.contract(*args,optimize=path); synchronize()
    return density_tensor_reduced(value,len(ordered),[ordered.index(q) for q in v['support']],xp),dict(path_search_s=planning,estimated_flops=float(info.opt_cost),largest_intermediate_elements=int(info.largest_intermediate))

def aer(v,cfg):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Kraus
    from qiskit_aer import AerSimulator
    method=cfg.get('aer_method','density_matrix'); qc=QuantumCircuit(v['n'])
    for sites,ks in v['kraus']:
        qc.append(Kraus(ks).to_instruction(),list(sites))
    qc.save_density_matrix(v['support'],label='rho')
    options=dict(method=method,device='GPU' if cfg.get('gpu') else 'CPU',max_parallel_threads=2,max_parallel_experiments=1)
    if method=='matrix_product_state':options.update(matrix_product_state_max_bond_dimension=cfg['chi'],matrix_product_state_truncation_threshold=1e-12)
    sim=AerSimulator(**options)
    res=sim.run(qc,shots=cfg.get('shots',1),seed_simulator=cfg.get('seed',19)).result()
    if not res.success: raise RuntimeError(str(res.status))
    md=res.results[0].metadata
    if cfg.get('gpu') and md.get('device')!='GPU':raise RuntimeError('GPU not confirmed')
    return np.asarray(res.data(0)['rho']),dict(device=md.get('device'),engine=md.get('method'),shots=cfg.get('shots',1))

def apply_columns(x,op,sites,n,xp):
    axes=[n-1-q for q in reversed(sites)]; rest=[q for q in range(n) if q not in axes]+[n]
    shape=x.reshape((2,)*n+(x.shape[1],))
    perm=axes+rest; inv=np.argsort(perm)
    out=xp.asarray(op)@shape.transpose(perm).reshape(2**len(sites),-1)
    return out.reshape((2,)*n+(x.shape[1],)).transpose(tuple(inv)).reshape(2**n,x.shape[1])

def low_rank(v,cfg):
    xp=__import__('cupy') if cfg.get('gpu') else np
    x=xp.zeros((2**v['n'],1),complex); x[0,0]=1; discarded=0.; actual=1
    for sites,ks in v['kraus']:
        x=xp.concatenate([apply_columns(x,k,sites,v['n'],xp) for k in ks],axis=1)
        if x.shape[1]>cfg['rank']:
            u,s,_=xp.linalg.svd(x,full_matrices=False); keep=min(cfg['rank'],len(s))
            discarded+=float(xp.sum(s[keep:]**2).item()); x=u[:,:keep]*s[:keep]
        actual=max(actual,x.shape[1])
    axes=[v['n']-1-q for q in reversed(v['support'])]; other=[q for q in range(v['n']) if q not in axes]
    x=x.reshape((2,)*v['n']+(x.shape[1],)).transpose(axes+other+[v['n']]).reshape(2**len(axes),-1)
    rho=x@x.conj().T; synchronize()
    return host(rho),dict(rank_cap=cfg['rank'],max_rank=actual,discarded_trace=discarded,representation='global low-rank purification; not local purification')

def sparse_pauli(v,cfg):
    from qiskit.quantum_info import PTM,SuperOp
    n=v['n']; terms={sum(3<<(2*q) for q in range(n) if bits>>q&1):1. for bits in range(2**n)}
    discarded=0.; peak=len(terms)
    for sites,ch in v['stream']:
        ptm=np.asarray(PTM(SuperOp(ch)).data).real
        output_codes=[sum(((row>>(2*j))&3)<<(2*q) for j,q in enumerate(sites)) for row in range(len(ptm))]
        nonzero=[[(output_codes[row],float(a)) for row,a in enumerate(ptm[:,col]) if abs(a)>1e-13] for col in range(len(ptm))]
        clear=sum(3<<(2*q) for q in sites); new={}
        for code,coeff in terms.items():
            col=sum(((code>>(2*q))&3)<<(2*j) for j,q in enumerate(sites)); base=code&~clear
            for encoded,a in nonzero[col]:
                out=base|encoded; new[out]=new.get(out,0.)+a*coeff
        peak=max(peak,len(new)); kept={k:c for k,c in new.items() if k==0 or abs(c)>=cfg.get('cutoff',1e-6)}
        discarded+=sum(abs(c) for k,c in new.items() if k not in kept)
        if len(kept)>cfg['terms']:
            chosen=sorted(kept,key=lambda k:(k!=0,-abs(kept[k]),k))[:cfg['terms']]
            chosen_set=set(chosen)
            discarded+=sum(abs(c) for k,c in kept.items() if k not in chosen_set); kept={k:kept[k] for k in chosen}
        terms=kept
    support=v['support']; mask=sum(3<<(2*q) for q in support); rho=np.zeros((2**len(support),)*2,complex)
    for code,c in terms.items():
        if code&~mask: continue
        p=np.ones((1,1),complex)
        for q in reversed(support):p=np.kron(p,PAULI[(code>>(2*q))&3])
        rho+=c*p/(2**len(support))
    return rho,dict(retained_terms=len(terms),peak_terms=peak,discarded_coefficient_l1=discarded,representation='forward density Pauli expansion; not LOWESA')

def evaluate(v,cfg):
    if not v['support']:return np.ones((1,1)),dict(identity_only=True)
    methods={'dense':dense,'density_mps':density_mps,'dmrg':dmrg,'tensor_network':tensor_network,'aer':aer,'low_rank':low_rank,'sparse_pauli':sparse_pauli}
    return methods[cfg['family']](v,cfg)

def configurations():
    return [
        dict(id='exact_cpu',family='dense',gpu=False),
        dict(id='exact_gpu',family='dense',gpu=True),
        dict(id='density_mps_cpu_chi16',family='density_mps',gpu=False,chi=16),
        dict(id='density_mps_gpu_chi4',family='density_mps',gpu=True,chi=4),
        dict(id='density_mps_gpu_chi16',family='density_mps',gpu=True,chi=16),
        dict(id='dmrg_gpu_chi16',family='dmrg',gpu=True,chi=16,sweeps=4),
        dict(id='trajectory_dense_128',family='aer',aer_method='statevector',gpu=False,shots=128),
        dict(id='trajectory_dense_1024',family='aer',aer_method='statevector',gpu=False,shots=1024),
        dict(id='trajectory_mps_128_chi8',family='aer',aer_method='matrix_product_state',gpu=False,shots=128,chi=8),
        dict(id='tensor_network_gpu',family='tensor_network',gpu=True),
        dict(id='low_rank_gpu_r8',family='low_rank',gpu=True,rank=8),
        dict(id='low_rank_gpu_r32',family='low_rank',gpu=True,rank=32),
        dict(id='sparse_pauli_256',family='sparse_pauli',gpu=False,terms=256,cutoff=1e-5),
        dict(id='sparse_pauli_2048',family='sparse_pauli',gpu=False,terms=2048,cutoff=1e-7),
    ]
