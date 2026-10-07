"""Same numerical families as the original suite, with stage timing and progress."""
import time, gc
import numpy as np
from . import common
from . import kernels as s

def sync():
    import sys
    if 'cupy' in sys.modules:
        sys.modules['cupy'].cuda.runtime.deviceSynchronize()

def scores(rho,psi,words,*,eigensolver_gpu=True):
    """Legacy projected fidelity without constructing the projected dense matrix.

    All eigenpairs are still used; this is an algebraic optimization, not a new
    approximation. No trace-norm distance to the ideal is needed for this study.
    """
    raw=np.vdot(psi,rho@psi);trace=np.trace(rho)
    obs=common.pauli_expectations(rho,words)
    xp=__import__('cupy') if eigensolver_gpu else np
    h=xp.asarray(rho);h=(h+h.conj().T)/2
    vals,vecs=xp.linalg.eigh(h)
    descending=vals[::-1]
    offsets=(xp.cumsum(descending)-1)/xp.arange(1,len(vals)+1)
    active=xp.flatnonzero(descending>offsets)[-1]
    projected=xp.maximum(vals-offsets[active],0)
    amps=vecs.conj().T@xp.asarray(psi)
    fidelity=float(xp.sum(projected*xp.abs(amps)**2).real.item())
    minval=float(vals[0].item());negative=float(-xp.minimum(vals,0).sum().item())
    # Orthogonal Hermitian/anti-Hermitian parts; block to bound host temporaries.
    anti_sq=0.;norm_sq=0.
    for lo in range(0,len(rho),256):
        a=rho[lo:lo+256]; diff=a-rho[:,lo:lo+256].conj().T
        anti_sq+=float(np.vdot(diff,diff).real);norm_sq+=float(np.vdot(a,a).real)
    projection_sq=float(xp.sum((projected-vals)**2).item())+anti_sq/4
    if eigensolver_gpu:sync()
    return dict(state_fidelity=float(np.clip(fidelity,0,1)),raw_overlap_real=float(raw.real),
        raw_overlap_imag=float(raw.imag),raw_trace_real=float(trace.real),raw_trace_imag=float(trace.imag),
        minimum_eigenvalue=minval,negative_eigenvalue_mass=negative,
        raw_is_physical=bool(minval>=-1e-8 and abs(trace-1)<=1e-8 and anti_sq<=1e-16),
        hermiticity_error=float(np.sqrt(anti_sq)),projection_frobenius_distance=float(np.sqrt(projection_sq)),
        raw_frobenius_error=float(np.sqrt(max(0,norm_sq+1-2*raw.real))),
        observables=obs.tolist(),eigensolver_device='GPU' if eigensolver_gpu else 'CPU')

def reference_errors(rho,metrics,reference_path,reference_metrics):
    ref=np.load(reference_path,mmap_mode='r');sq=0.
    for lo in range(0,len(rho),256):
        delta=rho[lo:lo+256]-ref[lo:lo+256];sq+=float(np.vdot(delta,delta).real)
    return dict(fidelity_score_abs_error=abs(metrics['state_fidelity']-reference_metrics['state_fidelity']),
        observable_vector_rmse=float(np.sqrt(np.mean((np.array(metrics['observables'])-reference_metrics['observables'])**2))),
        full_density_frobenius_error=float(np.sqrt(sq)))

def evaluate(v,cfg,progress):
    """Evolution/output phases are explicit for directly implemented methods."""
    family=cfg['family'];xp=__import__('cupy') if cfg.get('gpu') else np
    tick=time.perf_counter();timings={};n=v['n']
    def advance(i,total,**extra): progress('evolution',i,total,extra)
    if family=='dense':
        state=xp.zeros((4,)*n,dtype=xp.complex128);state[(0,)*n]=1
        for i,(sites,ch) in enumerate(v['stream']):
            mat=xp.asarray(s.interleaved_channel(ch,len(sites))).reshape((4,)*(2*len(sites)))
            state=xp.tensordot(mat,state,axes=(list(range(len(sites),2*len(sites))),list(sites)))
            state=xp.ascontiguousarray(xp.moveaxis(state,list(range(len(sites))),list(sites)))
            advance(i+1,len(v['stream']))
        sync();timings['evolution_s']=time.perf_counter()-tick;tick=time.perf_counter()
        progress('output',0,1,{})
        rho=s.density_tensor_reduced(state,n,v['support'],xp);meta={}
    elif family=='density_mps':
        state=s.DensityMPS(n,cfg['chi'],gpu=cfg.get('gpu',False))
        for i,(sites,ch) in enumerate(v['stream']):
            state.apply(sites,ch);advance(i+1,len(v['stream']))
        sync();timings['evolution_s']=time.perf_counter()-tick;tick=time.perf_counter()
        progress('output',0,1,{})
        rho=s.host(state.reduced(v['support']));meta=state.diagnostics()
    elif family=='low_rank':
        x=xp.zeros((2**n,1),complex);x[0,0]=1;discarded=0.;actual=1;svds=0
        for i,(sites,ks) in enumerate(v['kraus']):
            cols=x.shape[1]*len(ks)
            # Conservative dense SVD storage guard including right singular vectors.
            m=2**n;k=min(m,cols);estimate=16*(4*m*cols+3*k*cols+3*m*k)
            if estimate>cfg.get('workspace_bytes',40*1024**3):
                raise MemoryError(f'low-rank estimated SVD workspace {estimate} bytes exceeds declared budget')
            x=xp.concatenate([s.apply_columns(x,k,sites,n,xp) for k in ks],axis=1)
            if x.shape[1]>cfg['rank']:
                u,sv,vh=xp.linalg.svd(x,full_matrices=False);keep=min(cfg['rank'],len(sv));svds+=1
                discarded+=float(xp.sum(sv[keep:]**2).item());x=u[:,:keep]*sv[:keep]
                del u,sv,vh
            actual=max(actual,x.shape[1]);advance(i+1,len(v['kraus']),rank=x.shape[1],discarded_trace=discarded)
        sync();timings['evolution_s']=time.perf_counter()-tick;tick=time.perf_counter()
        progress('output',0,1,{})
        axes=[n-1-q for q in reversed(v['support'])];other=[q for q in range(n) if q not in axes]
        x=x.reshape((2,)*n+(x.shape[1],)).transpose(axes+other+[n]).reshape(2**len(axes),-1)
        rho=s.host(x@x.conj().T)
        meta=dict(rank_cap=cfg['rank'],max_rank=actual,discarded_trace=discarded,svd_count=svds)
    elif family=='sparse_pauli':
        from qiskit.quantum_info import PTM,SuperOp
        terms={sum(3<<(2*q) for q in range(n) if bits>>q&1):1. for bits in range(2**n)}
        discarded=0.;peak=len(terms)
        for i,(sites,ch) in enumerate(v['stream']):
            ptm=np.asarray(PTM(SuperOp(ch)).data).real
            codes=[sum(((row>>(2*j))&3)<<(2*q) for j,q in enumerate(sites)) for row in range(len(ptm))]
            nonzero=[[(codes[row],float(a)) for row,a in enumerate(ptm[:,col]) if abs(a)>1e-13] for col in range(len(ptm))]
            clear=sum(3<<(2*q) for q in sites);new={}
            for code,c in terms.items():
                col=sum(((code>>(2*q))&3)<<(2*j) for j,q in enumerate(sites));base=code&~clear
                for encoded,a in nonzero[col]:
                    out=base|encoded;new[out]=new.get(out,0.)+a*c
            peak=max(peak,len(new));kept={k:c for k,c in new.items() if k==0 or abs(c)>=cfg['cutoff']}
            discarded+=sum(abs(c) for k,c in new.items() if k not in kept)
            if len(kept)>cfg['terms']:
                chosen=sorted(kept,key=lambda k:(k!=0,-abs(kept[k]),k))[:cfg['terms']];chosen=set(chosen)
                discarded+=sum(abs(c) for k,c in kept.items() if k not in chosen);kept={k:c for k,c in kept.items() if k in chosen}
            terms=kept;advance(i+1,len(v['stream']),retained_terms=len(terms))
        timings['evolution_s']=time.perf_counter()-tick;tick=time.perf_counter()
        support=v['support'];dim=2**len(support);rho=np.zeros((dim,dim),complex)
        mask=sum(3<<(2*q) for q in support);columns=np.arange(dim)
        # Same Pauli matrices as the old kron implementation, sparse permutation
        # accumulation avoids constructing one dense matrix per Pauli term.
        for j,(code,c) in enumerate(terms.items()):
            if code&~mask:continue
            rows=columns.copy();phase=np.ones(dim,complex)
            for q,site in enumerate(support):
                p=(code>>(2*site))&3;bit=(columns>>q)&1
                if p in (1,2):rows^=1<<q
                if p==2:phase*=1j*(1-2*bit)
                if p==3:phase*=1-2*bit
            rho[rows,columns]+=c*phase/dim
            progress('output',j+1,len(terms),{})
        meta=dict(retained_terms=len(terms),peak_terms=peak,discarded_coefficient_l1=discarded,
                  output_optimization='exact sparse Pauli permutation accumulation')
    elif family=='tensor_network':
        import opt_einsum as oe
        args=[];legs=list(range(n));counter=n;zero=xp.array([1,0,0,0],complex)
        for leg in legs:args.extend([zero,[leg]])
        for sites,ch in v['stream']:
            ins=[legs[q] for q in sites];outs=list(range(counter,counter+len(sites)));counter+=len(sites)
            args.extend([xp.asarray(s.interleaved_channel(ch,len(sites))).reshape((4,)*(2*len(sites))),outs+ins])
            for q,l in zip(sites,outs):legs[q]=l
        for q in range(n):
            if q not in v['support']:args.extend([xp.array([1,0,0,1],complex),[legs[q]]])
        ordered=sorted(v['support']);args.append([legs[q] for q in ordered])
        progress('contraction_planning',0,1,{})
        path,info=oe.contract_path(*args,optimize='greedy');planning=time.perf_counter()-tick
        if int(info.largest_intermediate)*16>cfg.get('tn_intermediate_bytes',16*1024**3):raise MemoryError('TN intermediate exceeds configured memory budget')
        progress('contraction',0,len(path),dict(estimated_flops=float(info.opt_cost)))
        value=oe.contract(*args,optimize=path);sync()
        timings['evolution_s']=time.perf_counter()-tick;tick=time.perf_counter();progress('output',0,1,{})
        rho=s.density_tensor_reduced(value,len(ordered),[ordered.index(q) for q in v['support']],xp)
        meta=dict(path_search_s=planning,estimated_flops=float(info.opt_cost),largest_intermediate_elements=int(info.largest_intermediate))
    elif family=='aer':
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Kraus
        from qiskit_aer import AerSimulator
        qc=QuantumCircuit(n)
        for i,(sites,ks) in enumerate(v['kraus']):
            qc.append(Kraus(ks).to_instruction(),list(sites));progress('aer_circuit_build',i+1,len(v['kraus']),{})
        qc.save_density_matrix(v['support'],label='rho')
        options=dict(method=cfg['aer_method'],device='GPU' if cfg.get('gpu') else 'CPU',
            max_parallel_threads=cfg.get('threads',2),max_parallel_experiments=1)
        if cfg['aer_method']=='matrix_product_state':
            options.update(matrix_product_state_max_bond_dimension=cfg['chi'],matrix_product_state_truncation_threshold=1e-12)
        sim=AerSimulator(**options);timings['aer_circuit_build_s']=time.perf_counter()-tick;tick=time.perf_counter()
        progress('trajectory_simulation',0,cfg['shots'],dict(progress_scope='Aer exposes no within-run shot progress'))
        res=sim.run(qc,shots=cfg['shots'],seed_simulator=cfg['seed']).result()
        if not res.success:raise RuntimeError(str(res.status))
        md=res.results[0].metadata
        if cfg.get('gpu') and md.get('device')!='GPU':raise RuntimeError('Aer GPU execution not confirmed')
        rho=np.asarray(res.data(0)['rho']);sync()
        timings['combined_evolution_output_s']=time.perf_counter()-tick
        meta=dict(device=md.get('device'),engine=md.get('method'),shots=cfg['shots'],aer_metadata=md,
            pure_state_maximum_possible_bond=2**(n//2),effective_bond_cap=min(cfg.get('chi',2**(n//2)),2**(n//2)))
        return rho,meta,timings
    else:
        progress('dmrg_sweeps',0,cfg.get('sweeps',1),dict(progress_scope='library sweep progress unavailable'))
        rho,meta=s.evaluate(v,cfg);sync()
        timings['combined_evolution_output_s']=time.perf_counter()-tick
        return rho,meta,timings
    sync();timings['output_s']=time.perf_counter()-tick
    return rho,meta,timings

def cleanup():
    import cupy as cp
    cp.cuda.Stream.null.use();sync();gc.collect()
    cp.get_default_memory_pool().free_all_blocks();cp.get_default_pinned_memory_pool().free_all_blocks()
