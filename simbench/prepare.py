import time, pickle, hashlib
import numpy as np
from . import common
from .circuits import parent_circuit
MODEL=None

def prepare(spec,folder,progress,config):
    global MODEL
    from qiskit import transpile,qpy
    from qiskit.transpiler import Target
    from qiskit.quantum_info import Statevector,SuperOp,Operator,Kraus,Clifford
    from .backend import BackendModel
    from .backend import graph_features
    started=time.perf_counter();times={};progress('backend_preparation',0,1,{})
    if MODEL is None: MODEL=BackendModel(config['noise']['backend'])
    times['backend_model_s']=time.perf_counter()-started
    n=spec['n'];adj={}
    for a,b in MODEL.backend.coupling_map.get_edges():
        adj.setdefault(a,set()).add(b);adj.setdefault(b,set()).add(a)
    physical=[min(adj)];todo=[physical[0]]
    while len(physical)<n and todo:
        q=todo.pop(0)
        for other in sorted(adj[q]):
            if other not in physical:
                physical.append(other);todo.append(other)
                if len(physical)==n:break
    if len(physical)!=n:raise ValueError('cannot find a connected backend subgraph')
    mapping={q:i for i,q in enumerate(physical)};target=Target(num_qubits=n)
    for name in MODEL.backend.target.operation_names:
        instruction=MODEL.backend.target.operation_from_name(name);properties={}
        if isinstance(instruction,type):continue
        for sites,prop in MODEL.backend.target[name].items():
            if sites is not None and all(q in mapping for q in sites):properties[tuple(mapping[q] for q in sites)]=prop
        if properties:target.add_instruction(instruction,properties,name=name)
    tick=time.perf_counter();parent=parent_circuit(spec)
    psi=Statevector.from_instruction(parent).data
    with (folder/'parent.qpy').open('wb') as f:qpy.dump(parent,f)
    measured=parent.copy();measured.measure_all();progress('transpilation',0,1,{})
    routed=transpile(measured,target=target,initial_layout=list(range(n)),optimization_level=config['noise']['optimization_level'],seed_transpiler=spec['seed'])
    if routed.num_qubits!=n:raise ValueError('compiled width changed')
    support=[None]*n
    for inst in routed.data:
        if inst.operation.name=='measure':support[routed.find_bit(inst.clbits[0]).index]=routed.find_bit(inst.qubits[0]).index
    if sorted(support)!=list(range(n)):raise ValueError('invalid logical output mapping')
    compact=routed.remove_final_measurements(inplace=False)
    with (folder/'compiled.qpy').open('wb') as f:qpy.dump(compact,f)
    times['generation_transpile_ideal_s']=time.perf_counter()-tick
    # Independent, inexpensive ideal-state mapping check at every size.
    routed_psi=Statevector.from_instruction(compact).data
    axes=[n-1-q for q in reversed(support)]
    routed_psi=routed_psi.reshape((2,)*n).transpose(axes).reshape(-1)
    mapping_error=abs(1-abs(np.vdot(psi,routed_psi))**2)
    if mapping_error>1e-8:raise ValueError(f'ideal logical mapping failed: {mapping_error}')
    tick=time.perf_counter();stream=[];ideal=[];kraus=[];infidelities=[];pairs=[];missing=[]
    progress('channel_extraction',0,len(compact.data),{})
    for i,inst in enumerate(compact.data):
        if inst.operation.name in ['barrier','delay']:continue
        sites=tuple(compact.find_bit(q).index for q in inst.qubits)
        u=SuperOp(Operator(inst.operation)).data
        phys=tuple(physical[q] for q in sites)
        noise=MODEL.channels.get((inst.operation.name,phys),MODEL.channels.get((inst.operation.name,None)))
        ch=u if noise is None else noise@u
        ks=Kraus(SuperOp(ch)).data
        if np.linalg.norm(sum(k.conj().T@k for k in ks)-np.eye(2**len(sites)))>1e-7:raise ValueError('channel is not trace preserving')
        if noise is None:missing.append((inst.operation.name,phys))
        dim=2**len(sites);infidelities.append(0. if noise is None else float(1-(np.trace(noise).real+dim)/(dim*(dim+1))))
        stream.append((sites,ch));ideal.append((sites,u));kraus.append((sites,ks))
        if len(sites)==2:pairs.append(sites)
        progress('channel_extraction',i+1,len(compact.data),{})
    times['channels_s']=time.perf_counter()-tick
    v=dict(n=n,support=support,stream=stream,ideal_stream=ideal,kraus=kraus,
           physical=physical,output_count=n,sign_count=0)
    features=dict(compiled_width=n,compiled_depth=compact.depth(),compiled_gates=len(stream),
        compiled_two_qubit=len(pairs),parent_depth=parent.depth(),parent_gates=len(parent.data),
        **graph_features(n,pairs),sum_gate_infidelity=float(sum(infidelities)),
        max_channel_kraus_rank=max(len(k) for _,k in kraus),physical=physical,
        max_gate_crossings=max(sum(min(a,b)<=cut<max(a,b) for a,b in pairs) for cut in range(n-1)),
        gates_without_noise=sorted(set(str(x) for x in missing)),ideal_mapping_error=mapping_error)
    words=common.observables(n)
    # Pure-state expectations avoid forming an ideal 4^n matrix.
    from qiskit.quantum_info import SparsePauliOp
    ideal_obs=[float(Statevector(psi).expectation_value(SparsePauliOp(w[::-1])).real) for w in words]
    with (folder/'input.pkl').open('wb') as f:pickle.dump(dict(variant=v,psi=psi,spec=spec,words=words,ideal_obs=ideal_obs),f,protocol=5)
    result=dict(status='success',spec=spec,features=features,timings=times,prepare_s=time.perf_counter()-started,
                input_sha256=hashlib.sha256((folder/'input.pkl').read_bytes()).hexdigest())
    common.atomic_json(folder/'prepared.json',result)
    common.atomic_json(folder/'backend_snapshot.json',MODEL.snapshot)
    return result
