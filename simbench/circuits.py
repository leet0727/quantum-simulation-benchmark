import numpy as np

def make_parent(family,n,depth,seed):
    from qiskit import QuantumCircuit
    rng=np.random.default_rng(seed); qc=QuantumCircuit(n,name=f'{family}_{n}_{depth}_{seed}')
    for q in range(n): qc.ry(float(rng.uniform(.15,1.3)),q)
    for layer in range(depth):
        if family=='clifford':
            for q in range(n):
                qc.h(q)
                if (q+layer)%2: qc.s(q)
            pairs=[(q,q+1) for q in range(layer%2,n-1,2)]
        elif family in ('chain','brickwork'):
            pairs=[(q,q+1) for q in range(n-1)] if family=='chain' else [(q,q+1) for q in range(layer%2,n-1,2)]
        elif family=='clusters':
            h=n//2
            pairs=[(a,b) for lo,hi in [(0,h),(h,n)] for a in range(lo,hi) for b in range(a+1,hi)]
            if layer==depth//2: pairs.append((h-1,h))
        elif family=='all_to_all':
            order=list(map(int,rng.permutation(n))); pairs=list(zip(order[::2],order[1::2]))
        elif family=='qaoa': pairs=[(q,(q+1)%n) for q in range(n)]
        else: raise ValueError(family)
        for a,b in pairs:
            if family=='qaoa': qc.rzz(float(rng.uniform(.15,.8)),a,b)
            else: qc.cx(a,b)
        if family!='clifford':
            for q in range(n): qc.ry(float(rng.uniform(-1,1)),q); qc.rz(float(rng.uniform(-1,1)),q)
    # The Clifford family has a genuinely Clifford initial state too.
    if family=='clifford':
        from qiskit.circuit.library import HGate
        for q in range(n): qc.data[q]=qc.data[q].replace(operation=HGate())
    return qc


from .config import FAMILIES

def parent_circuit(spec):
    from qiskit import QuantumCircuit
    f,n,d,seed=(spec[k] for k in ['family','n','depth','seed'])
    if f in FAMILIES[:6]: return make_parent(f,n,d,seed)
    rng=np.random.default_rng(seed);qc=QuantumCircuit(n)
    for q in range(n): qc.ry(float(rng.uniform(.15,1.3)),q)
    for layer in range(d):
        if f=='grid2d':
            cols=int(np.ceil(np.sqrt(n)))
            edges=[(q,q+1) for q in range(n-1) if q//cols==(q+1)//cols]
            edges += [(q,q+cols) for q in range(n-cols)]
            for a,b in edges: qc.cx(a,b)
        elif f=='qft':
            # Repeated QFT-style blocks with seeded local rotations between them.
            # Not repeated bare QFTs, which would have a short algebraic period.
            for a in range(n):
                qc.h(a)
                for b in range(a+1,n): qc.cp(np.pi/(2**(b-a)),b,a)
        for q in range(n):
            qc.ry(float(rng.uniform(-1,1)),q);qc.rz(float(rng.uniform(-1,1)),q)
    return qc
