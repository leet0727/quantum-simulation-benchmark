import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np

def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, default=lambda x: x.item() if isinstance(x,np.generic) else str(x))+'\n')
    temp.replace(path)

def synchronize():
    if 'cupy' in sys.modules:
        sys.modules['cupy'].cuda.get_current_stream().synchronize()

def observables(n):
    h=n//2
    words=[]
    for p in 'XYZ':
        for q in (0,h):
            word=['I']*n; word[q]=p; words.append(''.join(word))
    for a,b,p in [(h-1,h,'Z'),(h-1,h,'X'),(h-1,h,'Y'),(0,n-1,'Z')]:
        word=['I']*n; word[a]=p; word[b]=p; words.append(''.join(word))
    words += ['X'*n,'Z'*n]
    return list(dict.fromkeys(words))

def pauli_expectations(rho, words):
    n=int(round(np.log2(len(rho)))); columns=np.arange(2**n)
    result=[]
    for word in words:
        rows=columns.copy(); phases=np.ones(len(columns),complex)
        for q,p in enumerate(word):
            bit=(columns>>q)&1
            if p in 'XY': rows ^= 1<<q
            if p=='Y': phases *= 1j*(1-2*bit)
            if p=='Z': phases *= 1-2*bit
        result.append(float(np.sum(phases*rho[columns,rows]).real))
    return np.array(result)

PAULI = [np.eye(2,dtype=complex),np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),np.diag([1.,-1.]).astype(complex)]
