"""Self-contained calibrated channel extraction; no qtpu/SBCC imports."""
from qiskit.quantum_info import SuperOp
from qiskit_aer.noise import NoiseModel,QuantumError

class BackendModel:
    def __init__(self,name):
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        self.backend=FakeMarrakesh()
        self.snapshot=self.backend.properties().to_dict()
        errors=NoiseModel.from_backend(self.backend).to_dict()['errors'] if name!='none' else []
        self.channels={}
        for entry in errors:
            if entry['type']!='qerror':continue
            channel=SuperOp(QuantumError.from_dict(entry)).data
            for op in entry['operations']:
                for sites in entry.get('gate_qubits',[None]):
                    self.channels[op,None if sites is None else tuple(sites)]=channel

def graph_features(n,pairs):
    edges={tuple(sorted(p)) for p in pairs};degree=[0]*n
    for a,b in edges:degree[a]+=1;degree[b]+=1
    return dict(interaction_edges=len(edges),interaction_density=2*len(edges)/(n*(n-1)) if n>1 else 0,
        max_interaction_degree=max(degree,default=0),mean_site_distance=sum(abs(a-b) for a,b in pairs)/len(pairs) if pairs else 0,
        long_range_gate_fraction=sum(abs(a-b)>1 for a,b in pairs)/len(pairs) if pairs else 0)
