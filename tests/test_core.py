import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from simbench.config import load,expand_methods
from simbench.schedule import expand
ROOT=Path(__file__).resolve().parents[1]

class ConfigTests(unittest.TestCase):
    def test_original_campaign_and_order(self):
        c=load(ROOT/'input.json');jobs,methods=expand(c)
        self.assertEqual(len(methods),52);self.assertEqual(len(jobs),18720)
        self.assertEqual(sum(j['phase']==[0] for j in jobs),320)
        widths=[j['spec']['n'] for j in jobs];self.assertEqual(widths,sorted(widths))
        seen=set()
        for j in jobs:
            if j['case'] not in seen:self.assertEqual(j['config']['id'],'dense_gpu');seen.add(j['case'])
        self.assertEqual(jobs,expand(c)[0])
    def test_validation_and_custom_order(self):
        raw=json.loads((ROOT/'configs/smoke_cpu.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.json'
            for patch in [dict(unknown=5),dict(execution=dict(gpu_ids=[0,0])),dict(timeout_s=-1),dict(pilot=dict(qubits=0))]:
                p.write_text(json.dumps(raw|patch))
                with self.assertRaises(ValueError):load(p)
            raw['scheduling']=dict(mode='custom',order_by=['layers','qubits','budget'],preliminary=dict(enabled=False))
            raw['circuits']['layers']=[2,1];p.write_text(json.dumps(raw));c=load(p)
            depths=[j['spec']['depth'] for j in expand(c)[0]];self.assertEqual(depths,sorted(depths))
    def test_no_launch_on_n_or_plan(self):
        raw=json.loads((ROOT/'configs/smoke_cpu.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            raw['output_dir']=str(Path(d)/'no-output');p=Path(d)/'c.json';p.write_text(json.dumps(raw))
            for flags in [[],['--plan']]:
                r=subprocess.run([sys.executable,str(ROOT/'run_sim.py'),'--config',str(p),*flags],input='N\n',text=True,capture_output=True)
                self.assertEqual(r.returncode,0,r.stderr);self.assertFalse(Path(raw['output_dir']).exists())

class NumericalTests(unittest.TestCase):
    def test_small_noisy_nonlocal_circuit(self):
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator,SuperOp,Kraus,Statevector
        from qiskit_aer.noise import amplitude_damping_error
        from simbench.engine import evaluate,scores
        from simbench.kernels import aer
        qc=QuantumCircuit(3);qc.h(0);qc.cx(0,2);qc.ry(.4,1);qc.cx(2,1)
        stream=[];kraus=[]
        for inst in qc.data:
            sites=tuple(qc.find_bit(q).index for q in inst.qubits);ch=SuperOp(Operator(inst.operation)).data
            stream.append((sites,ch));kraus.append((sites,Kraus(SuperOp(ch)).data))
        for site in range(3):
            ch=SuperOp(amplitude_damping_error(.12)).data;stream.append(((site,),ch));kraus.append(((site,),Kraus(SuperOp(ch)).data))
        v=dict(n=3,support=[2,0,1],stream=stream,kraus=kraus)
        reference,_=aer(v,dict(aer_method='density_matrix',gpu=False))
        configs=[dict(family='dense'),dict(family='density_mps',chi=16),dict(family='low_rank',rank=8),
                 dict(family='sparse_pauli',terms=64,cutoff=0),dict(family='tensor_network')]
        for cfg in configs:
            with self.subTest(method=cfg):
                rho,_,_=evaluate(v,cfg,lambda *a:None);self.assertLess(np.linalg.norm(rho-reference),1e-10)
        # An SVD convergence failure must recover without changing the noisy output.
        from unittest.mock import patch
        svd=np.linalg.svd
        calls=[0]
        def fail_once(*args,**kwargs):
            calls[0]+=1
            if calls[0]==1:raise np.linalg.LinAlgError('SVD did not converge')
            return svd(*args,**kwargs)
        with patch('numpy.linalg.svd',side_effect=fail_once):
            rho,meta,_=evaluate(v,dict(family='density_mps',chi=16),lambda *a:None)
        self.assertLess(np.linalg.norm(rho-reference),1e-10)
        self.assertEqual(meta['svd_fallback_count'],1)
        # Projected-fidelity result agrees with an explicit Euclidean PSD/trace-one projection.
        rng=np.random.default_rng(13);a=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8));rho=a+a.conj().T
        psi=Statevector.from_instruction(qc).data;vals,vec=np.linalg.eigh(rho);u=vals[::-1];offset=(np.cumsum(u)-1)/np.arange(1,9)
        theta=offset[np.where(u>offset)[0][-1]];psd=(vec*np.maximum(vals-theta,0))@vec.conj().T
        actual=scores(rho,psi,['ZZZ','XII'],eigensolver_gpu=False)
        self.assertAlmostEqual(actual['state_fidelity'],np.vdot(psi,psd@psi).real,places=12)

if __name__=='__main__':unittest.main()
