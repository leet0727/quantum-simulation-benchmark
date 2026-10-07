"""Rebuild portable reports from per-run records; never infer missing accuracy."""
import csv,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path
from .common import atomic_json

def collect(out,config):
    rows=[]
    for path in sorted((out/'cases').glob('*/*__r*.json')):
        if path.name.endswith('.partial.json'):continue
        r=json.loads(path.read_text());j=r.get('job',{});s=j.get('spec',r.get('spec',{}));c=j.get('config',r.get('config',{}))
        m=dict(r.get('metrics',{}));reference=path.parent/'reference.json'
        if reference.exists() and 'state_fidelity' in m:
            rm=json.loads(reference.read_text())['metrics']
            m['fidelity_score_abs_error']=abs(m['state_fidelity']-rm['state_fidelity'])
            m['observable_vector_rmse']=math.sqrt(statistics.mean((a-b)**2 for a,b in zip(m['observables'],rm['observables'])))
        row=dict(case=path.parent.name,family=s.get('family'),qubits=s.get('n'),layers=s.get('depth'),seed=s.get('seed'),
            method=c.get('id'),repeat=j.get('repeat'),status=r.get('status'),gpu_assignment=r.get('gpu_assignment'),
            worker_wall_s=r.get('worker_wall_s'),supervisor_wall_s=r.get('supervisor_wall_s'),worker_cpu_s=r.get('worker_cpu_s'),
            peak_rss_gib=r.get('sampled_peak_rss_bytes',0)/1024**3,
            gpu_peak_gib=(r['gpu_peak_memory_bytes']/1024**3 if r.get('gpu_peak_memory_bytes') is not None else None),
            error=r.get('error'),**r.get('timings',{}))
        for k in ['state_fidelity','observable_rmse','fidelity_score_abs_error','observable_vector_rmse','full_density_frobenius_error','negative_eigenvalue_mass','projection_frobenius_distance']:row[k]=m.get(k)
        valid=r.get('status')=='success' and all(row.get(k) is not None for k in config['accuracy'])
        row['meets_joint_accuracy']=all(row[k]<=v for k,v in config['accuracy'].items()) if valid else None
        rows.append(row)
    return rows

def write(out,config):
    rows=collect(out,config)
    summary=dict(total_records=len(rows),statuses=dict(Counter(r['status'] for r in rows)),groups=[])
    groups=defaultdict(list)
    for row in rows:groups[(row['family'],row['qubits'],row['method'])].append(row)
    for (family,n,method),group in sorted(groups.items(),key=lambda x:str(x[0])):
        success=[r for r in group if r['status']=='success'];scored=[r for r in success if r['meets_joint_accuracy'] is not None]
        summary['groups'].append(dict(family=family,qubits=n,method=method,runs=len(group),success=len(success),
            scored=len(scored),joint_pass=sum(r['meets_joint_accuracy'] for r in scored),
            median_simulation_s=statistics.median(r['simulation_s'] for r in success) if success else None))
    atomic_json(out/'summary.json',summary)
    if rows:
        columns=list(dict.fromkeys(k for r in rows for k in r))
        temp=out/'results.csv.tmp'
        with temp.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
        temp.replace(out/'results.csv')
    return summary

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--plots',action='store_true');a=p.parse_args()
    config=json.loads((a.output/'manifest.json').read_text())['config'];summary=write(a.output,config)
    print(json.dumps(summary,indent=2))
    if a.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rows=collect(a.output,config);families=sorted({r['family'] for r in rows});dest=a.output/'plots';dest.mkdir(exist_ok=True)
        for family in families:
            fig,axes=plt.subplots(1,2,figsize=(12,5))
            subset=[r for r in rows if r['family']==family and r['status']=='success']
            methods=sorted({r['method'] for r in subset})
            for method in methods:
                group=[r for r in subset if r['method']==method]
                for ax,key in zip(axes,['fidelity_score_abs_error','observable_vector_rmse']):
                    selected=[r for r in group if r.get(key) is not None]
                    ax.scatter([r['simulation_s'] for r in selected],[max(r[key],1e-16) for r in selected],label=method,s=20,alpha=.7)
                    ax.set(xscale='log',yscale='log',xlabel='Simulation seconds',ylabel=key.replace('_',' '))
                    ax.axhline(config['accuracy'][key],ls='--',color='grey',lw=.8)
            fig.suptitle(family+' (all tested sizes; individual runs)');axes[-1].legend(fontsize=6,bbox_to_anchor=(1.02,1),loc='upper left')
            fig.tight_layout();fig.savefig(dest/(family+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
