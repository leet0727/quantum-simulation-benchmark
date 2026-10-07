#!/usr/bin/env python3
"""Build the same Aer revision used in validation, with CUDA support.

Requires a CUDA 12 toolkit (nvcc), a compatible C++ compiler, Git and OpenBLAS
headers/libraries. Run in the intended Python environment. No sudo is invoked.
"""
import argparse,os,shlex,shutil,subprocess,sys
from pathlib import Path

REVISION='51c679814c3a292d0d7c59bb39976bd6ff91f60e'  # Aer 0.17.2
URL='https://github.com/Qiskit/qiskit-aer.git'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir',type=Path,default=Path('.build/aer-0.17.2'))
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--cuda-arch',default='Auto',help='Aer CUDA architecture, e.g. 8.6; default auto-detect')
    p.add_argument('--blas-lib-path',type=Path)
    p.add_argument('--plan',action='store_true')
    a=p.parse_args()
    if sys.version_info<(3,11):p.error('Python 3.11 or newer is required')
    if a.jobs<1:p.error('--jobs must be positive')
    dest=a.build_dir.expanduser().resolve()
    command=[sys.executable,'setup.py','bdist_wheel','--','-DAER_THRUST_BACKEND=CUDA',
             '-DAER_ENABLE_CUQUANTUM=OFF',f'-DAER_CUDA_ARCH={a.cuda_arch}']
    if a.blas_lib_path:command.append('-DAER_BLAS_LIB_PATH='+str(a.blas_lib_path.expanduser().resolve()))
    print(f'Build {URL} at {REVISION} in {dest}\nParallel compiler jobs: {a.jobs}')
    print(shlex.join(command))
    if a.plan:return 0
    for tool in ['git','nvcc','c++']:
        if not shutil.which(tool):p.error(f'{tool} is required on PATH; install the CUDA toolkit/compiler first')
    def run(cmd,**kwargs):subprocess.run(cmd,check=True,**kwargs)
    if dest.exists():
        if not (dest/'.git').is_dir():p.error('build directory exists but is not the expected Git checkout')
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=dest,text=True).strip()
        if head!=REVISION:p.error('build checkout has a different revision; choose a new --build-dir')
        dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=dest,text=True).strip()
        if dirty:p.error('build checkout has tracked changes; choose a new --build-dir')
    else:
        dest.mkdir(parents=True)
        run(['git','init'],cwd=dest)
        run(['git','fetch','--depth','1',URL,REVISION],cwd=dest)
        run(['git','checkout','--detach','FETCH_HEAD'],cwd=dest)
    run([sys.executable,'-m','pip','install','wheel','setuptools','scikit-build>=0.11',
         'cmake>=3.26,<4','ninja','pybind11>2.6','conan<2'])
    env=dict(os.environ,CMAKE_BUILD_PARALLEL_LEVEL=str(a.jobs),QISKIT_AER_PACKAGE_NAME='qiskit-aer',
             QISKIT_ADD_CUDA_REQUIREMENTS='false')
    run(command,cwd=dest,env=env)
    wheels=sorted((dest/'dist').glob('qiskit_aer-0.17.2-*.whl'),key=lambda x:x.stat().st_mtime)
    if not wheels:raise RuntimeError('build produced no matching wheel')
    # Avoid overlapping CPU/GPU distributions that both provide qiskit_aer.
    run([sys.executable,'-m','pip','uninstall','-y','qiskit-aer','qiskit-aer-gpu'])
    run([sys.executable,'-m','pip','install',str(wheels[-1])])
    run([sys.executable,'-c',"from qiskit_aer import AerSimulator; d=AerSimulator().available_devices(); print('Aer devices:', d); assert 'GPU' in d, 'GPU support not detected'"])
    print('GPU Aer installed. Run the benchmark pilot before starting large cases.')
    return 0

if __name__=='__main__':raise SystemExit(main())
