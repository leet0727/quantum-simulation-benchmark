# GPU installation

The validated environment uses **Aer 0.17.2 built from source with CUDA enabled**, distributed under the package name `qiskit-aer`. A CPU-only wheel of the same name/version does not provide equivalent GPU capability. At packaging time the public `qiskit-aer-gpu` wheel index offered releases only through 0.15.1. Do not substitute that older wheel silently when reproducing this benchmark.

The `gpu` extra installs CuPy, cuQuantum and NVML support. Aer is built separately:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[gpu,plots]'
python scripts/install_gpu_aer.py --plan
python scripts/install_gpu_aer.py --jobs 2
python run_sim.py --config input.json --plan
python run_sim.py --config input.json
```

Before building, provide a compatible NVIDIA driver, **CUDA 12 toolkit with `nvcc` on PATH**, a CUDA-compatible C++ compiler, Git and OpenBLAS development libraries. The helper does not install system packages or use sudo. On Debian/Ubuntu the compiler/OpenBLAS prerequisites typically come from `build-essential`, `libopenblas-dev` and `git`; obtain the CUDA toolkit appropriate for the host from NVIDIA or the environment administrator. CUDA runtime Python wheels alone do not supply the full compiler toolkit.

The helper fetches exactly commit `51c679814c3a292d0d7c59bb39976bd6ff91f60e` from Qiskit Aer, installs its Python build tools, builds a wheel, replaces any existing Aer distribution in the **active Python environment**, and verifies that Aer advertises GPU support. Use a dedicated environment. A failed build retains its directory for inspection. `--jobs` limits compiler parallelism and memory pressure; compilation can take substantially longer than a small benchmark run.

Optional arguments:

```bash
python scripts/install_gpu_aer.py --jobs 4 --cuda-arch 8.6
python scripts/install_gpu_aer.py --blas-lib-path /path/to/openblas/lib
```

`--cuda-arch Auto` is the default. Explicit `8.6` is merely an example for a matching device; set the architecture for the destination GPU. Do not copy this example blindly onto a different GPU. An existing compatible GPU-enabled Aer 0.17.2 installation can be retained instead of running the helper.

Aer is compiled using `AER_THRUST_BACKEND=CUDA`; its optional cuQuantum integration is disabled, matching the numerical path used here. The separate projection-DMRG method uses the installed **cuQuantum Python** package. These are independent uses of the GPU stack. This avoids coupling Aer to an older bundled cuQuantum version. The build procedure follows [Aer's GPU source-build instructions](https://qiskit.github.io/qiskit-aer/getting_started.html#building-with-gpu-support); the pinned [setup file](https://github.com/Qiskit/qiskit-aer/blob/0.17.2/setup.py) and [build requirements](https://github.com/Qiskit/qiskit-aer/blob/0.17.2/pyproject.toml) specify its package/build controls.

The numerical GPU paths were tested on the existing source-built environment. The helper's command construction and pinned revision were verified, but a fresh full CUDA compilation on the destination hardware has not been performed. The small pilot remains the destination-side check before automatic campaign progression.
