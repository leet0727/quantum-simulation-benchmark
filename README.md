# Quantum simulation benchmark

A standalone runner for comparing noisy **whole-circuit** simulation methods across circuit structures and approximation budgets. It contains the numerical implementations used in the earlier simulation-method benchmark. It does not perform circuit cutting, SBCC search, or candidate screening, and does not require those repositories. The original campaign is not resumed by installing or previewing this project.

## Install and first run

Python 3.11+ on Linux. Use a fresh virtual environment. Choose **one** Aer installation: CPU or GPU (the two wheels provide the same Python module).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[cpu,plots]'
python run_sim.py --config configs/smoke_cpu.json
```

The runner prints the expanded settings, counts and initial schedule, then asks `Launch this experiment? [Y/N]`. `N` cancels without creating results or initializing simulators. The smoke configuration runs two small circuit families with two CPU workers. It is a functionality check, not a meaningful performance comparison.

For NVIDIA GPUs, use a separate environment:

```bash
python -m pip install -e '.[gpu,plots]'
python scripts/install_gpu_aer.py --jobs 2
python run_sim.py --config input.json --plan
python run_sim.py --config input.json
```

The GPU dependencies target CUDA 12 on Linux. The build helper also needs the CUDA toolkit (`nvcc`), a compatible C++ compiler, Git and OpenBLAS development libraries. See [GPU_INSTALL.md](GPU_INSTALL.md) before installing. The `gpu` extra installs CuPy/cuQuantum; the helper separately builds GPU-enabled Aer 0.17.2 from the pinned source revision used here. Projection DMRG uses cuQuantum's experimental MPS API and needs the specified recent cuQuantum release. GPU availability is validated when a worker starts and methods run in the pilot. A GPU wheel is not a guarantee that a particular host supports every method.

`--plan` only validates and prints settings; it requires no numerical packages. `--yes` explicitly bypasses the prompt for a batch scheduler. Nothing starts on a timer.

## Configuration and scheduling

The current two-GPU study uses a balanced main comparison and a separate high-budget subset:
**12,600 evaluations across six stages**, with 300/900-second evaluation deadlines.
See [BALANCED_STUDY.md](BALANCED_STUDY.md) for fixed budgets, provenance, launcher and monitor commands.
The balanced and local GPU presets use machine-specific NAS output paths; edit them for another environment.

Edit `input.json`; a complete option guide is in [CONFIGURATION.md](CONFIGURATION.md). The provided full configuration contains:

- Eight families: chain, brickwork, Clifford, clusters, random long-range pairs, ring QAOA, grid, and QFT-style blocks with random rotations.
- 10, 12, 14 qubits; 16, 32, 64 layers; five seeds: 360 circuit cases.
- 52 method/device/budget settings: **18,720 scheduled runs**, plus a 96-run small pilot.
- Two-hour per-run deadline; separate preparation and worker initialization deadlines.
- Density-MPS and projection-DMRG bond caps through 1024, low-rank caps through 1024, trajectories through 65,536 shots, sparse-Pauli caps through 524,288 terms.

A layer is a family-specific circuit block, not compiled circuit depth. Both compiled depth and actual gate counts are recorded.

The default starts with 320 preliminary runs: the smallest width and layer count, eight inexpensive method settings, all families and seeds. Then it completes 10-qubit work before 12- and 14-qubit work, using cheaper approximation budgets first within each size. Exact references precede approximations in this default schedule. The pilot uses every selected family and the lowest setting of every method/device combination. It checks execution and independently cross-checks exact outputs against Aer; it does **not** require approximate methods to meet an accuracy threshold. A pilot failure stops automatic progression. A successful pilot proceeds directly to the campaign without another prompt.

Choose `qubits_first`, `budgets_first`, `case_first`, or `custom`; reorder methods, set custom sort keys, or disable preliminary runs. Scheduling phases have barriers: later phases wait for earlier ones to finish. Jobs for the same case run serially so preparation and reference files cannot race. This can leave a worker idle when only one case is eligible.

### One launcher, multiple workers

You launch **one** `run_sim.py`. With `execution.mode: "parallel"`, it starts persistent worker processes: one per configured GPU and the requested number of CPU workers. See [configs/parallel_gpu.json](configs/parallel_gpu.json), and edit its GPU IDs and thread count for the destination machine. Its example assumes two GPUs; it does not automatically discover or reserve hardware.

CPU jobs use CPU workers when configured, otherwise they run on the GPU worker's CPU. GPU jobs never fall back silently to CPU. Set `gpu_ids` to physical ordinals or GPU UUIDs; each worker sees its assigned GPU as local device 0. Set CPU worker count and threads so their product leaves room for GPU host work. Host memory limits are **per worker**; budget their sum for the machine.

`isolated` uses one worker, useful for comparing per-method latency. `parallel` increases throughput; workers share CPU, host memory, storage and sometimes other hardware resources, so latency and device-wide telemetry can change. The output-directory lock prevents accidentally launching a second controller against the same results. There is no machine-wide lock across different output directories.

## Progress, results, stop and resume

A terminal shows a live progress bar, completed count, success/timeout counts, elapsed time and a rate-based ETA. Redirected logs show a text bar periodically. ETA reflects completed jobs, not a reliable prediction for exponentially harder later sizes. Current worker phases, elapsed times and sampled memory are in `progress.json`.

```bash
python run_sim.py --status results/campaign
watch -n 2 'python run_sim.py --status results/campaign'
python -m simbench.report results/campaign --plots
```

`--status` displays the saved snapshot; check `updated_at` and the PID if the machine or controller crashed. It does not claim a stale PID is still running.

Use Ctrl-C or send SIGTERM to the controller. Workers are stopped and interrupted jobs can be retried:

```bash
python run_sim.py --config input.json --resume
```

Resume requires the same scientific configuration and source hash. Output location, worker assignments and telemetry settings may change to allow migration. Completed success, error, timeout and resource-excluded records are retained; interrupted/missing jobs retry. To rerun terminal failures with different limits or corrected code, use a new output directory. Never modify an active output directory.

Files include:

| File | Contents |
| --- | --- |
| `manifest.json`, `launches.jsonl` | Expanded settings, source fingerprint, package versions and hardware assignments |
| `progress.json` | Controller progress and active worker phases |
| `results.csv`, `summary.json` | Per-run results and summaries by family, size and method; updated every 20 completions and on exit |
| `cases/<case>/prepared.json`, `preparation.json` | Circuit features, routing/mapping check, setup times and supervised preparation resources |
| `parent.qpy`, `compiled.qpy`, `backend_snapshot.json` | Reproducible circuit and backend artifacts |
| `<method>__rN.json` | Metrics, diagnostics, timings, outcome, memory, CPU time and worker assignment |
| `<method>__rN.resources.jsonl` | Sampled process RSS, CPU time, I/O, context switches and available GPU telemetry |
| `workers/<lane>/startups.jsonl`, `worker.log` | Cold initialization timing and process diagnostics |
| `pilot/` | Separate small validation results |

Setup, simulation, output materialization (where exposed), scoring, reference I/O and cleanup/supervisor time are distinguished. Aer and projection DMRG expose combined evolution/output time. Workers stay warm between jobs; cold initialization is recorded separately. Backend/channel preparation is cached once per case. Clocks, power and utilization depend on driver support; unsupported counters are recorded as missing with reasons. **GPU kernel cycle counts are not collected.** Device telemetry is sampled and can miss short peaks or include unrelated device activity.

## Scientific interpretation and limits

See [METHODS.md](METHODS.md) for implementations, noise and score definitions. This study measures approximation error in two scores relative to an exact noisy reference, as well as full density-matrix Frobenius error when the reference matrix remains cached. It does not by itself establish preservation of candidate rankings in a cutting pipeline.

All methods return a full logical output density matrix for common scoring. Its complex128 storage alone is `16 × 4^n` bytes: **4 GiB at 14 qubits**. Scoring uses a full eigendecomposition and may dominate runtime or memory. Larger qubit settings need correspondingly larger output, workspace and host/GPU limits; adjusting a cap does not make exponential storage disappear. These are end-to-end score benchmarks, not observable-only simulations that avoid dense output.

The full exact-reference matrix cache is bounded; permanent scalar scores and observable vectors allow accuracy comparisons after matrix eviction. Full density errors are then unavailable, not zero. Failed or missing references remain unscored in the report. Do not compare successful-run medians without also reporting timeout/exclusion counts.

## Tests and publication

```bash
python -m unittest discover -s tests -v
python run_sim.py --config configs/smoke_cpu.json
```

Local validation is recorded in [VALIDATION.md](VALIDATION.md). No full campaign results are bundled. Code provenance and third-party notices are in [NOTICE.md](NOTICE.md) and [PROVENANCE.json](PROVENANCE.json).

To publish after creating an empty **public** repository on GitHub (without an initial README):

```bash
git remote add origin https://github.com/YOUR_USERNAME/quantum-simulation-benchmark.git
git push -u origin main
```

Or, after `gh auth login`, create and push it with:

```bash
gh repo create quantum-simulation-benchmark --public --source=. --remote=origin --push
```
