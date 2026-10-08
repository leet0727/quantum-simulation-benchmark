# Two-GPU setup on this machine

## Current campaign

The active study is `balanced_simulation_study`, launched in tmux session
`campaign-balanced`. See [BALANCED_STUDY.md](BALANCED_STUDY.md) for its six-stage
schedule and resume instructions. The broader campaigns described below are
historical presets; their outputs are preserved.

```bash
.venv/bin/python scripts/monitor_study.py --watch
.venv/bin/python scripts/show_run.py 1400 --config configs/balanced_main_n10.json
tmux attach -t campaign-balanced
```

The monitor is read-only; Ctrl-C exits it without stopping the campaign.

Hardware: two NVIDIA A100X GPUs (80 GiB each), CUDA toolkit 12.3,
128 available CPU threads, and approximately 754 GiB host RAM.

Use the dedicated Python 3.11 environment from the repository root:

```bash
source .venv/bin/activate
python run_sim.py --config configs/smoke_two_gpu.json --plan
python run_sim.py --config configs/smoke_two_gpu.json
```

The smoke test covers all eight method families with small three-qubit
chain/grid circuits, a pilot, two GPU workers and two CPU workers.
Only its disk reserve is reduced to 0.1 GiB because its outputs are tiny.
Use a fresh output directory if repeating it with changed code or settings.

The full campaign configuration uses GPUs 0 and 1, four CPU workers,
eight threads per worker, a 64 GiB host-RAM guard for every worker,
and a 72 GiB device memory guard per GPU worker.
A single launcher starts all six workers. The GPUs run independent circuit
jobs; their memory is not combined for one simulation.

```bash
python run_sim.py --config configs/two_gpu_64gib.json --plan
python run_sim.py --config configs/two_gpu_64gib.json
```

The full campaign saves results to the NAS at
`/mnt/main/tklee/quantum-simulation-benchmark/results/two_gpu_campaign_64gib`.
The mount was verified writable with approximately 26 TB available.
Its 20 GiB free-space guard and 48 GiB reference cache are retained. A 14-qubit reference matrix
alone occupies 4 GiB; the disk guard checks free space but does not reserve it.
The full schedule contains 18,720 campaign runs and 96 pilot runs.
A successful pilot automatically starts the campaign.

Monitor and resume:

```bash
python run_sim.py --status /mnt/main/tklee/quantum-simulation-benchmark/results/two_gpu_campaign_64gib
python run_sim.py --config configs/two_gpu_64gib.json --resume
python -m simbench.report /mnt/main/tklee/quantum-simulation-benchmark/results/two_gpu_campaign_64gib --plots
```

Stop with Ctrl-C or SIGTERM to the launcher. Resume keeps terminal results
and retries interrupted or missing jobs.

To rebuild the pinned GPU Aer version, activate the environment first:

```bash
python scripts/install_gpu_aer.py --jobs 8 --cuda-arch 8.0
```

Keep `.build/python` in place: the virtual environment uses that interpreter.
See `GPU_INSTALL.md`, `CONFIGURATION.md`, and `METHODS.md` for details.

## Validation on 2026-10-08

GPU Aer 0.17.2 built successfully from the pinned source with CUDA 12.3
and architecture 8.0. CuPy linear algebra passed on both GPUs. The runtime
uses CuPy 14.2.0, cuQuantum 26.9.0, Qiskit 2.5.2 and NumPy 2.4.6.
All five automated tests passed, including subprocess supervision.
The two-GPU smoke test completed 24 pilot and 24 main smoke records,
all with successful execution, covering all eight method families.
Both physical GPUs and both CPU workers were used. Small-shot trajectories
can miss the accuracy threshold; execution success does not imply accuracy.
The full campaign was launched by the user and stopped cleanly before changing worker settings. See `CAMPAIGN_DIAGNOSIS.md`.

Build-only Conan was removed after compiling because Conan 1 requires
older urllib3 than IBM Runtime permits. `pip check` passes in the runtime
environment. The existing wheel is retained in `.build/aer-0.17.2/dist`.
If rebuilding with the helper, remove Conan afterwards and reinstall the
project extras to restore compatible runtime dependencies:

```bash
python -m pip uninstall -y conan
python -m pip install -e '.[gpu,plots]'
python -m pip check
```

The corrected broad campaign subsequently stopped at 938 terminal records.
It was superseded by the balanced study, which reuses compatible successes with
explicit provenance. The former `campaign-fixed` session is no longer the active launcher.
