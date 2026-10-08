# Balanced simulation study presets

This replaces the broad campaign with two fixed, predefined studies. Every family receives the same settings within each qubit count. Settings are not chosen based on early accuracy results. The old campaign remains at `../two_gpu_campaign_64gib`.

## Main comparison: 10,680 scheduled runs

Eight families; 10, 12, 14 qubits; depths 16, 32, 64; seeds 22101–22105.

| Method | Budgets | Devices |
|---|---|---|
| Dense | Exact | CPU, GPU |
| Density MPS | Bond dimensions 16, 64, 256 | CPU, GPU |
| DMRG | Bond dimensions 16, 64, 256; 4 sweeps | GPU |
| Low rank | Ranks 8, 32, 128 | CPU, GPU |
| Dense trajectories | 256, 1024 trajectories | CPU, GPU |
| MPS trajectories | Bonds 16/32 at 10q; 16/64 at 12q; 16/64/128 at 14q; each crossed with 256/1024 trajectories | CPU |
| Sparse Pauli | Paired terms/cutoffs: 2048/1e-5, 8192/1e-6, 32768/1e-7 | CPU |
| Tensor network | Existing 16 GiB intermediate cap | GPU |

Each evaluation has a 300-second deadline, including scoring. Preparation and worker startup retain their separate 7200- and 300-second deadlines. Timeouts remain results indicating that a setting exceeded its evaluation budget; they are not accuracy scores. A missing exact reference must also be reported explicitly.

## High-budget scaling: 1,920 scheduled runs

All eight families and all three widths; depths 16 and 64; seeds 22101 and 22102.

| Method | Budgets | Devices |
|---|---|---|
| Exact reference | Dense | GPU |
| Density MPS | Bonds 512, 1024 | CPU, GPU |
| DMRG | Bonds 512, 1024; 4 sweeps | GPU |
| Low rank | Ranks 256, 1024 | CPU, GPU |
| Dense trajectories | 4096, 16384 trajectories | CPU, GPU |
| MPS trajectories | Bond 32/64/128 for 10/12/14q respectively; 256, 4096, 16384 trajectories | CPU |
| Sparse Pauli | Paired terms/cutoffs: 131072/1e-8, 524288/1e-9 | CPU |

Each evaluation has a 900-second deadline. No 65536-trajectory settings. Density MPS and DMRG keep their larger bond dimensions: their density representation has a different capacity ceiling from pure-state MPS trajectories.

## Execution and provenance

Stages run sequentially: main 10q, 12q, 14q, then scaling 10q, 12q, 14q. Two GPU workers, four CPU workers, eight threads per worker. Host RSS limit 64 GiB per worker; GPU device limit 72 GiB; workspace 40 GiB; exact-reference cache 48 GiB. FakeMarrakesh transpilation at optimization level 1 is unchanged. Existing guards and scoring are unchanged.

Completed successful records from the stopped campaign are reused only when the numerical job, random seed, source hash, noise, reference, resources and execution settings match, and both worker and supervisor durations fit the new deadline. Each imported record has `reuse_provenance`, and each stage has `reuse_manifest.json`. Imported timings and telemetry were measured in the original campaign, not the new launch; distinguish reused and newly measured results in timing analyses. Failures and over-budget successes are not imported. Prepared circuit inputs and exact oracles are copied alongside compatible records.

The old results, including over-budget successes, remain valuable for separate accuracy and runtime analyses. Do not silently pool their different deadlines with this study. The study still contains 12,600 evaluations plus pilots and can take days; fixed deadlines bound individual evaluations, not overall completion time.

## Commands

Attach: `tmux attach -t campaign-balanced`

Follow current-stage progress: `tail -f main/n10/campaign.log` (replace path for later stages).

Read progress: `cat main/n10/progress.json`.

From the repository, restart the sequential launcher with `.venv/bin/python scripts/run_balanced_study.py`; it resumes existing manifests and skips finished stages. Only one launcher can own the study lock. Send SIGTERM to the launcher PID to stop the active controller cleanly and prevent the next stage from starting. Do not stop only the child controller if you intend to stop the whole study.


## Live monitor and individual results

From the repository root:

```bash
.venv/bin/python scripts/monitor_study.py --watch
.venv/bin/python scripts/monitor_study.py --json
.venv/bin/python scripts/show_run.py 1400 --config configs/balanced_main_n10.json
```

The monitor refreshes every 30 seconds, with progress per stage/method, remaining
CPU/GPU evaluations, statuses, active workers, provisional accuracy/runtime
leaders and optimistic/realistic/maximum finish scenarios. Ctrl-C stops only the
monitor. Imported successes are excluded from recent throughput; identical job
IDs reused across stages are deduplicated in method statistics.

ETA scenarios extrapolate from observed runtimes using explicit width, depth and
budget heuristics. Realistic adds scheduling overhead and a recent-rate floor.
Maximum is a conservative serial timeout envelope including startup and missing
preparation; pauses, stalls and retries can exceed it. These are not confidence
intervals. Future preparation cost is included only in the maximum envelope.
Successful-only runtime medians exclude timeouts. Provisional leaders have
unequal coverage and do not establish preserved circuit rankings.

`show_run.py` completion numbers are one-based and ordered by recorded timestamp,
including imports, within the selected stage. Its historical default points to
the stopped broad campaign: specify the balanced config or `--output` explicitly.

## Portability and initialization

These six configs are local two-A100 presets. They default to the NAS path
`/mnt/main/tklee/quantum-simulation-benchmark/results/balanced_simulation_study`.
Before a new deployment, edit each config's output directory and GPU IDs; create
`study.json` in the new study root with a `stages` list whose entries contain
`config` (repository-relative config path) and matching `output_dir`.
The shipped configs themselves remain usable directly with `run_sim.py`.

Use `scripts/run_balanced_study.py --root /path/to/study` and
`scripts/monitor_study.py --root /path/to/study --watch` for a relocated study.
The launcher resumes existing manifests and takes an exclusive study lock.
Scientific config/source changes require fresh outputs; execution-only worker
settings follow the runner's resume rules.

`prepare_balanced_study.py` is the one-time migration used on this machine, not a
routine launcher. It requires the original stopped campaign and refuses an
already initialized study before changing files. Do not use it to resume.
Raw results, GPU builds, virtual environments and diagnostics under `.build`
remain outside Git. The repository versions the settings and analysis tools,
not the NAS result archive.
