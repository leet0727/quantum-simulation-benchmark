# Configuration reference

Unknown keys are rejected to catch spelling mistakes. Paths are relative to the launch working directory. `--plan` shows all resolved defaults without creating outputs.

| Key | Meaning / default |
| --- | --- |
| `output_dir` | `results/experiment`; use a new directory for changed scientific conditions |
| `circuits.families` | Any nonempty subset of `chain`, `brickwork`, `clifford`, `clusters`, `all_to_all`, `qaoa`, `grid2d`, `qft` |
| `circuits.qubits` / `layers` | Arrays of positive integers; at least 2 qubits |
| `circuits.seeds` | Explicit positive seeds; alternatively `seed_count` (5) and `seed_start` (22101) |
| `methods` | Nonempty array of method objects below |
| `trajectory_repeats` | Independent trajectory repetitions per circuit/setting, default 1; deterministic distinct simulator seeds |
| `timeout_s` | 7200, per simulation including scoring and output; initialization/preparation have separate limits |
| `preparation_timeout_s` | 7200 per circuit preparation attempt |
| `startup_timeout_s` | 300 per worker startup |
| `noise.backend` | `FakeMarrakesh`, or `none` for ideal channels on the same routed target |
| `noise.optimization_level` | Qiskit transpilation level, 0–3; default 1 |
| `reference.method` | `auto`, `dense_cpu`, `dense_gpu`, `none`; auto chooses GPU if assigned; an exact reference setting is added if absent |
| `pilot` | `enabled: true`, `qubits: 4`, `layers: 2`, `seed: 22001` |
| `accuracy` | Thresholds `fidelity_score_abs_error: 0.01`, `observable_vector_rmse: 0.01`; both required for joint pass |

`reference.method: "none"` disables automatic reference insertion. An explicitly selected exact dense or tensor-network method can still establish a reference. Without an exact result, reports leave reference-based errors blank.

## Methods

Each object has `name`, `devices: ["cpu"]` (or `["gpu"]`, or both), and the applicable fields:

| `name` | Budget fields | Supported devices |
| --- | --- | --- |
| `dense` | None | CPU/GPU |
| `density_mps` | `bond_dimensions` | CPU/GPU |
| `dmrg` | `bond_dimensions`, `sweeps` (4) | GPU |
| `low_rank` | `ranks` | CPU/GPU |
| `trajectory_dense` | `shots` | CPU/GPU |
| `trajectory_mps` | `bond_dimensions`, `shots`, `sweep`, `fixed_shots` | CPU |
| `sparse_pauli` | `terms`, `cutoffs` | CPU |
| `tensor_network` | None; greedy exact contraction | CPU/GPU |

Budget ladders accept `[16, 64, 256, 1024]` or `{"start":16,"max":1024,"factor":4}`. Cutoffs pair positionally with the **sorted** term ladder. Specify one cutoff per cap. Sparse-Pauli's retained identity term is preserved.

MPS trajectory `sweep` choices:

- `paired`: pair bond and shot ladders; a one-element ladder is fixed across the other ladder.
- `paired_with_fixed_shots` (default): paired settings plus a fixed-shot bond sweep, removing duplicates. `fixed_shots` defaults to the smallest shot value.
- `cartesian`: every bond × shots combination.

For pure trajectories, the largest possible Schmidt rank is `2^floor(n/2)`. At 14 qubits, caps above 128 cannot increase representational capacity. Density MPS represents a vectorized density operator with local dimension 4 and has different bond scaling. Diagnostics record the trajectory effective cap.

## Execution and resources

```json
"execution": {
  "mode": "parallel",
  "gpu_ids": [0, 1],
  "cpu_workers": 2,
  "threads_per_worker": 4,
  "scoring_device": "auto"
}
```

Default: isolated mode, GPU 0, no dedicated CPU workers, 2 threads. CPU-only: `gpu_ids: []`, `cpu_workers: 1` in isolated mode. `scoring_device` may be `auto`, `cpu`, or `gpu`; auto uses the worker's GPU if it has one. GPU scoring with dedicated CPU workers is rejected. A CPU method executed on a GPU lane may still use GPU **scoring**, recorded separately from the simulation device.

Resource values are positive GiB quantities:

| `resources` field | Default | Scope |
| --- | --- | --- |
| `host_rss_gib` | 96 | Sampled host RSS per worker; worker killed when exceeded |
| `gpu_memory_gib` | 42 | Sampled device-wide memory; requires working NVML telemetry |
| `workspace_gib` | 40 | Conservative low-rank SVD estimate |
| `tn_intermediate_gib` | 16 | Largest planned tensor-network intermediate |
| `max_output_gib` | 8 | Reject cases whose full density output alone is larger |
| `reference_cache_gib` | 48 | Exact reference matrices; eviction between periodic report updates |
| `min_free_disk_gib` | 20 | Free disk required before dispatch |

These are supervisory guards, not reservations or universal allocator limits. Sampling can miss fast spikes, reference-cache eviction is periodic and can temporarily overshoot, and unavailable GPU telemetry cannot enforce its sampled cap. The OS/driver can still report an allocation failure; it is recorded rather than treated as an accuracy result.

`telemetry.enabled` enables/disables GPU NVML collection; `telemetry.interval_s` (1) controls trace sampling. Host supervision remains active.

## Scheduling

`qubits_first` orders by qubits, budget, layers, seed, family, method, device, repetition. `budgets_first` swaps the first two. `case_first` orders qubits, layers, seed, family before budget. The configured family and seed array order determines their relative priority. A budget's order is local to its method, not a claim of equal computational cost across methods.

```json
"scheduling": {
  "mode": "custom",
  "order_by": ["layers", "qubits", "budget", "seed", "family", "method", "device", "repeat"],
  "seed": 2718,
  "method_order": ["dense", "density_mps", "dmrg", "low_rank", "trajectory_dense", "trajectory_mps", "sparse_pauli", "tensor_network"],
  "preliminary": {
    "enabled": false,
    "methods": ["dense:gpu", "density_mps:cpu"]
  }
}
```

All eight method names must appear once in `method_order`, even when only a subset is selected. Custom sort keys may be a subset of those shown. The first key creates scheduling phase barriers after the preliminary phase. `scheduling.seed` determines simulator seed derivation; it does not shuffle the schedule. The preliminary phase selects lowest-budget settings at the minimum qubit/layer values, across all seeds/families. The designated exact reference is included automatically in this phase.
