# Methods, channels and scores

These are the existing benchmark's numerical families, not implementations of the proposed newer R1–R6 acceleration methods. The comparison operates on complete circuits; there is no cut search or proxy ranking.

| Method | Representation / evolution | Implementation and lineage |
| --- | --- | --- |
| Dense exact | Vectorized density matrix; local superoperators contract into a `4^n` tensor | Direct NumPy/CuPy implementation of channel evolution |
| Density MPS | Vectorized density operator as local-dimension-4 MPS; two-site update followed by truncated SVD | Local qTPU research implementation; operator-space MPS / TEBD ideas |
| Projection DMRG | MPS compression via variational projection sweeps, with explicit bond cap | Extracted local SBCC numerical engine using cuQuantum experimental MPS APIs; no SBCC framework dependency |
| Global low rank | `rho = X X†`; Kraus updates concatenate `[K1 X, K2 X, …]`; SVD retains a capped rank | Direct benchmark implementation inspired by low-rank density evolution; not a reproduction of a particular paper |
| Dense trajectories | Sample Kraus branches while evolving pure statevectors; average output density estimates | Qiskit Aer `statevector` with explicit Kraus channels and a density save instruction |
| MPS trajectories | Same stochastic channel unravelling, each pure trajectory stored as MPS with bond truncation | Qiskit Aer `matrix_product_state` |
| Sparse Pauli | Forward Pauli-transfer expansion, pruning coefficients by magnitude and retained-term cap | Direct benchmark implementation; not an observable-backpropagation / Heisenberg light-cone method |
| Tensor network | Channel tensors contracted using an `opt_einsum` greedy path | Exact contraction (subject to floating-point error), guarded by largest intermediate size; not truncated MPS |

Qiskit Aer and cuQuantum provide documented simulation primitives, not guarantees that the custom benchmark implementations equal an optimized research method. See the source provenance records for the code that was carried over. No new empirical speedup claim is made by this package.

## Circuit/noise construction

For each case, select a connected, induced subset of FakeMarrakesh's coupling graph with exactly the requested width. Transpile to that target using its basis and properties. Map the physical output back to logical order and validate it against the parent's ideal statevector.

Obtain gate noise channels from Aer `NoiseModel.from_backend(FakeMarrakesh())`, including its modeled gate/relaxation errors. Apply each noisy channel after its corresponding ideal gate. All methods receive the same explicit channels/Kraus operators. There is no separate DeviceSpec rate added on top. Readout errors are excluded because the target is the pre-measurement quantum state; barriers/delays are skipped, so no additional idle-noise schedule is modeled. `noise.backend: "none"` keeps routing but removes noise. Backend property snapshots and compiled QPY circuits are saved per case.

Family names describe generated structures; routing can change their interaction graph. Recorded features describe the compiled circuit. `all_to_all` uses random pairs, not a complete all-pairs layer; `qft` contains repeated QFT-style blocks with seeded rotations, not an unmodified standalone QFT.

## Score definitions

Let `rho_*` be the exact noisy output and `|psi>` the noiseless logical parent output. An approximation may return a matrix that is not positive or normalized. The benchmark retains that raw matrix, reports trace, negative eigenvalue mass and Hermiticity, and projects it onto positive trace-one matrices **only for its fidelity score**:

`Pi(rho) = argmin_{sigma >= 0, tr(sigma)=1} ||sigma - rho||_F`.

The Hermitian part's eigenvalues are projected onto the probability simplex: `lambda'_i = max(lambda_i - theta, 0)` with `sum lambda'_i = 1`. This is the legacy Euclidean projection, not just division by trace. The fidelity score is

`F(rho) = <psi|Pi(rho)|psi>`.

Its approximation error is `|F(rho_approx) - F(rho_*)|`. This is an error in the **ideal-state fidelity score**, not mixed-state Uhlmann fidelity between the approximate and exact noisy outputs.

For the common Pauli observable suite, raw expectations are `o_j(rho) = Re tr(P_j rho)`. The primary observable approximation error is

`RMSE = sqrt(mean_j (o_j(rho_approx) - o_j(rho_*))^2)`.

The separate `observable_rmse` field compares with the **ideal** state, measuring the simulated noise effect rather than approximation accuracy. `full_density_frobenius_error = ||rho_approx - rho_*||_F` is available when the dense exact reference remains cached. Sparse-Pauli/low-rank truncation diagnostics help distinguish projection artifacts from physical fidelity.

A joint pass requires both primary score errors to be at most the configured thresholds. Exact methods are references; they should not be declared the winning *approximation*. Reports preserve unsuccessful/unknown cases so timeout-heavy methods are not mistaken for uniformly fast ones. Trajectory repetitions produce separate seeded records; increasing shots generally reduces sampling error but a particular finite-shot realization need not improve monotonically.
