# Packaging validation — 2026-10-07

The original long campaign remained stopped. Only small new validation runs were launched.

| Check | Result |
| --- | --- |
| Configuration/default schedule | 360 cases × 52 settings = 18,720 runs; 320 preliminary runs; deterministic ordering and monotonic 10 → 12 → 14 widths |
| Automated tests | Five passed with runtime tests enabled: config/schedule checks, cancellation/preview, numerical equivalence, timeout/restart/interruption |
| CPU integration | Two concurrent CPU workers, chain/grid circuits, 14 pilot + 14 campaign records; all successful |
| GPU integration | One GPU worker, chain/QFT-style circuits, all eight numerical families and selected CPU/GPU variants: 24 pilot + 24 campaign records; all successful |
| Independent exact pilot checks | Four CPU and six GPU-suite checks against Aer density simulation; maximum discrepancy across those checks 1.57e-14 in Frobenius norm |
| Resume | Completed CPU output reopened without repeating simulation jobs |
| Timeout recovery | A deliberately sleeping worker was killed at its deadline; a new worker completed the next request; old successful output was not reused on timeout |
| Interruption | Worker supervision returned an interrupted record on a stop event |
| Reports | CSV/JSON summaries and per-family plots generated successfully |
| Wheel | Standalone wheel built successfully |
| Fresh installation | Clean Python 3.11 environment installed `.[cpu]`; dependency check passed; four ordinary tests passed and runtime test was opt-in skipped; another 14 pilot + 14 campaign records all succeeded; resume succeeded |

Small numerical tests use a noisy three-qubit circuit with a nonlocal gate and nonidentity logical output mapping. Dense, untruncated density MPS, full-rank factorization, complete Pauli expansion and exact tensor contraction agree with an independent Aer result to better than 1e-10. A separate explicit eigenvalue projection verifies the fidelity score.

CPU/GPU integration used the versions in `TESTED_ENVIRONMENT.json`. The fresh CPU environment resolved NumPy 2.4.6 rather than 2.2.6; Qiskit 2.5.2, Aer 0.17.2 and IBM Runtime 0.49.0 were unchanged. Small numerical agreement is a correctness check, not evidence that approximate rankings generalize to larger circuits.

Not established here: a full 18,720-run campaign, performance on the destination hardware, multiple physical GPU workers running concurrently, a fresh GPU source build on the destination machine, exhaustive behavior of every high approximation budget, or GPU kernel cycle counters. Multiworker dispatch was exercised with two CPU workers; GPU numerical paths were exercised on one GPU.
