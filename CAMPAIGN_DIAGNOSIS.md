# Stopped campaign and SVD diagnosis — 2026-10-08

Historical diagnosis and decisions, recorded before the balanced redesign.
The production fallback is implemented; the later 64 GiB campaign stopped at
938 records and was superseded by [the balanced study](BALANCED_STUDY.md).
Statements below about stopped processes and unmodified source describe the
diagnosis stage, not the current running study.

The NAS campaign stopped cleanly at 215 terminal records: 214 successful,
one error, and four additional interrupted jobs. Interrupted jobs retry on resume.
No benchmark controller or workers remain running.

The only error was `clusters_n10_d16_s22103/density_mps_cpu_chi16__r0.json`.
Replaying its saved input reproduced NumPy SVD failure at zero-based stream
gate 3798, on SVD call 1882, during an internal swap. The matrix was complex
64 by 64, entirely finite, with maximum absolute entry 0.002391407429080908
and Frobenius norm 0.00505251252582964. NumPy's default solver and SciPy
`gesdd` both failed on the saved matrix; rescaling did not resolve it.
SciPy `gesvd` succeeded with relative reconstruction error 4.56e-15.
This isolates a solver convergence failure on this matrix, rather than
an observed memory exhaustion or non-finite input. Its deeper cause was
not established.

A diagnostic-only fallback reran the entire case successfully in 5.89 seconds,
with one fallback among 2717 SVD operations. Output was finite; fidelity-score
error was 0.00356 and observable-vector RMSE was 0.00802, both below 0.01.
This diagnostic did not overwrite the existing campaign's failed record.
Scripts, failed matrix, logs and detailed validation are in `.build/diagnostics`.
Benchmark source has not been changed.

`configs/two_gpu.json` now selects four CPU workers, keeping eight threads
and the existing 96 GiB host-RSS limit per worker. Its scientific fingerprint
matches the stopped manifest, so it is eligible for resume with unchanged code.
The recorded error remains terminal and will not retry automatically.

`configs/two_gpu_64gib.json` prepares four CPU workers and a 64 GiB host-RSS
limit per worker, including GPU workers' host RAM, with a fresh NAS output
directory ending in `two_gpu_campaign_64gib`. Its fingerprint differs from
the original campaign. Both configurations pass plan validation.

Changing code also changes the source hash and prevents resuming the old
campaign. A production SVD fallback therefore requires a fresh campaign
under the current reproducibility rules. Neither campaign has been resumed.

## Selected route

The user selected a fresh campaign with the solver fallback, four CPU workers
and 64 GiB host-RSS guards. The CPU fallback is now implemented and records
`svd_fallback_count`; non-finite inputs and GPU errors are not retried with
the CPU solver. All five automated tests pass, including an independent
Aer comparison with an induced default-SVD failure. The actual failed case
also succeeds with the production change and exactly one fallback.

The corrected run was launched in tmux session `campaign-fixed` with
`configs/two_gpu_64gib.json`. It reruns the pilot and all campaign jobs in
the fresh NAS output, preserving the old campaign. The old configuration
is no longer resumable with the corrected source because its source hash differs.
