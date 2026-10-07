# Source notices

This repository packages the user's existing simulation-method benchmark and adds a standalone configurable launcher. `PROVENANCE.json` records hashes of the copied numerical sources and their packaged forms.

- `simbench/density_mps.py` was copied from the local qTPU research implementation (`evaluation/iontrap/density_mps.py`). The qTPU Apache 2.0 license is retained in `licenses/qtpu-APACHE-2.0.txt`. Upstream qTPU: https://github.com/TUM-DSE/qtpu . The local research file is not claimed to be an upstream released component.
- `simbench/dmrg_engine.py` was carried from the local benchmark's extracted SBCC projection-MPS engine. It is a numerical implementation, not the SBCC search framework. Its source checkout supplied no repository-wide license. This packaging does not assert a new license for it.
- `simbench/kernels.py` and `simbench/engine.py` derive from the existing local simulation benchmark. Imports were made relative, resources configurable, and production Aer thread counts configurable. Mathematical scoring and numerical families were retained.
- Qiskit, Qiskit Aer, IBM Runtime's fake backends, NumPy, SciPy, CuPy, cuQuantum, opt_einsum, psutil and Rich are external dependencies, governed by their respective licenses; their packages are not vendored here.

No blanket license is assigned to the mixed-origin repository. Public visibility by itself does not grant an additional software license. The retained Apache text applies to the corresponding source lineage, not automatically to every file.
