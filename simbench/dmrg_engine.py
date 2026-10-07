"""Unmodified numerical primitives extracted from SBCC; see source_provenance.json."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import time

def _array(value: Any) -> Any:
    """Cast while preserving a CuPy-like device array."""
    module = type(value).__module__
    if module.startswith("cupy") and hasattr(value, "astype"):
        return value.astype(value.dtype if value.dtype.kind == "c" else "complex128", copy=False)
    return np.asarray(value, dtype=np.complex128)

def _three(tensor: Any, *, site: int | None = None, num_sites: int | None = None) -> Any:
    value = _array(tensor)
    if value.ndim == 2:
        if site is None or num_sites is None:
            raise ValueError("rank-2 edge tensors require explicit site and num_sites")
        if site == 0:
            return value[None, :, :]
        if site == num_sites - 1:
            return value[:, :, None]
        raise ValueError("rank-2 tensors are valid only at MPS edges")
    if value.ndim != 3:
        raise ValueError("MPS tensors must have rank 2 or 3")
    return value

def mps_overlap(left: Sequence[Any], right: Sequence[Any]) -> complex:
    """Contract ``<left|right>`` without constructing a statevector."""
    if len(left) != len(right) or not left:
        raise ValueError("MPS sequences must be non-empty and equally long")
    first = _three(left[0], site=0, num_sites=len(left))
    xp = __import__("cupy") if type(first).__module__.startswith("cupy") else np
    environment = xp.ones((1, 1), dtype=xp.complex128)
    for site, (left_tensor, right_tensor) in enumerate(zip(left, right)):
        left_value = _three(left_tensor, site=site, num_sites=len(left))
        right_value = _three(right_tensor, site=site, num_sites=len(right))
        if left_value.shape[1] != right_value.shape[1] or environment.shape != (
            left_value.shape[0],
            right_value.shape[0],
        ):
            raise ValueError("incompatible MPS tensor dimensions")
        environment = xp.einsum("ab,asi,bsj->ij", environment, left_value.conj(), right_value, optimize=True)
    if environment.shape != (1, 1):
        raise ValueError("MPS must have open boundary bond dimensions one")
    return complex(environment[0, 0].item() if hasattr(environment[0, 0], "item") else environment[0, 0])

def _load_cuda():
    try:
        import cupy as cp
        import cuquantum
        from cuquantum.bindings import cutensornet as cutn
    except Exception as exc:  # pragma: no cover - hardware dependent
        raise RuntimeError(
            "the DMRG oracle needs CuPy, cuQuantum (pip install -e '.[gpu]') and a CUDA device"
        ) from exc
    return cp, cuquantum, cutn

def _mps_change(a, b) -> float:
    """``1 - |<a|b>|^2 / (<a|a><b|b>)``: the squared angle between successive sweeps."""
    ab = mps_overlap(a, b)
    aa, bb = mps_overlap(a, a).real, mps_overlap(b, b).real
    if aa <= 0 or bb <= 0:
        return float("inf")
    return float(max(0.0, 1.0 - abs(ab) ** 2 / (aa * bb)))

class ProjectionMPSEngine:
    """cuTensorNet projection-MPS DMRG on one GPU, holding one library handle for its lifetime.

    Args:
        bond_dim: maximum MPS bond dimension ``chi``. A bond is *saturated* when ``chi``
            reaches ``d**min(i+1, n-i-1)``, its exact rank bound; with every bond
            saturated the fit is exact up to convergence.
        max_sweeps: full (left-right-left) sweeps at most.
        tol: stop once two consecutive sweeps differ by less than this in squared angle.
            That is a *squared* quantity, so the MPS itself moves by about ``sqrt(tol)``.
        scratch_fraction: share of free GPU memory the workspace may request.
        seed: fixes the random initial MPS, so every run is reproducible.
    """

    def __init__(self, *, bond_dim: int = 64, max_sweeps: int = 4, tol: float = 1e-13,
                 scratch_fraction: float = 0.2, device_id: int = 0, seed: int = 1729) -> None:
        if bond_dim < 1 or max_sweeps < 1 or not 0.0 < scratch_fraction <= 1.0:
            raise ValueError("bond_dim and max_sweeps must be positive and scratch_fraction in (0, 1]")
        self.bond_dim = int(bond_dim)
        self.max_sweeps = int(max_sweeps)
        self.tol = float(tol)
        self.scratch_fraction = float(scratch_fraction)
        self.device_id = int(device_id)
        self.seed = int(seed)
        self._handle = None
        self._stream = None

    def close(self) -> None:
        if self._handle is not None:
            _, _, cutn = _load_cuda()
            cutn.destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:                                      # noqa: BLE001
            pass

    def run(self, dims: Sequence[int], ops: Sequence[tuple]) -> tuple[list[np.ndarray], dict[str, Any]]:
        """Fit an MPS to the network ``ops |0...0>`` over sites of dimension ``dims``."""
        n = len(dims)
        d = int(dims[0])
        if n < 1 or any(int(x) != d for x in dims):
            raise ValueError("need at least one site and a uniform local dimension")
        started = time.perf_counter()
        if n == 1:
            # No bond to optimise: a single site is its own exact MPS.
            vec = np.zeros(d, dtype=np.complex128)
            vec[0] = 1.0
            for _, matrix, _ in ops:
                vec = matrix @ vec
            return [vec.reshape(1, d, 1)], {
                "method": "single_site", "sweeps": 0, "final_delta": 0.0,
                "bond_saturated": True, "max_bond": 1, "seconds": time.perf_counter() - started,
            }

        cp, cuquantum, cutn = _load_cuda()
        full_rank = [d ** min(i + 1, n - i - 1) for i in range(n - 1)]
        bonds = [min(r, self.bond_dim) for r in full_rank]
        with cp.cuda.Device(self.device_id):
            if self._handle is None:
                self._stream = cp.cuda.Stream()
                self._handle = cutn.create()
            handle, stream = self._handle, self._stream
            stream.use()
            state = projection = work_desc = None
            try:
                state = cutn.create_state(handle, cutn.StatePurity.PURE, n, tuple(int(x) for x in dims),
                                          cuquantum.cudaDataType.CUDA_C_64F)
                operators = []          # the state keeps raw device pointers to these
                for sites, matrix, unitary in ops:
                    k = len(sites)
                    tensor = cp.asarray(np.asarray(matrix).T, dtype=cp.complex128, order="F")
                    if k > 1:
                        tensor = tensor.reshape((d,) * (2 * k), order="F")
                    operators.append(tensor)
                    cutn.state_apply_tensor_operator(handle, state, k, list(sites), tensor.data.ptr,
                                                     0, 1, 0, int(bool(unitary)))

                rng = np.random.default_rng(self.seed)
                extents, tensors = [], []
                for i in range(n):
                    left = 1 if i == 0 else bonds[i - 1]
                    right = 1 if i == n - 1 else bonds[i]
                    shape = (d, right) if i == 0 else (left, d) if i == n - 1 else (left, d, right)
                    host = rng.normal(size=shape) + 1j * rng.normal(size=shape)
                    host /= np.linalg.norm(host)
                    extents.append(shape)
                    tensors.append(cp.asarray(host, dtype=cp.complex128))

                env_bounds = np.asarray([(i - 1, i + 1) for i in range(n)], dtype=cutn.mps_env_bounds_dtype)
                ortho = np.asarray([(-1, n)], dtype=cutn.mps_env_bounds_dtype)
                projection = cutn.create_state_projection_mps(
                    handle, 1, [state], 0, False, len(env_bounds), env_bounds.ctypes.data,
                    cutn.BoundaryCondition.OPEN, n, 0, extents,
                    [[s // t.itemsize for s in t.strides] for t in tensors],
                    [t.data.ptr for t in tensors], ortho.ctypes.data,
                )
                work_desc = cutn.create_workspace_descriptor(handle)
                budget = int(cp.cuda.Device().mem_info[0] * self.scratch_fraction)
                cutn.state_projection_mps_prepare(handle, projection, budget, work_desc, stream.ptr)
                required = int(cutn.workspace_get_memory_size(
                    handle, work_desc, cutn.WorksizePref.RECOMMENDED,
                    cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH))
                if required > budget:
                    raise RuntimeError(f"DMRG workspace needs {required} bytes, budget is {budget}")
                scratch = cp.cuda.alloc(required)
                cutn.workspace_set_memory(handle, work_desc, cutn.Memspace.DEVICE,
                                          cutn.WorkspaceKind.SCRATCH, scratch.ptr, required)

                previous = [t.copy() for t in tensors]
                sweeps, delta = 0, float("inf")
                for sweeps in range(1, self.max_sweeps + 1):
                    for order in (range(n), range(n - 1, -1, -1)):
                        for i in order:
                            local = np.asarray([(i - 1, i + 1)], dtype=cutn.mps_env_bounds_dtype)
                            current = cp.empty_like(tensors[i])
                            update = cp.empty_like(tensors[i])
                            strides = [s // update.itemsize for s in update.strides]
                            # NVIDIA's sample extracts before computing the environment; without
                            # it compute_tensor_env returned INVALID_VALUE (cuTensorNet 2.12.2).
                            cutn.state_projection_mps_extract_tensor(
                                handle, projection, local.ctypes.data, strides, current.data.ptr,
                                work_desc, stream.ptr)
                            cutn.state_projection_mps_compute_tensor_env(
                                handle, projection, local.ctypes.data, 0, 0, strides, update.data.ptr,
                                0, 0, work_desc, stream.ptr)
                            cutn.state_projection_mps_insert_tensor(
                                handle, projection, local.ctypes.data, local.ctypes.data, strides,
                                update.data.ptr, work_desc, stream.ptr)
                    stream.synchronize()
                    # insert_tensor writes through to the registered buffers in ``tensors``
                    delta = _mps_change(previous, tensors)
                    previous = [t.copy() for t in tensors]
                    if sweeps >= 2 and delta <= self.tol:
                        break
                host = [cp.asnumpy(t) for t in tensors]
                del operators, scratch
            finally:
                if work_desc is not None:
                    cutn.destroy_workspace_descriptor(work_desc)
                if projection is not None:
                    cutn.destroy_state_projection_mps(projection)
                if state is not None:
                    cutn.destroy_state(state)
                stream.synchronize()
        return host, {
            "method": "cutensornet_projection_mps", "sweeps": sweeps, "final_delta": delta,
            "bond_saturated": all(self.bond_dim >= r for r in full_rank),
            "max_bond": max(bonds), "seconds": time.perf_counter() - started,
        }
