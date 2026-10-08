"""Density-space MPS (physical dimension 4) with canonical SVD truncation.

This is an MPO/vectorized-density method, NOT a pure-state trajectory method.
Full noise channels act locally; no trajectory sampling error. GPU uses CuPy.
Truncation is Hilbert-Schmidt, not a guaranteed positive or trace-preserving map.
No renormalization/clipping is performed; exact validation is mandatory.
"""
import numpy as np


def interleaved_channel(superop, k):
    """Qiskit column-vector density convention -> site (row, column) digits."""
    digits = np.array(np.unravel_index(np.arange(4**k), (4,) * k)).T
    row = (digits // 2) @ (2 ** np.arange(k))
    col = (digits % 2) @ (2 ** np.arange(k))
    idx = row + 2**k * col
    return np.asarray(superop)[np.ix_(idx, idx)]


class DensityMPS:
    def __init__(self, n, chi, *, gpu=True):
        if n < 1 or chi < 1:
            raise ValueError("n and chi must be positive")
        if gpu:
            import cupy as xp
        else:
            xp = np
        self.xp, self.chi = xp, chi
        self.tensors = [xp.zeros((1, 4, 1), dtype=xp.complex128) for _ in range(n)]
        for t in self.tensors:
            t[0, 0, 0] = 1
        self.discarded = 0.0
        self.svd_count = 0
        self.svd_fallback_count = 0
        self.swap_count = 0

    def center_pair(self, site):
        xp, ts = self.xp, self.tensors
        for j in range(site):
            a, d, b = ts[j].shape
            q, r = xp.linalg.qr(ts[j].reshape(a*d, b))
            ts[j] = q.reshape(a, d, q.shape[1])
            ts[j+1] = xp.einsum('ab,bsc->asc', r, ts[j+1])
        for j in range(len(ts)-1, site+1, -1):
            a, d, b = ts[j].shape
            q, r = xp.linalg.qr(ts[j].reshape(a, d*b).T)
            ts[j] = q.T.reshape(q.shape[1], d, b)
            ts[j-1] = xp.einsum('asb,bc->asc', ts[j-1], r.T)

    def pair(self, site, channel):
        xp, ts = self.xp, self.tensors
        self.center_pair(site)
        left, right = ts[site], ts[site+1]
        merged = xp.einsum('asi,itb->astb', left, right)
        merged = xp.einsum('uvst,astb->auvb', xp.asarray(channel).reshape(4,4,4,4), merged)
        matrix = merged.reshape(left.shape[0]*4, 4*right.shape[2])
        try:
            u, s, vh = xp.linalg.svd(matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            # LAPACK's divide-and-conquer solver can fail on finite inputs.
            # Retry CPU matrices with the classical solver; preserve GPU errors.
            if xp is not np or not np.all(np.isfinite(matrix)):
                raise
            from scipy.linalg import svd
            u, s, vh = svd(matrix, full_matrices=False, lapack_driver='gesvd')
            self.svd_fallback_count += 1
        keep = min(self.chi, len(s))
        self.discarded += float(xp.sum(s[keep:]**2).real.item())
        ts[site] = u[:, :keep].reshape(left.shape[0], 4, keep)
        ts[site+1] = (s[:keep, None]*vh[:keep]).reshape(keep, 4, right.shape[2])
        self.svd_count += 1

    def apply(self, qargs, superop):
        k = len(qargs)
        channel = interleaved_channel(superop, k)
        if k == 1:
            q = qargs[0]
            self.tensors[q] = self.xp.einsum('st,ati->asi', self.xp.asarray(channel), self.tensors[q])
        elif k == 2:
            a, b = qargs
            if a > b:
                a, b = b, a
                channel = channel.reshape(4,4,4,4).transpose(1,0,3,2).reshape(16,16)
            swap = np.eye(16).reshape(4,4,4,4).transpose(1,0,2,3).reshape(16,16)
            for j in range(b-1, a, -1):
                self.pair(j, swap)
                self.swap_count += 1
            self.pair(a, channel)
            for j in range(a+1, b):
                self.pair(j, swap)
                self.swap_count += 1
        else:
            raise ValueError(f"Unsupported channel arity {k}")

    def reduced(self, support):
        """Retain ordered Qiskit little-endian support; trace other sites directly."""
        xp = self.xp
        support = list(support)
        value = xp.ones((1, 1), dtype=xp.complex128)
        retained = []
        for site, t in enumerate(self.tensors):
            if site in support:
                value = xp.einsum('xa,asb->xsb', value, t).reshape(-1, t.shape[2])
                retained.append(site)
            else:
                value = value @ (t[:,0,:] + t[:,3,:])
        n = len(support)
        if not n:
            return value.reshape(1,1)
        tensor = value.reshape((2,2)*n)
        axes = [2*retained.index(q) for q in reversed(support)]
        axes += [2*retained.index(q)+1 for q in reversed(support)]
        return tensor.transpose(axes).reshape(2**n,2**n)

    def diagnostics(self):
        return {"bond_cap": self.chi, "actual_max_bond": max(t.shape[2] for t in self.tensors),
                "discarded_hilbert_schmidt_weight_sum": self.discarded,
                "svd_count": self.svd_count, "internal_swap_count": self.swap_count,
                "svd_fallback_count": self.svd_fallback_count,
                "device": "GPU" if self.xp.__name__ == "cupy" else "CPU"}
