"""Min-plus kernels with path-attribute tracking.

Every cost matrix ``D`` travels with three companion matrices that describe the
*chosen* (cost-minimizing) route: land km, sea km and number of sea legs. When
a cheaper route replaces the incumbent, the companions are replaced by the sums
along the new route, so the companions always describe the arg-min route.

Numba (parallel) is used when available; NumPy fallbacks are provided and give
identical results (tested), only slower.
"""
from __future__ import annotations

import numpy as np

try:
    import numba as nb

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False


def _fw_numpy(D, Lk, Sk, Nl):
    P = D.shape[0]
    for k in range(P):
        cand = D[:, k:k + 1] + D[k:k + 1, :]
        better = cand < D
        if not better.any():
            continue
        lcand = Lk[:, k:k + 1] + Lk[k:k + 1, :]
        scand = Sk[:, k:k + 1] + Sk[k:k + 1, :]
        ncand = Nl[:, k:k + 1] + Nl[k:k + 1, :]
        D[better] = cand[better]; Lk[better] = lcand[better]
        Sk[better] = scand[better]; Nl[better] = ncand[better]
    return D, Lk, Sk, Nl


def _minplus_numpy(A, LA, SA, NA, B, LB, SB, NB, block=16):
    m, p = A.shape
    n = B.shape[1]
    C = np.full((m, n), np.inf); LC = np.zeros((m, n)); SC = np.zeros((m, n)); NC = np.zeros((m, n), np.int32)
    for r0 in range(0, m, block):
        r1 = min(r0 + block, m)
        tot = A[r0:r1, :, None] + B[None, :, :]          # (b, p, n)
        q = np.argmin(tot, axis=1)                          # (b, n)
        rows = np.arange(r1 - r0)[:, None]
        C[r0:r1] = np.take_along_axis(tot, q[:, None, :], axis=1)[:, 0, :]
        LC[r0:r1] = LA[r0:r1][rows, q] + LB[q, np.arange(n)[None, :]]
        SC[r0:r1] = SA[r0:r1][rows, q] + SB[q, np.arange(n)[None, :]]
        NC[r0:r1] = NA[r0:r1][rows, q] + NB[q, np.arange(n)[None, :]]
    return C, LC, SC, NC


if HAVE_NUMBA:

    @nb.njit(parallel=True, cache=True)
    def _fw_numba(D, Lk, Sk, Nl):
        P = D.shape[0]
        for k in range(P):
            dk = D[k].copy(); lk = Lk[k].copy(); sk = Sk[k].copy(); nk = Nl[k].copy()
            for i in nb.prange(P):
                dik = D[i, k]
                if dik == np.inf:
                    continue
                lik = Lk[i, k]; sik = Sk[i, k]; nik = Nl[i, k]
                for j in range(P):
                    c = dik + dk[j]
                    if c < D[i, j]:
                        D[i, j] = c
                        Lk[i, j] = lik + lk[j]
                        Sk[i, j] = sik + sk[j]
                        Nl[i, j] = nik + nk[j]
        return D, Lk, Sk, Nl

    @nb.njit(parallel=True, cache=True)
    def _minplus_numba(A, LA, SA, NA, B, LB, SB, NB):
        m, p = A.shape
        n = B.shape[1]
        C = np.full((m, n), np.inf); LC = np.zeros((m, n)); SC = np.zeros((m, n))
        NC = np.zeros((m, n), np.int32)
        for i in nb.prange(m):
            # only intermediate nodes reachable from i matter
            qs = np.flatnonzero(A[i] < np.inf)
            for j in range(n):
                best = np.inf; bq = -1
                for t in range(qs.size):
                    q = qs[t]
                    c = A[i, q] + B[q, j]
                    if c < best:
                        best = c; bq = q
                if bq >= 0:
                    C[i, j] = best
                    LC[i, j] = LA[i, bq] + LB[bq, j]
                    SC[i, j] = SA[i, bq] + SB[bq, j]
                    NC[i, j] = NA[i, bq] + NB[bq, j]
        return C, LC, SC, NC


def floyd_warshall(D, Lk, Sk, Nl, use_numba=True):
    if HAVE_NUMBA and use_numba:
        return _fw_numba(D, Lk, Sk, Nl)
    return _fw_numpy(D, Lk, Sk, Nl)


def minplus(A, LA, SA, NA, B, LB, SB, NB, use_numba=True):
    if HAVE_NUMBA and use_numba:
        return _minplus_numba(A, LA, SA, NA, B, LB, SB, NB)
    return _minplus_numpy(A, LA, SA, NA, B, LB, SB, NB)
