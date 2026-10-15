"""Planar crossing junctions.

The country road extracts contain the main carriageways but not the
``*_link`` ramp ways, so two roads that meet at a grade-separated interchange
share no node and are disconnected in the raw graph (verified on Denmark: the
E45 motorway crosses every radial road around Aarhus without a shared node).
This module finds every *proper* crossing of two segments (interiors intersect,
no shared end point), inserts a node at the crossing and splits both segments.

Method
------
* Segments are treated as straight lines in (lon, lat). Whether two segments
  intersect, and the fractional positions t, s of the crossing along each, are
  invariant under affine maps, so no projection is needed; for segments of a
  few hundred metres the difference from geodesic lines is negligible.
* Segments are bucketed on a uniform grid (``cell_deg``) by bounding box.
  A candidate pair is tested only in the grid cell containing the lower-left
  corner of the overlap of the two bounding boxes; both segments are always
  registered in that cell, so every pair is tested exactly once.
* Orientation predicates decide proper crossings; segments that share an end
  node are skipped (they are already joined).
* Crossing points are quantized to micro-degrees like all other nodes.

Caveat (see METHODS.md): where one road passes over another with no
interchange, a junction that does not exist is introduced. Among the major road
classes in the extracts most such crossings are interchanges. Set
``insert_crossings=False`` if the input already includes link roads.
"""
from __future__ import annotations

import numpy as np

try:
    import numba as nb
except Exception:  # pragma: no cover
    nb = None

_OFF = np.int64(1 << 20)


def _register_impl(x0, y0, x1, y1, inv_cell):
    m = x0.shape[0]
    cnt = np.zeros(m + 1, dtype=np.int64)
    for i in range(m):
        cx0 = int(np.floor(min(x0[i], x1[i]) * inv_cell)); cx1 = int(np.floor(max(x0[i], x1[i]) * inv_cell))
        cy0 = int(np.floor(min(y0[i], y1[i]) * inv_cell)); cy1 = int(np.floor(max(y0[i], y1[i]) * inv_cell))
        cnt[i + 1] = (cx1 - cx0 + 1) * (cy1 - cy0 + 1)
    for i in range(m):
        cnt[i + 1] += cnt[i]
    seg = np.empty(cnt[m], dtype=np.int64)
    key = np.empty(cnt[m], dtype=np.int64)
    for i in range(m):
        cx0 = int(np.floor(min(x0[i], x1[i]) * inv_cell)); cx1 = int(np.floor(max(x0[i], x1[i]) * inv_cell))
        cy0 = int(np.floor(min(y0[i], y1[i]) * inv_cell)); cy1 = int(np.floor(max(y0[i], y1[i]) * inv_cell))
        k = cnt[i]
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                seg[k] = i
                key[k] = (np.int64(cx) + _OFF) * (np.int64(4) * _OFF) + (np.int64(cy) + _OFF)
                k += 1
    return seg, key


def _test_impl(seg_sorted, key_sorted, starts, x0, y0, x1, y1, a_id, b_id, inv_cell, cap):
    out_i = np.empty(cap, dtype=np.int64)
    out_j = np.empty(cap, dtype=np.int64)
    out_t = np.empty(cap, dtype=np.float64)
    out_s = np.empty(cap, dtype=np.float64)
    n_out = 0
    for c in range(starts.shape[0] - 1):
        lo = starts[c]
        hi = starts[c + 1]
        if hi - lo < 2:
            continue
        k = key_sorted[lo]
        cx = k // (np.int64(4) * _OFF) - _OFF
        cy = k % (np.int64(4) * _OFF) - _OFF
        for p in range(lo, hi):
            i = seg_sorted[p]
            for q in range(p + 1, hi):
                j = seg_sorted[q]
                if a_id[i] == a_id[j] or a_id[i] == b_id[j] or b_id[i] == a_id[j] or b_id[i] == b_id[j]:
                    continue
                ixmin = max(min(x0[i], x1[i]), min(x0[j], x1[j]))
                iymin = max(min(y0[i], y1[i]), min(y0[j], y1[j]))
                ixmax = min(max(x0[i], x1[i]), max(x0[j], x1[j]))
                iymax = min(max(y0[i], y1[i]), max(y0[j], y1[j]))
                if ixmin > ixmax or iymin > iymax:
                    continue
                if int(np.floor(ixmin * inv_cell)) != cx or int(np.floor(iymin * inv_cell)) != cy:
                    continue
                rx = x1[i] - x0[i]; ry = y1[i] - y0[i]
                sx = x1[j] - x0[j]; sy = y1[j] - y0[j]
                o1 = rx * (y0[j] - y0[i]) - ry * (x0[j] - x0[i])
                o2 = rx * (y1[j] - y0[i]) - ry * (x1[j] - x0[i])
                o3 = sx * (y0[i] - y0[j]) - sy * (x0[i] - x0[j])
                o4 = sx * (y1[i] - y0[j]) - sy * (x1[i] - x0[j])
                if o1 * o2 < 0.0 and o3 * o4 < 0.0:
                    den = rx * sy - ry * sx
                    if den == 0.0:
                        continue
                    t = ((x0[j] - x0[i]) * sy - (y0[j] - y0[i]) * sx) / den
                    s = ((x0[j] - x0[i]) * ry - (y0[j] - y0[i]) * rx) / den
                    if n_out < cap:
                        out_i[n_out] = i; out_j[n_out] = j; out_t[n_out] = t; out_s[n_out] = s
                    n_out += 1
    return out_i, out_j, out_t, out_s, n_out


if nb is not None:
    _register = nb.njit(cache=True)(_register_impl)
    _test = nb.njit(cache=True)(_test_impl)
else:  # pragma: no cover
    _register, _test = _register_impl, _test_impl


def find_crossings(lat, lon, u, v, cell_deg=0.02):
    """Return (seg_i, seg_j, t_i, t_j): proper crossings between edges u-v.

    Edges spanning the antimeridian (|dlon| > 180) are ignored.
    """
    x0, y0, x1, y1 = lon[u], lat[u], lon[v], lat[v]
    ok = np.abs(x1 - x0) <= 180.0
    idx = np.flatnonzero(ok)
    seg, key = _register(x0[idx], y0[idx], x1[idx], y1[idx], 1.0 / cell_deg)
    order = np.argsort(key, kind="stable")
    seg, key = seg[order], key[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
    cap = max(1024, idx.size // 4)
    while True:
        oi, oj, ot, os_, n = _test(seg, key, starts, x0[idx], y0[idx], x1[idx], y1[idx],
                                   u[idx], v[idx], 1.0 / cell_deg, cap)
        if n <= cap:
            break
        cap = n + 1
    return idx[oi[:n]], idx[oj[:n]], ot[:n], os_[:n]


def split_at_crossings(lat, lon, keys, u, v, cell_deg=0.02):
    """Insert crossing nodes and split edges.

    Returns (new_keys, new_u, new_v, n_crossings) with edge lengths to be
    recomputed by the caller from node coordinates.
    """
    from .graphops import pack_keys, quantize

    si, sj, ti, tj = find_crossings(lat, lon, u, v, cell_deg)
    n = si.size
    if n == 0:
        return keys, u, v, 0
    # crossing coordinates
    plat = lat[u[si]] + ti * (lat[v[si]] - lat[u[si]])
    plon = lon[u[si]] + ti * (lon[v[si]] - lon[u[si]])
    pkey = pack_keys(quantize(plat), quantize(plon))
    # each crossing splits two edges: list (edge, t, crossing key)
    e = np.concatenate([si, sj])
    t = np.concatenate([ti, tj])
    k = np.concatenate([pkey, pkey])
    new_keys = np.unique(np.concatenate([keys, pkey]))
    remap = np.searchsorted(new_keys, keys)
    U = remap[u]; V = remap[v]
    kidx = np.searchsorted(new_keys, k)
    order = np.lexsort((t, e))
    e, t, kidx = e[order], t[order], kidx[order]
    split = np.zeros(u.size, dtype=bool); split[e] = True
    # unsplit edges stay; split edges become chains u -> c1 -> c2 ... -> v
    bnd = np.flatnonzero(np.r_[True, e[1:] != e[:-1], True])
    cu, cv = [U[~split]], [V[~split]]
    first = bnd[:-1]; last = bnd[1:] - 1
    ee = e[first]
    cu.append(U[ee]); cv.append(kidx[first])           # u -> first crossing
    cu.append(kidx[last]); cv.append(V[e[last]])        # last crossing -> v
    inner = np.ones(e.size, dtype=bool); inner[last] = False
    cu.append(kidx[inner]); cv.append(kidx[np.flatnonzero(inner) + 1])  # between consecutive crossings
    return new_keys, np.concatenate(cu), np.concatenate(cv), n
