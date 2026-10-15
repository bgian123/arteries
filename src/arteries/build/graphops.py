"""Vectorized graph operations used by the build pipeline.

Graphs are represented as plain arrays:

* ``keys``   int64[n]  -- node identity: packed integer micro-degree coordinates
* ``u, v``   int64[m]  -- endpoint indices into ``keys`` (undirected, u < v)
* ``w``      float64[m] -- edge length in km

All routines are exact with respect to shortest-path distances between the
*retained* nodes (see docs/METHODS.md, Section 3, for proofs).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

COORD_SCALE = 1_000_000  # micro-degrees; OSM extracts were stored at 6 decimals


# ---------------------------------------------------------------------------
# coordinate keys
# ---------------------------------------------------------------------------
def pack_keys(lat_i: np.ndarray, lon_i: np.ndarray) -> np.ndarray:
    """Pack int micro-degree coordinates into a single sortable int64 key."""
    return (lat_i.astype(np.int64) << 32) | (lon_i.astype(np.int64) & 0xFFFFFFFF)


def unpack_keys(keys: np.ndarray):
    keys = np.asarray(keys, dtype=np.int64)
    lat_i = (keys >> 32).astype(np.int64)
    lon_i = (keys & 0xFFFFFFFF).astype(np.int64)
    lon_i = np.where(lon_i >= 2**31, lon_i - 2**32, lon_i)
    return lat_i / COORD_SCALE, lon_i / COORD_SCALE


def quantize(x: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(x, dtype=np.float64) * COORD_SCALE).astype(np.int64)


# ---------------------------------------------------------------------------
# basic edge hygiene
# ---------------------------------------------------------------------------
def canonical_edges(u, v, w):
    """Orient u<v, drop self-loops, collapse parallel edges keeping the minimum length."""
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    a = np.minimum(u, v)
    b = np.maximum(u, v)
    keep = a != b
    a, b, w = a[keep], b[keep], w[keep]
    if a.size == 0:
        return a, b, w
    order = np.lexsort((w, b, a))  # by a, then b, then w ascending
    a, b, w = a[order], b[order], w[order]
    first = np.ones(a.size, dtype=bool)
    first[1:] = (a[1:] != a[:-1]) | (b[1:] != b[:-1])
    return a[first], b[first], w[first]


def degrees(n: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.bincount(u, minlength=n) + np.bincount(v, minlength=n)


def components(n: int, u: np.ndarray, v: np.ndarray):
    if n == 0:
        return 0, np.zeros(0, dtype=np.int64)
    g = coo_matrix((np.ones(u.size, dtype=np.int8), (u, v)), shape=(n, n)).tocsr()
    return connected_components(g, directed=False)


def compact(keys, u, v, w, extra_node_arrays=(), keep=None):
    """Remove isolated nodes (except those flagged in ``keep``) and re-index."""
    n = keys.size
    used = np.zeros(n, dtype=bool) if keep is None else keep.copy()
    used[u] = True
    used[v] = True
    newidx = np.cumsum(used) - 1
    out = [keys[used], newidx[u], newidx[v], w]
    out.extend(arr[used] for arr in extra_node_arrays)
    return out


# ---------------------------------------------------------------------------
# simplification
# ---------------------------------------------------------------------------
def _prune_impl(n, ptr, adj_node, adj_edge, deg, protected, alive_edge):
    queue = np.empty(n, dtype=np.int64)
    qh = 0
    qt = 0
    for x in range(n):
        if deg[x] == 1 and not protected[x]:
            queue[qt] = x
            qt += 1
    removed = 0
    while qh < qt:
        x = queue[qh]
        qh += 1
        if deg[x] != 1:
            continue
        for k in range(ptr[x], ptr[x + 1]):
            e = adj_edge[k]
            if alive_edge[e]:
                alive_edge[e] = False
                removed += 1
                deg[x] -= 1
                y = adj_node[k]
                deg[y] -= 1
                if deg[y] == 1 and not protected[y]:
                    queue[qt] = y
                    qt += 1
                break
    return removed


try:
    import numba as _nb

    _prune = _nb.njit(cache=True)(_prune_impl)
except Exception:  # pragma: no cover
    _prune = _prune_impl


def prune_leaves(n, u, v, w, protected, max_rounds=None):
    """Delete unprotected degree-1 nodes until none remain (dangling trees).

    A dangling unprotected node cannot be an interior vertex of any simple path
    between two other nodes, so removing it leaves all shortest-path distances
    between remaining nodes unchanged. Implemented as a single O(n + m) queue
    sweep (Numba), equivalent to repeating the one-shot pruning to a fixed point.
    """
    if u.size == 0:
        return u, v, w, 0
    m = u.size
    ends = np.concatenate([u, v])
    other = np.concatenate([v, u])
    eid = np.concatenate([np.arange(m), np.arange(m)])
    order = np.argsort(ends, kind="stable")
    adj_node = other[order]
    adj_edge = eid[order]
    ptr = np.searchsorted(ends[order], np.arange(n + 1))
    deg = (ptr[1:] - ptr[:-1]).astype(np.int64)
    alive = np.ones(m, dtype=np.bool_)
    removed = _prune(n, ptr.astype(np.int64), adj_node.astype(np.int64), adj_edge.astype(np.int64),
                     deg, protected.astype(np.bool_), alive)
    return u[alive], v[alive], w[alive], int(removed)


def contract_degree2(n, u, v, w, protected):
    """Replace every maximal chain of unprotected degree-2 nodes by one edge.

    ``anchor`` = protected or degree != 2. Each connected component of the
    sub-graph induced by non-anchor nodes is a simple path whose two ends are
    joined to anchors by exactly two boundary edges (or a cycle with none). The
    chain is replaced by a single edge anchor1-anchor2 whose length is the sum
    of all edges on it. Pure cycles (no anchor) are dropped; chains whose two
    boundary edges hit the same anchor become self-loops and are dropped. Both
    operations are distance-preserving for anchor pairs.
    """
    if u.size == 0:
        return u, v, w
    deg = degrees(n, u, v)
    anchor = protected | (deg != 2)
    au, av = anchor[u], anchor[v]

    keep_aa = au & av  # anchor-anchor edges stay as they are
    inner = ~au & ~av  # both endpoints are chain interiors
    bnd = au ^ av  # boundary edge: exactly one anchor

    # label chain interiors
    nonanchor_idx = np.flatnonzero(~anchor)
    if nonanchor_idx.size == 0:
        return u, v, w
    remap = np.full(n, -1, dtype=np.int64)
    remap[nonanchor_idx] = np.arange(nonanchor_idx.size)
    iu, iv = remap[u[inner]], remap[v[inner]]
    ncomp, lab = components(nonanchor_idx.size, iu, iv)

    chain_len = np.bincount(lab[iu], weights=w[inner], minlength=ncomp)

    bu, bv, bw = u[bnd], v[bnd], w[bnd]
    b_anchor = np.where(anchor[bu], bu, bv)
    b_inner = np.where(anchor[bu], bv, bu)
    b_lab = lab[remap[b_inner]]
    order = np.argsort(b_lab, kind="stable")
    b_lab, b_anchor, bw = b_lab[order], b_anchor[order], bw[order]
    counts = np.bincount(b_lab, minlength=ncomp)
    if np.any((counts != 0) & (counts != 2)):
        raise RuntimeError("degree-2 chain with != 2 boundary edges; graph is inconsistent")
    starts = np.flatnonzero(np.r_[True, b_lab[1:] != b_lab[:-1]]) if b_lab.size else np.zeros(0, np.int64)
    c = b_lab[starts]
    new_u = b_anchor[starts]
    new_v = b_anchor[starts + 1]
    new_w = chain_len[c] + bw[starts] + bw[starts + 1]

    u2 = np.concatenate([u[keep_aa], new_u])
    v2 = np.concatenate([v[keep_aa], new_v])
    w2 = np.concatenate([w[keep_aa], new_w])
    return canonical_edges(u2, v2, w2)


def simplify(keys, u, v, w, protected, *, drop_unprotected_components=True, log=None):
    """Full exact simplification loop. Returns compacted (keys, u, v, w, protected, stats)."""
    n = keys.size
    stats = {"nodes_in": int(n), "edges_in": int(u.size)}
    u, v, w = canonical_edges(u, v, w)

    if drop_unprotected_components and u.size:
        ncomp, lab = components(n, u, v)
        has_prot = np.zeros(ncomp, dtype=bool)
        has_prot[lab[protected]] = True
        keep = has_prot[lab[u]]
        stats["components_in"] = int(ncomp)
        stats["components_dropped_no_terminal_or_border"] = int((~has_prot).sum())
        stats["edges_dropped_in_unprotected_components"] = int((~keep).sum())
        u, v, w = u[keep], v[keep], w[keep]

    it = 0
    while True:
        it += 1
        m0 = u.size
        u, v, w = contract_degree2(n, u, v, w, protected)
        u, v, w, r = prune_leaves(n, u, v, w, protected)
        # re-index to the surviving nodes so later iterations cost O(current size)
        keys, u, v, w, protected = compact(keys, u, v, w, (protected,), keep=protected)
        n = keys.size
        if log:
            log(f"  simplify iter {it}: edges {m0:,} -> {u.size:,} (leaf edges removed {r:,})")
        if u.size == m0:
            break
    keys2, u2, v2, w2, prot2 = keys, u, v, w, protected
    stats.update({"nodes_out": int(keys2.size), "edges_out": int(u2.size), "iterations": it,
                  "km_out": float(w2.sum())})
    return keys2, u2, v2, w2, prot2, stats


# ---------------------------------------------------------------------------
# gap repair: link dead ends to *other* components within a radius
# ---------------------------------------------------------------------------
def candidate_links(q_xyz, q_idx, q_lab, t_xyz, t_lab, radius_km, t_tree=None):
    """All (query, target) pairs within ``radius_km`` whose component labels differ.

    Returns arrays (q_node, t_node, dist_km). Distances use the chord on the
    mean sphere converted back to arc length; final connector lengths are
    recomputed with the exact geodesic by the caller.
    """
    from scipy.spatial import cKDTree

    from ..geodesy import chord_to_km, km_to_chord

    if q_idx.size == 0 or t_lab.size == 0:
        e = np.zeros(0, dtype=np.int64)
        return e, e.copy(), np.zeros(0)
    qt = cKDTree(q_xyz)
    tt = t_tree if t_tree is not None else cKDTree(t_xyz)
    sdm = qt.sparse_distance_matrix(tt, km_to_chord(radius_km), output_type="coo_matrix")
    qi, ti, d = sdm.row.astype(np.int64), sdm.col.astype(np.int64), sdm.data
    ok = q_lab[qi] != t_lab[ti]
    return q_idx[qi[ok]], ti[ok], chord_to_km(d[ok])


def boruvka_select(src_lab, dst_lab, dist):
    """Keep, for every source component, its single shortest outgoing link."""
    if src_lab.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((dist, src_lab))
    s = src_lab[order]
    first = np.r_[True, s[1:] != s[:-1]]
    return order[first]


def bridge_components(lat, lon, u, v, w, radius_km, rounds=3, k_local=32, log=None):
    """Repair data gaps: link dead ends to the nearest node of a *different*
    connected component within ``radius_km`` (Borůvka rounds).

    Scalable formulation (the global raw graph has ~10^8 nodes):
      (a) dead ends of non-giant components -> exact nearest node of the giant
          component (k=1 query on a giant-only tree);
      (b) dead ends of non-giant components -> nearest node of another non-giant
          component (k=k_local query on a non-giant tree, first foreign label);
      (c) dead ends of the giant component -> exact nearest non-giant node.
    Every non-giant component then keeps only its shortest candidate link
    (Borůvka step), so a round adds at most (#components - 1) links and never
    adds a link inside a component. Link length = exact WGS84 geodesic.
    Returns (u, v, w, added_u, added_v, added_w, per_round_stats).
    """
    from scipy.spatial import cKDTree

    from ..geodesy import chord_to_km, geodesic_km, km_to_chord, to_unit_xyz

    n = lat.size
    xyz = to_unit_xyz(lat, lon)
    rch = km_to_chord(radius_km)
    add_u, add_v, add_w, rstats = [], [], [], []
    for r in range(rounds):
        ncomp, lab = components(n, u, v)
        if ncomp <= 1:
            break
        has_edge = np.zeros(n, dtype=bool); has_edge[u] = True; has_edge[v] = True
        size = np.bincount(lab[has_edge], minlength=ncomp)
        giant = int(np.argmax(size))
        deg = degrees(n, u, v)
        dead = np.flatnonzero(deg == 1)
        in_g = lab == giant
        g_idx = np.flatnonzero(in_g & has_edge)
        ng_idx = np.flatnonzero(~in_g & has_edge)
        dead_ng = dead[~in_g[dead]]
        dead_g = dead[in_g[dead]]
        cu, cv, cd = [], [], []
        if ng_idx.size and g_idx.size and dead_ng.size:
            t = cKDTree(xyz[g_idx])
            d, j = t.query(xyz[dead_ng], k=1, distance_upper_bound=rch)
            ok = np.isfinite(d)
            cu.append(dead_ng[ok]); cv.append(g_idx[j[ok]]); cd.append(d[ok])
        if ng_idx.size > 1:
            t = cKDTree(xyz[ng_idx])
            if dead_ng.size:
                kk = min(k_local, ng_idx.size)
                d, j = t.query(xyz[dead_ng], k=kk, distance_upper_bound=rch)
                d = d.reshape(dead_ng.size, -1); j = j.reshape(dead_ng.size, -1)
                valid = np.isfinite(d)
                jj = np.where(valid, j, 0)
                foreign = valid & (lab[ng_idx[jj]] != lab[dead_ng][:, None])
                has = foreign.any(axis=1)
                first = np.argmax(foreign, axis=1)
                rows = np.flatnonzero(has)
                cu.append(dead_ng[rows]); cv.append(ng_idx[j[rows, first[rows]]]); cd.append(d[rows, first[rows]])
            if dead_g.size:
                d, j = t.query(xyz[dead_g], k=1, distance_upper_bound=rch)
                ok = np.isfinite(d)
                cu.append(dead_g[ok]); cv.append(ng_idx[j[ok]]); cd.append(d[ok])
        if not cu:
            break
        cu = np.concatenate(cu); cv = np.concatenate(cv); cd = np.concatenate(cd)
        if cu.size == 0:
            break
        # Borůvka: every non-giant component owns its incident candidates
        own = np.concatenate([lab[cu], lab[cv]])
        idx2 = np.concatenate([np.arange(cu.size), np.arange(cu.size)])
        dd = np.concatenate([cd, cd])
        m = own != giant
        own, idx2, dd = own[m], idx2[m], dd[m]
        sel = np.unique(idx2[boruvka_select(own, own, dd)])
        a, b = cu[sel], cv[sel]
        gw = geodesic_km(lat[a], lon[a], lat[b], lon[b])
        add_u.append(a); add_v.append(b); add_w.append(gw)
        u, v, w = canonical_edges(np.concatenate([u, a]), np.concatenate([v, b]), np.concatenate([w, gw]))
        st = {"round": r + 1, "components_before": int(ncomp), "giant_nodes": int(size[giant]),
              "links_added": int(a.size), "link_km_mean": float(gw.mean()) if gw.size else 0.0,
              "link_km_max": float(gw.max()) if gw.size else 0.0}
        rstats.append(st)
        if log:
            log(f"  gap-repair round {r+1}: {st}")
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dtype=dt)
    return u, v, w, cat(add_u, np.int64), cat(add_v, np.int64), cat(add_w, np.float64), rstats
