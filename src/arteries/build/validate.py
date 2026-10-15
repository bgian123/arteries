"""Independent correctness checks for a finished build (used by scripts/validate_release.py
and the test-suite). Every check recomputes something from scratch with a
different code path and compares.

1. simplification exactness: Dijkstra on the *unsimplified* merged graph (plus
   the same connectors) vs. the packaged land matrix, for sampled sources;
2. metric sanity: symmetry, zero diagonal, triangle inequality on sampled
   triples, and road distance >= geodesic distance between terminals;
3. multimodal exactness: Dijkstra on the full multimodal graph (roads +
   connectors + sea lanes, weights a*km on land, b*km at sea, port_cost on
   every port<->sea connector) vs. Network.route_costs, for sampled cities and
   several weightings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from ..geodesy import geodesic_km
from . import graphops as G


def _raw_graph_with_connectors(work):
    z = np.load(work / "merged_raw.npz")
    keys, u, v, w = z["keys"], z["u"], z["v"], z["w"]
    lat, lon = G.unpack_keys(keys)
    t = pd.read_csv(work / "terminals.csv")
    nr = keys.size
    U, V, W = [u], [v], [w]
    real = ~t["kind"].eq("port_synthetic").to_numpy()
    tr = t[real]
    has = tr["snapped"].to_numpy(bool)
    tnode = nr + np.flatnonzero(real)[has]
    U.append(tnode); V.append(np.searchsorted(keys, tr["attach_key"].to_numpy()[has])); W.append(tr["attach_km"].to_numpy()[has])
    # synthetic land links / ports were defined on simplified-graph nodes: map by coordinates
    lg = np.load(work / "land_graph.npz")
    glat, glon = lg["lat"], lg["lon"]
    gkey = G.pack_keys(G.quantize(glat), G.quantize(glon))
    links = pd.read_csv(work / "synthetic_land_links.csv") if (work / "synthetic_land_links.csv").stat().st_size > 5 else pd.DataFrame()

    def to_raw(node_idx):
        node_idx = np.asarray(node_idx)
        out = np.empty(node_idx.size, dtype=np.int64)
        isterm = node_idx >= int(lg["n_road"])
        out[isterm] = nr + (node_idx[isterm] - int(lg["n_road"]))
        out[~isterm] = np.searchsorted(keys, gkey[node_idx[~isterm]])
        return out

    if len(links):
        U.append(to_raw(links["a"].to_numpy())); V.append(to_raw(links["b"].to_numpy())); W.append(links["km"].to_numpy())
    n = nr + len(t)
    g = coo_matrix((np.concatenate(W), (np.concatenate(U), np.concatenate(V))), shape=(n, n)).tocsr()
    term_raw = to_raw(t["node"].to_numpy())
    return g, term_raw


def check_simplification(work, L, n_sources=12, seed=0):
    g, term_raw = _raw_graph_with_connectors(work)
    rng = np.random.default_rng(seed)
    src = rng.choice(L.shape[0], size=min(n_sources, L.shape[0]), replace=False)
    d = dijkstra(g, directed=False, indices=term_raw[src])[:, term_raw]
    a, b = d, L[src].astype(np.float64)
    fin = np.isfinite(a) & np.isfinite(b)
    same_reach = bool(np.array_equal(np.isfinite(a), np.isfinite(b)))
    rel = np.abs(a[fin] - b[fin]) / np.maximum(a[fin], 1e-9)
    return {"sources": int(src.size), "same_reachability": same_reach,
            "max_abs_diff_km": float(np.abs(a[fin] - b[fin]).max()) if fin.any() else 0.0,
            "max_rel_diff": float(rel.max()) if rel.size else 0.0}


def check_metric(L, lat, lon, n_triples=200_000, seed=0):
    rng = np.random.default_rng(seed)
    T = L.shape[0]
    Ld = L.astype(np.float64)
    i, j, k = rng.integers(0, T, (3, n_triples))
    lhs, rhs = Ld[i, j], Ld[i, k] + Ld[k, j]
    ok = np.isfinite(rhs)
    tri_viol = float(np.max(lhs[ok] - rhs[ok])) if ok.any() else 0.0
    fin = np.isfinite(Ld)
    ii, jj = np.nonzero(fin)
    gc = geodesic_km(lat[ii], lon[ii], lat[jj], lon[jj])
    ratio = Ld[ii, jj] / np.maximum(gc, 1e-9)
    off = (ii != jj) & (gc > 0.01)
    return {"diag_zero": bool(np.all(np.diag(Ld) == 0)),
            "symmetric": bool(np.array_equal(np.isfinite(Ld), np.isfinite(Ld.T))),
            "max_triangle_violation_km(float32 rounding)": tri_viol,
            "min_road_over_geodesic": float(ratio[off].min()) if off.any() else None,
            "circuity_p5_p50_p95": np.quantile(ratio[off & (gc > 50)], [.05, .5, .95]).tolist() if (off & (gc > 50)).any() else None,
            "share_pairs_road_connected": float(fin.mean())}


def check_multimodal(work, net, weightings=((1, 1, 0), (4, 1, 100), (1, 3, 25)), n_cities=8, seed=0):
    """Full-graph Dijkstra vs. the terminal-graph algebra in Network.route_costs."""
    lg = np.load(work / "land_graph.npz")
    t = net.terminals
    sea = net.sea
    nl = lg["lat"].size
    ports = np.flatnonzero(t["kind"].str.startswith("port").to_numpy())
    pnode = t["node"].to_numpy()[ports] if "node" in t else pd.read_csv(work / "terminals.csv")["node"].to_numpy()[ports]
    e = t["sea_edge"].to_numpy()[ports].astype(np.int64)
    d_lane = t["sea_km_to_lane"].to_numpy()[ports]
    du = t["sea_km_u"].to_numpy()[ports]; dv = t["sea_km_v"].to_numpy()[ports]
    keep = ~np.isin(sea.passage.astype(str), ["northwest"])
    # a projection node per port on its lane edge (equivalent to the port<->u / port<->v links)
    proj = nl + sea.n + np.arange(ports.size)
    n = nl + sea.n + ports.size
    rng = np.random.default_rng(seed)
    cities = np.flatnonzero(t["kind"].eq("city").to_numpy())
    pick = rng.choice(cities, size=min(n_cities, cities.size), replace=False)
    tnodes = pd.read_csv(work / "terminals.csv")["node"].to_numpy()
    out = []
    for a, b, tau in weightings:
        U = [lg["u"], sea.u[keep] + nl, proj, proj, pnode]
        V = [lg["v"], sea.v[keep] + nl, sea.u[e] + nl, sea.v[e] + nl, proj]
        W = [a * lg["w"], b * sea.km[keep], b * (du - d_lane), b * (dv - d_lane), b * d_lane + tau]
        ok = keep[e]
        U[2], V[2], W[2] = U[2][ok], V[2][ok], W[2][ok]
        U[3], V[3], W[3] = U[3][ok], V[3][ok], W[3][ok]
        U[4], V[4], W[4] = U[4][ok], V[4][ok], W[4][ok]
        # ports projecting onto the same lane edge: direct along-lane links between projections
        alo = du - d_lane
        order = np.argsort(e, kind="stable"); es = e[order]
        bnd = np.flatnonzero(np.r_[True, es[1:] != es[:-1], True])
        for x0, x1 in zip(bnd[:-1], bnd[1:]):
            if x1 - x0 < 2 or not keep[es[x0]]:
                continue
            grp = order[x0:x1]; ii, jj = np.triu_indices(grp.size, 1)
            U.append(proj[grp[ii]]); V.append(proj[grp[jj]]); W.append(b * np.abs(alo[grp[ii]] - alo[grp[jj]]))
        g = coo_matrix((np.concatenate(W), (np.concatenate(U), np.concatenate(V))), shape=(n, n)).tocsr()
        d = dijkstra(g, directed=False, indices=tnodes[pick])[:, tnodes[pick]]
        r = net.route_costs(a, b, tau, origins=t["source_id"].to_numpy()[pick],
                            destinations=t["source_id"].to_numpy()[pick])
        c = r.cost
        fin = np.isfinite(d) & np.isfinite(c)
        out.append({"weights": [a, b, tau], "same_reachability": bool(np.array_equal(np.isfinite(d), np.isfinite(c))),
                    "max_abs_diff": float(np.abs(d[fin] - c[fin]).max()) if fin.any() else 0.0,
                    "max_rel_diff": float((np.abs(d[fin] - c[fin]) / np.maximum(d[fin], 1e-9)).max()) if fin.any() else 0.0})
    return out
