"""Build pipeline stages. Each stage reads the outputs of earlier stages from
``cfg.work`` and writes its own outputs atomically, plus a JSON stats file in
``work/stats``. Stages are idempotent and can be re-run individually.

  1 ingest      [array: one task per country]  pickle -> node/edge arrays
  2 merge       [1 task, large memory]         global node identity + gap repair
  3 terminals   [1 task]                       cities & ports, snapping
  4 simplify    [1 task, large memory]         exact leaf pruning + chain contraction
  5 assemble    [1 task]                       terminal connectors, island/port logic, sea snapping
  6 landmatrix  [array: chunks of terminals]   land-only shortest paths terminal->terminal
  7 package     [1 task]                       release files for the Python package
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..geodesy import geodesic_km, km_to_chord, segment_length_km, to_unit_xyz
from . import graphops as G
from .common import (BuildConfig, Log, country_files, env_info, iso_of, n_workers, save_npz,
                     write_json)
from .pickle_reader import read_segments


def _stats(cfg, name, obj):
    obj = dict(obj)
    obj["env"] = env_info()
    write_json(cfg.work / "stats" / f"{name}.json", obj)


# ---------------------------------------------------------------------------
# 1. ingest
# ---------------------------------------------------------------------------
def ingest(cfg: BuildConfig, iso: str, log: Log):
    path = Path(cfg.roads_dir) / f"{iso.lower()}_roads.pkl.gz"
    out = cfg.work / "countries" / f"{iso}.npz"
    la, oa, lb, ob = read_segments(path)
    links = Path(cfg.roads_dir) / f"{iso.lower()}_links.pkl.gz"
    n_links = 0
    if links.exists():  # optional *_link ramps from scripts/fetch_overpass.py
        extra = read_segments(links)
        n_links = int(extra[0].size)
        la, oa, lb, ob = (np.concatenate([x, y]) for x, y in zip((la, oa, lb, ob), extra))
        del extra
    n_raw = int(la.size)
    log(f"{iso}: {n_raw:,} stored segments ({n_links:,} from link file)")
    if n_raw == 0:
        _stats(cfg, f"ingest_{iso}", {"iso3": iso, "empty": True, "segments_stored": 0})
        return
    qa_lat, qa_lon, qb_lat, qb_lon = (G.quantize(x) for x in (la, oa, lb, ob))
    del la, oa, lb, ob
    ka = G.pack_keys(qa_lat, qa_lon)
    kb = G.pack_keys(qb_lat, qb_lon)
    del qa_lat, qa_lon, qb_lat, qb_lon
    keys = np.unique(np.concatenate([ka, kb]))
    u = np.searchsorted(keys, ka)
    v = np.searchsorted(keys, kb)
    del ka, kb
    u, v, _ = G.canonical_edges(u, v, np.zeros(u.size))
    n_cross = 0
    if cfg.insert_crossings:
        from .crossings import split_at_crossings

        lat, lon = G.unpack_keys(keys)
        keys, u, v, n_cross = split_at_crossings(lat, lon, keys, u, v, cfg.crossing_cell_deg)
        log(f"{iso}: inserted {n_cross:,} crossing junctions")
    lat, lon = G.unpack_keys(keys)
    w = segment_length_km(lat[u], lon[u], lat[v], lon[v])
    u, v, w = G.canonical_edges(u, v, w)
    ncomp, lab = G.components(keys.size, u, v)
    deg = G.degrees(keys.size, u, v)
    st = {
        "iso3": iso, "empty": False, "segments_stored": n_raw, "segments_from_link_file": n_links,
        "nodes": int(keys.size), "edges_undirected": int(u.size), "km": float(w.sum()),
        "crossing_junctions_inserted": int(n_cross), "components": int(ncomp), "giant_component_share_nodes": float(np.bincount(lab).max() / keys.size),
        "dead_ends": int((deg == 1).sum()), "junctions_deg3plus": int((deg >= 3).sum()),
        "segment_km_p50": float(np.median(w)), "segment_km_p99": float(np.quantile(w, 0.99)),
        "segment_km_max": float(w.max()),
        "bbox": [float(lat.min()), float(lon.min()), float(lat.max()), float(lon.max())],
    }
    save_npz(out, keys=keys, u=u.astype(np.int64), v=v.astype(np.int64), w=w)
    _stats(cfg, f"ingest_{iso}", st)
    log(f"{iso}: nodes={keys.size:,} edges={u.size:,} km={w.sum():,.0f} comps={ncomp:,}")


# ---------------------------------------------------------------------------
# 2. merge + gap repair
# ---------------------------------------------------------------------------
def merge(cfg: BuildConfig, log: Log):
    files = sorted((cfg.work / "countries").glob("*.npz"))
    log(f"merging {len(files)} countries")
    all_keys, owner = [], []
    for i, f in enumerate(files):
        z = np.load(f)
        all_keys.append(z["keys"]); owner.append(np.full(z["keys"].size, i, dtype=np.int32))
    cat_keys = np.concatenate(all_keys); cat_owner = np.concatenate(owner)
    del all_keys, owner
    keys, inv, counts = np.unique(cat_keys, return_inverse=True, return_counts=True)
    log(f"node keys: {cat_keys.size:,} country-level -> {keys.size:,} global "
        f"({(counts > 1).sum():,} shared across country files)")
    # land adjacency between country files (shared node coordinates)
    shared = counts[inv] > 1
    sk, so = inv[shared], cat_owner[shared]
    order = np.lexsort((so, sk)); sk, so = sk[order], so[order]
    pairs = {}
    start = 0
    names = [f.stem for f in files]
    bnd = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1], True])
    for a, b in zip(bnd[:-1], bnd[1:]):
        grp = np.unique(so[a:b])
        for x in range(grp.size):
            for y in range(x + 1, grp.size):
                p = (names[grp[x]], names[grp[y]])
                pairs[p] = pairs.get(p, 0) + 1
    pd.DataFrame([(a, b, n) for (a, b), n in sorted(pairs.items())],
                 columns=["iso3_a", "iso3_b", "shared_nodes"]).to_csv(
        cfg.work / "country_land_adjacency.csv", index=False)
    del cat_keys, cat_owner, shared, sk, so, inv, counts

    us, vs, ws = [], [], []
    for f in files:
        z = np.load(f)
        gidx = np.searchsorted(keys, z["keys"])
        us.append(gidx[z["u"]]); vs.append(gidx[z["v"]]); ws.append(z["w"])
    u, v, w = G.canonical_edges(np.concatenate(us), np.concatenate(vs), np.concatenate(ws))
    del us, vs, ws
    log(f"global raw graph: nodes={keys.size:,} edges={u.size:,} km={w.sum():,.0f}")
    lat, lon = G.unpack_keys(keys)
    ncomp0, _ = G.components(keys.size, u, v)
    u, v, w, bu, bv, bw, rstats = G.bridge_components(
        lat, lon, u, v, w, cfg.bridge_km, rounds=cfg.bridge_rounds, log=log)
    ncomp1, lab = G.components(keys.size, u, v)
    size = np.bincount(lab)
    save_npz(cfg.work / "merged_raw.npz", keys=keys, u=u, v=v, w=w,
             bridge_u=bu, bridge_v=bv, bridge_w=bw)
    _stats(cfg, "merge", {
        "countries": names, "nodes": int(keys.size), "edges": int(u.size), "km": float(w.sum()),
        "country_pairs_sharing_nodes": len(pairs),
        "components_before_repair": int(ncomp0), "components_after_repair": int(ncomp1),
        "largest_component_nodes": int(size.max()), "gap_repair_rounds": rstats,
        "gap_links_total": int(bu.size), "gap_links_km_total": float(bw.sum()),
    })


# ---------------------------------------------------------------------------
# 3. terminals: cities and ports
# ---------------------------------------------------------------------------
def load_cities(cfg: BuildConfig) -> pd.DataFrame:
    c = pd.read_csv(cfg.cities_csv)
    c = c[c["year"] <= cfg.city_max_year]
    c = c.sort_values("year").groupby(["iso3", "city", "lat", "lon"], as_index=False).tail(1)
    c = c[c["population"] >= cfg.city_min_population].copy()
    c = c.sort_values(["iso3", "city", "lat"]).reset_index(drop=True)
    sid = c["iso3"] + ":" + c["city"]
    dup = sid.groupby(sid).cumcount()
    sid = sid.where(dup == 0, sid + "#" + (dup + 1).astype(str))  # e.g. two US "Concord"s
    return pd.DataFrame({
        "kind": "city", "source_id": sid, "name": c["city"],
        "iso3": c["iso3"], "lat": c["lat"].astype(float), "lon": c["lon"].astype(float),
        "population": c["population"].astype(float), "pop_year": c["year"].astype(int),
        "pop_source": c["source_used"],
    })


def load_ports(cfg: BuildConfig) -> pd.DataFrame:
    p = pd.read_csv(cfg.ports_csv, encoding="utf-8-sig")
    if cfg.exclude_offshore_ports:
        off = p["portname"].str.contains("Offshore", case=False, na=False)
        p = p[~off]
    return pd.DataFrame({
        "kind": "port", "source_id": p["portid"].astype(str), "name": p["portname"],
        "iso3": p["ISO3"], "lat": p["lat"].astype(float), "lon": p["lon"].astype(float),
        "vessel_count_total": p["vessel_count_total"].astype(float),
    }).reset_index(drop=True)


def _snap(lat, lon, node_lat, node_lon, pool_idx, max_km):
    """Nearest node among ``pool_idx`` within max_km; returns (node or -1, km)."""
    from scipy.spatial import cKDTree

    if pool_idx.size == 0:
        return np.full(lat.size, -1), np.full(lat.size, np.inf)
    t = cKDTree(to_unit_xyz(node_lat[pool_idx], node_lon[pool_idx]))
    d, j = t.query(to_unit_xyz(lat, lon), k=1, distance_upper_bound=km_to_chord(max_km))
    ok = np.isfinite(d)
    node = np.where(ok, pool_idx[np.where(ok, j, 0)], -1)
    km = np.full(lat.size, np.inf)
    km[ok] = geodesic_km(lat[ok], lon[ok], node_lat[node[ok]], node_lon[node[ok]])
    return node, km


def terminals(cfg: BuildConfig, log: Log):
    z = np.load(cfg.work / "merged_raw.npz")
    keys, u, v, w = z["keys"], z["u"], z["v"], z["w"]
    lat, lon = G.unpack_keys(keys)
    ncomp, lab = G.components(keys.size, u, v)
    comp_km = np.bincount(lab[u], weights=w, minlength=ncomp)
    has_edge = np.zeros(keys.size, bool); has_edge[u] = True; has_edge[v] = True
    # snapping preference: nodes of "major" components (>= 10 km of road), so a
    # city is not attached to a 200-m fragment that happens to be closest.
    major = has_edge & (comp_km[lab] >= 10.0)
    major_idx = np.flatnonzero(major)
    any_idx = np.flatnonzero(has_edge)

    cities = load_cities(cfg)
    ports = load_ports(cfg)
    if cfg.terminal_iso3_filter:
        keep = set(cfg.terminal_iso3_filter.upper().split(","))
        cities = cities[cities["iso3"].isin(keep)].reset_index(drop=True)
        ports = ports[ports["iso3"].isin(keep)].reset_index(drop=True)
    log(f"cities (pop>={cfg.city_min_population:,.0f}, year<={cfg.city_max_year}): {len(cities):,}; "
        f"ports (non-offshore): {len(ports):,}")
    out = []
    for df, max_km in ((cities, cfg.city_snap_km), (ports, cfg.port_snap_km)):
        la, lo = df["lat"].to_numpy(), df["lon"].to_numpy()
        n1, d1 = _snap(la, lo, lat, lon, major_idx, max_km)
        n2, d2 = _snap(la, lo, lat, lon, any_idx, max_km)
        use2 = (n1 < 0) & (n2 >= 0)
        node = np.where(use2, n2, n1); km = np.where(use2, d2, d1)
        df = df.copy()
        # NOTE: packed keys are negative for southern latitudes, so "not snapped" must be an
        # explicit flag, never a sign test on attach_key (bug in the first full build).
        df["snapped"] = node >= 0
        df["attach_key"] = np.where(node >= 0, keys[np.maximum(node, 0)], 0)
        df["attach_km"] = np.where(node >= 0, km, np.nan)
        df["attach_minor_component"] = use2
        out.append(df)
        log(f"  {df['kind'].iat[0]}s snapped: {(node >= 0).sum():,}/{len(df):,} "
            f"(median {np.nanmedian(df['attach_km']):.3f} km, p95 {np.nanquantile(df['attach_km'], .95):.2f} km; "
            f"minor-component fallbacks {use2.sum()})")
    t = pd.concat(out, ignore_index=True)
    t.insert(0, "terminal_id", np.arange(len(t)))
    t.to_csv(cfg.work / "terminals_snapped.csv", index=False)
    _stats(cfg, "terminals", {
        "cities": int((t.kind == "city").sum()), "ports": int((t.kind == "port").sum()),
        "unsnapped_cities": t[(t.kind == "city") & ~t.snapped][["name", "iso3"]].values.tolist(),
        "unsnapped_ports": t[(t.kind == "port") & ~t.snapped][["name", "iso3"]].values.tolist(),
        "attach_km_quantiles_cities": t[t.kind == "city"]["attach_km"].quantile([.5, .9, .99, 1]).tolist(),
        "attach_km_quantiles_ports": t[t.kind == "port"]["attach_km"].quantile([.5, .9, .99, 1]).tolist(),
    })


# ---------------------------------------------------------------------------
# 4. simplify
# ---------------------------------------------------------------------------
def simplify(cfg: BuildConfig, log: Log):
    z = np.load(cfg.work / "merged_raw.npz")
    keys, u, v, w = z["keys"], z["u"], z["v"], z["w"]
    t = pd.read_csv(cfg.work / "terminals_snapped.csv")
    att = t["attach_key"].to_numpy()[t["snapped"].to_numpy(bool)]
    protected = np.isin(keys, att)
    log(f"simplify: nodes={keys.size:,} edges={u.size:,} protected={protected.sum():,}")
    k2, u2, v2, w2, p2, st = G.simplify(keys, u, v, w, protected, log=log)
    if not np.all(np.isin(att, k2)):
        raise RuntimeError("a protected attachment node was removed")
    save_npz(cfg.work / "land_simplified.npz", keys=k2, u=u2, v=v2, w=w2)
    st["reduction_nodes"] = 1 - st["nodes_out"] / max(st["nodes_in"], 1)
    st["reduction_edges"] = 1 - st["edges_out"] / max(st["edges_in"], 1)
    st["km_in_components_kept"] = float(w2.sum())
    _stats(cfg, "simplify", st)
    log(f"simplified: nodes {st['nodes_in']:,}->{st['nodes_out']:,}, edges {st['edges_in']:,}->{st['edges_out']:,}")


# ---------------------------------------------------------------------------
# 5. assemble
# ---------------------------------------------------------------------------
def assemble(cfg: BuildConfig, log: Log):
    from ..sea import load_marnet, snap_points_to_sea
    from scipy.spatial import cKDTree

    z = np.load(cfg.work / "land_simplified.npz")
    keys, u, v, w = z["keys"], z["u"], z["v"], z["w"]
    nr = keys.size
    rlat, rlon = G.unpack_keys(keys)
    t = pd.read_csv(cfg.work / "terminals_snapped.csv")
    nt = len(t)
    # terminal nodes are appended after road nodes
    tnode = nr + np.arange(nt)
    att = t["attach_key"].to_numpy()
    has = t["snapped"].to_numpy(bool)
    att_idx = np.searchsorted(keys, att[has])
    if not np.array_equal(keys[np.minimum(att_idx, keys.size - 1)], att[has]):
        raise RuntimeError("terminal attachment node missing from simplified graph")
    cu = tnode[has]; cv = att_idx; cw = t["attach_km"].to_numpy()[has]
    edge_kind = [np.zeros(u.size, np.int8), np.full(cu.size, 1, np.int8)]  # 0 road, 1 terminal connector
    U = np.concatenate([u, cu]); V = np.concatenate([v, cv]); W = np.concatenate([w, cw])
    nlat = np.concatenate([rlat, t["lat"].to_numpy()]); nlon = np.concatenate([rlon, t["lon"].to_numpy()])
    n = nr + nt

    # ---- island / port logic ------------------------------------------------
    sea = load_marnet()
    sea_tree = cKDTree(to_unit_xyz(sea.lat, sea.lon))
    is_port = (t["kind"] == "port").to_numpy()
    synth_rows, link_rows = [], []
    for it in range(50):
        ncomp, lab = G.components(n, U, V)
        port_comp = np.zeros(ncomp, bool); port_comp[lab[tnode[is_port]]] = True
        for r in synth_rows:
            port_comp[lab[r["node"]]] = True
        city_comps = np.unique(lab[tnode[~is_port]])
        orphan = city_comps[~port_comp[city_comps]]
        if orphan.size == 0:
            break
        pc_nodes = np.flatnonzero(port_comp[lab])
        pc_tree = cKDTree(to_unit_xyz(nlat[pc_nodes], nlon[pc_nodes]))
        order = np.argsort(lab, kind="stable"); sl = lab[order]
        for c in orphan:
            members = order[np.searchsorted(sl, c):np.searchsorted(sl, c, side="right")]
            xyz = to_unit_xyz(nlat[members], nlon[members])
            dl, jl = pc_tree.query(xyz, k=1)
            ds, js = sea_tree.query(xyz, k=1)
            i_l = int(np.argmin(dl)); i_s = int(np.argmin(ds))
            a = members[i_l]; b = pc_nodes[jl[i_l]]
            km_land = float(geodesic_km(nlat[a], nlon[a], nlat[b], nlon[b]))
            m_s = members[i_s]
            km_sea = float(geodesic_km(nlat[m_s], nlon[m_s], sea.lat[js[i_s]], sea.lon[js[i_s]]))
            if km_sea < km_land:
                synth_rows.append({"node": int(m_s), "lat": nlat[m_s], "lon": nlon[m_s],
                                   "nearest_sea_km": km_sea, "nearest_land_km": km_land})
            else:
                link_rows.append({"a": int(a), "b": int(b), "km": km_land,
                                  "from_lat": nlat[a], "from_lon": nlon[a], "to_lat": nlat[b], "to_lon": nlon[b]})
                U = np.append(U, a); V = np.append(V, b); W = np.append(W, km_land)
                edge_kind.append(np.full(1, 2, np.int8))  # 2 synthetic land link
        log(f"  island pass {it+1}: orphan city components={orphan.size}, "
            f"synthetic ports={len(synth_rows)}, synthetic land links={len(link_rows)}")
    edge_kind = np.concatenate(edge_kind)

    # synthetic ports become extra terminals located on an existing land node
    if synth_rows:
        sp = pd.DataFrame(synth_rows)
        extra = pd.DataFrame({
            "terminal_id": nt + np.arange(len(sp)), "kind": "port_synthetic",
            "source_id": [f"synthetic_{i}" for i in range(len(sp))], "name": "synthetic port",
            "iso3": "", "lat": sp["lat"], "lon": sp["lon"],
        })
        t = pd.concat([t, extra], ignore_index=True)
        term_node = np.concatenate([tnode, sp["node"].to_numpy()])
    else:
        term_node = tnode
    pd.DataFrame(link_rows).to_csv(cfg.work / "synthetic_land_links.csv", index=False)

    # ---- sea attachment for all ports --------------------------------------
    port_mask = t["kind"].str.startswith("port").to_numpy()
    pe, pd_, pdu, pdv = snap_points_to_sea(sea, t.loc[port_mask, "lat"].to_numpy(), t.loc[port_mask, "lon"].to_numpy())
    t["sea_edge"] = -1; t["sea_km_to_lane"] = np.nan; t["sea_km_u"] = np.nan; t["sea_km_v"] = np.nan
    t.loc[port_mask, "sea_edge"] = pe
    t.loc[port_mask, "sea_km_to_lane"] = pd_
    t.loc[port_mask, "sea_km_u"] = pd_ + pdu
    t.loc[port_mask, "sea_km_v"] = pd_ + pdv
    t["node"] = term_node
    ncomp, lab = G.components(n, U, V)
    t["land_component"] = lab[term_node]
    t.to_csv(cfg.work / "terminals.csv", index=False)
    import shutil

    shutil.rmtree(cfg.work / "landmatrix", ignore_errors=True)  # invalidate stale SSSP rows
    save_npz(cfg.work / "land_graph.npz", lat=nlat, lon=nlon, u=U, v=V, w=W, kind=edge_kind,
             n_road=np.array(nr))
    _stats(cfg, "assemble", {
        "nodes": int(n), "edges": int(U.size), "terminals": int(len(t)),
        "synthetic_ports": len(synth_rows), "synthetic_land_links": len(link_rows),
        "synthetic_land_link_km": [r["km"] for r in link_rows],
        "land_components_with_terminals": int(np.unique(lab[term_node]).size),
        "port_sea_snap_km_quantiles": np.quantile(pd_, [.5, .9, .99, 1]).tolist() if pd_.size else [],
    })
    log(f"assembled: nodes={n:,} edges={U.size:,} terminals={len(t):,}")


# ---------------------------------------------------------------------------
# 6. land distance matrix (terminal -> terminal, land only)
# ---------------------------------------------------------------------------
_GRAPH = None
_TARGETS = None


def _sssp(src):
    from scipy.sparse.csgraph import dijkstra

    d = dijkstra(_GRAPH, directed=False, indices=int(src))
    return d[_TARGETS].astype(np.float64)


def _land_csr(cfg):
    from scipy.sparse import coo_matrix

    z = np.load(cfg.work / "land_graph.npz")
    n = z["lat"].size
    g = coo_matrix((z["w"], (z["u"], z["v"])), shape=(n, n)).tocsr()
    g.sum_duplicates()
    return g


def n_landmatrix_tasks(cfg: BuildConfig) -> int:
    t = pd.read_csv(cfg.work / "terminals.csv")
    return math.ceil(len(t) / cfg.sssp_chunk)


def landmatrix(cfg: BuildConfig, task: int, log: Log):
    import multiprocessing as mp

    global _GRAPH, _TARGETS
    t = pd.read_csv(cfg.work / "terminals.csv")
    nodes = t["node"].to_numpy()
    lo, hi = task * cfg.sssp_chunk, min((task + 1) * cfg.sssp_chunk, len(t))
    if lo >= len(t):
        log(f"task {task}: beyond {len(t)} terminals; nothing to do"); return
    out = cfg.work / "landmatrix" / f"rows_{lo:06d}_{hi:06d}.npy"
    if out.exists():
        log(f"{out.name} exists; skipping"); return
    _GRAPH = _land_csr(cfg)
    _TARGETS = nodes
    nw = n_workers(cfg)
    log(f"task {task}: sources {lo}..{hi-1} on {nw} workers; graph n={_GRAPH.shape[0]:,} m={_GRAPH.nnz:,}")
    srcs = nodes[lo:hi]
    if nw > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(nw) as pool:
            rows = pool.map(_sssp, srcs, chunksize=1)
    else:
        rows = [_sssp(s) for s in srcs]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, np.vstack(rows))
    tmp.replace(out)
    log(f"wrote {out.name}")


# ---------------------------------------------------------------------------
# 7. package
# ---------------------------------------------------------------------------
def package(cfg: BuildConfig, log: Log, version: str = "0.1.0"):
    t = pd.read_csv(cfg.work / "terminals.csv", keep_default_na=False, na_values=[""])
    T = len(t)
    L = np.full((T, T), np.inf)
    files = sorted((cfg.work / "landmatrix").glob("rows_*.npy"))
    covered = np.zeros(T, bool)
    g_mtime = (cfg.work / "land_graph.npz").stat().st_mtime
    for f in files:
        if f.stat().st_mtime < g_mtime:
            raise RuntimeError(f"{f.name} predates land_graph.npz; re-run the landmatrix stage")
        lo, hi = (int(x) for x in f.stem.split("_")[1:3])
        L[lo:hi] = np.load(f); covered[lo:hi] = True
    if not covered.all():
        raise RuntimeError(f"land matrix incomplete: {(~covered).sum()} rows missing")
    fin = np.isfinite(L) & np.isfinite(L.T)
    asym = np.abs(L - L.T)[fin]
    max_asym = float(asym.max()) if asym.size else 0.0
    log(f"land matrix {T}x{T}: finite share {np.isfinite(L).mean():.3f}, max |L-L'| = {max_asym:.3e} km")
    if max_asym > 1e-6:
        raise RuntimeError("land matrix is not symmetric; graph or SSSP problem")
    L = np.minimum(L, L.T).astype(np.float32)
    # also record the straight-line distance for every pair (for circuity diagnostics)
    rel = cfg.work / "release"
    rel.mkdir(parents=True, exist_ok=True)
    cols = {c: t[c].to_numpy() for c in ["kind", "source_id", "name", "iso3"]}
    cols = {k: np.asarray(v, dtype="U") for k, v in cols.items()}
    num = {c: t[c].to_numpy(dtype=np.float64) for c in
           ["lat", "lon", "population", "attach_km", "sea_edge", "sea_km_to_lane", "sea_km_u", "sea_km_v",
            "land_component"] if c in t}
    np.savez_compressed(rel / f"arteries_core_v{version}.npz", L=L, **{f"t_{k}": v for k, v in cols.items()},
                        **{f"t_{k}": v for k, v in num.items()}, version=np.array(version))
    z = np.load(cfg.work / "land_graph.npz")
    np.savez_compressed(rel / f"arteries_landgraph_v{version}.npz", **{k: z[k] for k in z.files})
    t.to_csv(rel / f"terminals_v{version}.csv", index=False)
    _stats(cfg, "package", {"terminals": T, "max_asymmetry_km": max_asym,
                            "finite_share": float(np.isfinite(L).mean()), "version": version})
    log(f"release written to {rel}")
