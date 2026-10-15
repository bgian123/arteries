"""Unit tests for the exactness-critical pieces. Run:  pytest -q tests"""
import gzip
import pickle

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from arteries._kernels import HAVE_NUMBA, floyd_warshall, minplus
from arteries.build import graphops as G
from arteries.build.crossings import split_at_crossings
from arteries.build.pickle_reader import read_segments
from arteries.geodesy import geodesic_km, haversine_km, segment_length_km


def _sp(n, u, v, w, src):
    g = coo_matrix((w, (u, v)), shape=(n, n)).tocsr()
    return dijkstra(g, directed=False, indices=src)


def _random_road_graph(rng, n_junction=60, chain=12):
    """Junctions joined by long degree-2 chains + dangling trees + parallel edges."""
    pts = rng.uniform(0, 1, (n_junction, 2))
    U, V, W, nxt = [], [], [], n_junction
    for i in range(n_junction):
        for j in rng.choice(n_junction, 2, replace=False):
            if i == j:
                continue
            k = rng.integers(1, chain)
            path = [i] + list(range(nxt, nxt + k)) + [j]
            nxt += k
            for a, b in zip(path[:-1], path[1:]):
                U.append(a); V.append(b); W.append(rng.uniform(0.01, 1))
    for _ in range(40):  # dangling trees
        root = rng.integers(0, nxt)
        for d in range(rng.integers(1, 6)):
            U.append(root); V.append(nxt); W.append(rng.uniform(0.01, 1)); root = nxt; nxt += 1
    return nxt, np.array(U), np.array(V), np.array(W)


@pytest.mark.parametrize("seed", range(5))
def test_simplify_preserves_distances(seed):
    rng = np.random.default_rng(seed)
    n, u, v, w = _random_road_graph(rng)
    keys = np.arange(n, dtype=np.int64)
    prot = np.zeros(n, bool)
    prot[rng.choice(n, 25, replace=False)] = True
    k2, u2, v2, w2, p2, st = G.simplify(keys, u, v, w, prot)
    assert st["nodes_out"] < n
    P = np.flatnonzero(prot)
    d_full = _sp(n, u, v, w, P)[:, P]
    idx = np.searchsorted(k2, P)
    assert np.array_equal(k2[idx], P)
    d_simp = _sp(k2.size, u2, v2, w2, idx)[:, idx]
    assert np.array_equal(np.isfinite(d_full), np.isfinite(d_simp))
    f = np.isfinite(d_full)
    assert np.allclose(d_full[f], d_simp[f], rtol=1e-12, atol=1e-9)


def test_pickle_reader_matches_pickle(tmp_path):
    rng = np.random.default_rng(1)
    pts = [tuple(float(x) for x in np.round(p, 6)) for p in rng.uniform(-80, 80, (500, 2))]
    segs = set()
    for a, b in zip(pts[:-1], pts[1:]):
        segs.add((a, b)); segs.add((b, a))  # shared tuples -> memo GETs
    for i in range(0, 400, 7):
        segs.add(((pts[i][0], pts[i][1]), (pts[i + 3][0], pts[i + 3][1])))  # fresh tuple objects
    for proto in (2, 4, 5):
        f = tmp_path / f"x{proto}.pkl.gz"
        with gzip.open(f, "wb") as fh:
            pickle.dump(segs, fh, protocol=proto)
        a = read_segments(f)
        got = set(zip(zip(a[0], a[1]), zip(a[2], a[3])))
        assert got == segs


def test_crossings_split_x_shape():
    lat = np.array([0.0, 1.0, 0.0, 1.0]); lon = np.array([0.0, 1.0, 1.0, 0.0])
    keys = G.pack_keys(G.quantize(lat), G.quantize(lon))
    order = np.argsort(keys); keys = keys[order]
    inv = np.empty(4, int); inv[order] = np.arange(4)
    u = inv[np.array([0, 2])]; v = inv[np.array([1, 3])]
    lat2, lon2 = G.unpack_keys(keys)
    k2, u2, v2, n = split_at_crossings(lat2, lon2, keys, u, v, cell_deg=0.3)
    assert n == 1 and k2.size == 5 and u2.size == 4
    la, lo = G.unpack_keys(k2)
    c = np.flatnonzero((np.abs(la - 0.5) < 1e-6) & (np.abs(lo - 0.5) < 1e-6))
    assert c.size == 1 and np.sum((u2 == c[0]) | (v2 == c[0])) == 4


def test_segment_length_vs_geodesic():
    rng = np.random.default_rng(0)
    lat = rng.uniform(-70, 70, 1000); lon = rng.uniform(-180, 180, 1000)
    dlat = rng.uniform(-0.01, 0.01, 1000); dlon = rng.uniform(-0.01, 0.01, 1000)
    a = segment_length_km(lat, lon, lat + dlat, lon + dlon)
    b = geodesic_km(lat, lon, lat + dlat, lon + dlon)
    assert np.max(np.abs(a - b) / b) < 1e-5


def test_haversine_known():
    # JFK -> LHR great-circle ~ 5,540 km
    assert abs(haversine_km(40.6413, -73.7781, 51.4700, -0.4543) - 5540) < 15


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_kernels_numba_equals_numpy():
    rng = np.random.default_rng(3)
    P = 40
    D = rng.uniform(1, 100, (P, P)); D = np.minimum(D, D.T); D[rng.uniform(size=(P, P)) < .3] = np.inf
    np.fill_diagonal(D, 0)
    Lk = np.where(np.isfinite(D), D, 0); Sk = np.zeros_like(D); Nl = np.zeros((P, P), np.int32)
    r1 = floyd_warshall(D.copy(), Lk.copy(), Sk.copy(), Nl.copy(), True)
    r2 = floyd_warshall(D.copy(), Lk.copy(), Sk.copy(), Nl.copy(), False)
    for x, y in zip(r1, r2):
        assert np.allclose(np.nan_to_num(x, posinf=-1), np.nan_to_num(y, posinf=-1))
    A = rng.uniform(1, 50, (7, P)); A[A > 40] = np.inf
    Z = np.zeros_like(A)
    m1 = minplus(A, Z, Z, np.zeros(A.shape, np.int32), r1[0], r1[1], r1[2], r1[3], True)
    m2 = minplus(A, Z, Z, np.zeros(A.shape, np.int32), r1[0], r1[1], r1[2], r1[3], False)
    for x, y in zip(m1, m2):
        assert np.allclose(np.nan_to_num(x, posinf=-1), np.nan_to_num(y, posinf=-1))


def test_marnet_connected():
    from scipy.sparse.csgraph import connected_components

    from arteries.sea import load_marnet

    s = load_marnet()
    n, _ = connected_components(coo_matrix((s.km, (s.u, s.v)), shape=(s.n, s.n)), directed=False)
    assert n == 1 and np.all(s.km > 0)
