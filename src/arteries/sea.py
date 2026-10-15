"""Maritime network (MARNET as distributed with searoute-py 1.6.0).

Attribution: maritime graph from searoute-py (Apache-2.0, Gent Halili), itself
derived from Eurostat's SeaRoute / MARNET (based on the Oak Ridge National
Laboratory CTA global shipping-lane network). See data/MARNET_LICENSE_*.txt
and scripts/make_marnet.py for the cleaning applied.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import numpy as np

from .geodesy import geodesic_km, to_unit_xyz

PASSAGES = ("babalmandab", "bering", "bosporus", "chili", "dardanelles", "gibraltar", "malacca",
            "northwest", "ormuz", "panama", "south_africa", "suez", "sunda")
DEFAULT_CLOSED = ("northwest",)  # same default as searoute: no Arctic Northwest Passage


@dataclass
class SeaGraph:
    lat: np.ndarray
    lon: np.ndarray
    u: np.ndarray
    v: np.ndarray
    km: np.ndarray
    passage: np.ndarray

    @property
    def n(self) -> int:
        return self.lat.size


def load_marnet(path=None) -> SeaGraph:
    if path is None:
        with resources.files("arteries").joinpath("data/marnet.npz").open("rb") as fh:
            z = np.load(fh)
            d = {k: z[k] for k in z.files}
    else:
        z = np.load(path)
        d = {k: z[k] for k in z.files}
    return SeaGraph(d["lat"], d["lon"], d["u"], d["v"], d["km"], d["passage"])


def snap_points_to_sea(sea: SeaGraph, lat, lon, k: int = 24):
    """Attach points (ports) to the sea graph by orthogonal projection onto the
    nearest sea edge.

    Candidate edges are those incident to the k nearest sea nodes. Each point is
    projected onto every candidate edge in a local equirectangular frame
    centered on the point (exact enough for selecting the nearest edge; all
    returned lengths are WGS84 geodesics). The point is then linked to both end
    nodes of the chosen edge with lengths d(point, proj) + d(proj, end), which
    is equivalent to inserting the projection as a node on that edge.

    Returns (edge_idx, km_to_projection, km_to_u, km_to_v).
    """
    from scipy.spatial import cKDTree

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    tree = cKDTree(to_unit_xyz(sea.lat, sea.lon))
    kk = min(k, sea.n)
    _, near = tree.query(to_unit_xyz(lat, lon), k=kk)
    near = near.reshape(lat.size, -1)
    # incidence lists
    order = np.argsort(np.r_[sea.u, sea.v], kind="stable")
    ends = np.r_[sea.u, sea.v][order]
    eid = np.r_[np.arange(sea.u.size), np.arange(sea.u.size)][order]
    ptr = np.searchsorted(ends, np.arange(sea.n + 1))

    out_e = np.empty(lat.size, dtype=np.int64)
    out_d = np.empty(lat.size)
    out_du = np.empty(lat.size)
    out_dv = np.empty(lat.size)
    for i in range(lat.size):
        cand = np.unique(np.concatenate([eid[ptr[j]:ptr[j + 1]] for j in near[i]]))
        cl = max(np.cos(np.radians(lat[i])), 1e-6)
        la, lb = sea.lat[sea.u[cand]], sea.lat[sea.v[cand]]
        oa = ((sea.lon[sea.u[cand]] - lon[i] + 180.0) % 360.0) - 180.0
        ob = ((sea.lon[sea.v[cand]] - lon[i] + 180.0) % 360.0) - 180.0
        ax, ay = oa * cl, la - lat[i]
        bx, by = ob * cl, lb - lat[i]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = np.clip(-(ax * dx + ay * dy) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
        px, py = ax + t * dx, ay + t * dy
        j = int(np.argmin(px * px + py * py))
        e = int(cand[j])
        plat, plon = lat[i] + py[j], lon[i] + px[j] / cl
        out_e[i] = e
        out_d[i] = float(geodesic_km(lat[i], lon[i], plat, plon))
        out_du[i] = float(geodesic_km(plat, plon, sea.lat[sea.u[e]], sea.lon[sea.u[e]]))
        out_dv[i] = float(geodesic_km(plat, plon, sea.lat[sea.v[e]], sea.lon[sea.v[e]]))
    return out_e, out_d, out_du, out_dv
