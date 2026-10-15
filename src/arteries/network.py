"""User-facing API.

>>> import arteries
>>> net = arteries.load()                         # downloads the release data once
>>> res = net.route_costs(land_cost=3.0, sea_cost=1.0, port_cost=500.0,
...                       closed_passages=["suez", "babalmandab"])
>>> res.cost          # (n_cities x n_cities) generalized cost
>>> res.land_km       # land km on the cost-minimizing route
>>> res.sea_km        # sea km on the cost-minimizing route
>>> res.to_frame()    # long format: origin, destination, cost, land_km, sea_km, sea_legs

Model
-----
A route alternates land legs and sea legs and switches mode only at ports.
Its generalized cost is

    land_cost * (land km) + sea_cost * (sea km) + port_cost * (number of port calls)

where every sea leg involves two port calls (loading + unloading). Because each
land leg runs between two *terminals* (cities or ports) and uses roads only,
its best length is the land-only shortest-path distance between those
terminals, which is precomputed once on the full road network (matrix ``L``).
The lowest-cost route for any (land_cost, sea_cost, port_cost) is therefore
obtained exactly on a small terminal graph:

1. sea distances between all ports, S = Dijkstra on the maritime graph with
   closed passages removed;
2. port-to-port closure: D = Floyd-Warshall on min(land_cost*L, 2*port_cost + sea_cost*S);
3. city-to-city: min( land_cost*L[c,c'],  min_{p,q} land_cost*L[c,p] + D[p,q] + land_cost*L[q,c'] ).

Steps 2-3 are min-plus products, computed with Numba in parallel. The result is
identical to running Dijkstra on the full multimodal graph with the same weights.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ._kernels import floyd_warshall, minplus
from .sea import DEFAULT_CLOSED, PASSAGES, SeaGraph, load_marnet

DATA_VERSION = "0.1.0"

# Named sea regions that can be closed as a whole (lat_min, lon_min, lat_max, lon_max).
# Closing only the 'suez' and 'babalmandab' passage edges still lets routes call at
# ports *inside* the Red Sea and bridge the straits by land (e.g. Aden->Mokha,
# Eilat->Ashkelon). Closing the region removes every lane and port connection in it.
SEA_REGIONS = {
    "red_sea": (12.45, 32.0, 30.5, 43.6),
    "persian_gulf": (23.5, 47.5, 30.6, 56.6),   # west of the Strait of Hormuz
    "black_sea": (40.8, 27.3, 47.3, 42.0),
    "baltic_sea": (53.5, 9.5, 66.0, 30.5),
}
DEFAULT_URL = ("https://github.com/bgian123/arteries/releases/download/"
               f"data-v{DATA_VERSION}/arteries_core_v{DATA_VERSION}.npz")


def data_dir() -> Path:
    d = Path(os.environ.get("ARTERIES_DATA", Path.home() / ".cache" / "arteries"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(path: str | os.PathLike | None = None, url: str = DEFAULT_URL) -> "Network":
    """Load the precomputed network. Downloads it to ``~/.cache/arteries`` on first use
    (override the folder with the ARTERIES_DATA environment variable)."""
    if path is None:
        path = data_dir() / Path(url).name
        if not Path(path).exists():
            print(f"arteries: downloading {url} -> {path}")
            tmp = Path(str(path) + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
    return Network.from_file(path)


@dataclass
class RouteResult:
    origins: pd.DataFrame
    destinations: pd.DataFrame
    cost: np.ndarray
    land_km: np.ndarray
    sea_km: np.ndarray
    sea_legs: np.ndarray
    params: dict

    @property
    def total_km(self) -> np.ndarray:
        return self.land_km + self.sea_km

    def to_frame(self, drop_self: bool = True) -> pd.DataFrame:
        no, nd = len(self.origins), len(self.destinations)
        i = np.repeat(np.arange(no), nd)
        j = np.tile(np.arange(nd), no)
        df = pd.DataFrame({
            "origin": self.origins["name"].to_numpy()[i],
            "origin_iso3": self.origins["iso3"].to_numpy()[i],
            "origin_id": self.origins["source_id"].to_numpy()[i],
            "destination": self.destinations["name"].to_numpy()[j],
            "destination_iso3": self.destinations["iso3"].to_numpy()[j],
            "destination_id": self.destinations["source_id"].to_numpy()[j],
            "cost": self.cost.ravel(), "land_km": self.land_km.ravel(), "sea_km": self.sea_km.ravel(),
            "total_km": self.total_km.ravel(), "sea_legs": self.sea_legs.ravel(),
        })
        if drop_self:
            df = df[df["origin_id"] != df["destination_id"]]
        return df.reset_index(drop=True)


class Network:
    def __init__(self, terminals: pd.DataFrame, L: np.ndarray, sea: SeaGraph):
        self.terminals = terminals.reset_index(drop=True)
        self.L = L.astype(np.float64)
        self.sea = sea
        kind = self.terminals["kind"].to_numpy()
        self._city = np.flatnonzero(kind == "city")
        self._port = np.flatnonzero(np.char.startswith(kind.astype(str), "port"))
        self._sea_cache: dict = {}

    # ------------------------------------------------------------------ io
    @classmethod
    def from_file(cls, path) -> "Network":
        z = np.load(path, allow_pickle=False)
        cols = {k[2:]: z[k] for k in z.files if k.startswith("t_")}
        t = pd.DataFrame(cols)
        for c in ("kind", "source_id", "name", "iso3"):
            t[c] = t[c].astype(str)
        return cls(t, z["L"], load_marnet())

    @property
    def cities(self) -> pd.DataFrame:
        return self.terminals.iloc[self._city]

    @property
    def ports(self) -> pd.DataFrame:
        return self.terminals.iloc[self._port]

    # ------------------------------------------------------------ sea legs
    def sea_matrix(self, closed_passages=DEFAULT_CLOSED, sea_edge_multiplier=None,
                   closed_regions=()) -> np.ndarray:
        """Port x port shortest sea distance (km) with some passages closed.

        ``sea_edge_multiplier`` (optional, length = #sea edges) rescales individual
        sea edges, e.g. to model piracy risk premia or ice; the returned matrix is
        then in scaled km.
        ``closed_regions``: names from SEA_REGIONS or explicit boxes
        (lat_min, lon_min, lat_max, lon_max); every sea lane with an end inside a box
        is removed, so ports in the region lose their sea connection.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        closed = tuple(sorted(set(closed_passages or ())))
        bad = set(closed) - set(PASSAGES)
        if bad:
            raise ValueError(f"unknown passages {bad}; valid: {PASSAGES}")
        boxes = []
        for r in closed_regions or ():
            if isinstance(r, str):
                if r not in SEA_REGIONS:
                    raise ValueError(f"unknown region {r!r}; named regions: {sorted(SEA_REGIONS)}")
                boxes.append(SEA_REGIONS[r])
            else:
                boxes.append(tuple(float(x) for x in r))
        key = (closed, tuple(boxes),
               None if sea_edge_multiplier is None else hash(np.asarray(sea_edge_multiplier).tobytes()))
        if key in self._sea_cache:
            return self._sea_cache[key]
        s = self.sea
        keep = ~np.isin(s.passage.astype(str), closed)
        for (la0, lo0, la1, lo1) in boxes:
            inside = (s.lat >= la0) & (s.lat <= la1) & (s.lon >= lo0) & (s.lon <= lo1)
            keep &= ~(inside[s.u] | inside[s.v])
        km = s.km.copy()
        if sea_edge_multiplier is not None:
            km = km * np.asarray(sea_edge_multiplier, dtype=float)
        P = self._port.size
        t = self.terminals.iloc[self._port]
        e = t["sea_edge"].to_numpy().astype(np.int64)
        pnode = s.n + np.arange(P)
        d_lane = t["sea_km_to_lane"].to_numpy()
        du = t["sea_km_u"].to_numpy() - d_lane
        dv = t["sea_km_v"].to_numpy() - d_lane
        ok_e = keep[e]  # a port whose lane edge is closed attaches nowhere via that edge
        mult = np.ones(P) if sea_edge_multiplier is None else np.asarray(sea_edge_multiplier, float)[e]
        U = [s.u[keep], pnode[ok_e], pnode[ok_e]]
        V = [s.v[keep], s.u[e[ok_e]], s.v[e[ok_e]]]
        W = [km[keep], (d_lane + du * mult)[ok_e], (d_lane + dv * mult)[ok_e]]
        # ports projecting onto the same lane edge can reach each other along it directly
        order = np.argsort(e, kind="stable")
        es = e[order]
        bnd = np.flatnonzero(np.r_[True, es[1:] != es[:-1], True])
        for a, b in zip(bnd[:-1], bnd[1:]):
            if b - a < 2 or not keep[es[a]]:
                continue
            grp = order[a:b]
            ii, jj = np.triu_indices(grp.size, 1)
            pi, pj = grp[ii], grp[jj]
            U.append(pnode[pi]); V.append(pnode[pj])
            W.append(d_lane[pi] + d_lane[pj] + np.abs(du[pi] - du[pj]) * mult[pi])
        n = s.n + P
        g = coo_matrix((np.concatenate(W), (np.concatenate(U), np.concatenate(V))), shape=(n, n)).tocsr()
        S = dijkstra(g, directed=False, indices=pnode)[:, pnode]
        self._sea_cache[key] = S
        return S

    # ------------------------------------------------------------- routing
    def _select(self, which, pool) -> np.ndarray:
        if which is None:
            return pool
        t = self.terminals
        which = list(which)
        sel = []
        sub = t.iloc[pool]
        for w in which:
            m = (sub["source_id"] == str(w)) | (sub["name"] == str(w))
            if not m.any():
                raise KeyError(f"no city matching {w!r}")
            sel.extend(pool[np.flatnonzero(m.to_numpy())])
        return np.array(sel, dtype=np.int64)

    def route_costs(self, land_cost: float = 1.0, sea_cost: float = 1.0, port_cost: float = 0.0,
                    origins=None, destinations=None, closed_passages=DEFAULT_CLOSED,
                    closed_regions=(), sea_edge_multiplier=None, max_port_lane_km=None,
                    use_numba: bool = True) -> RouteResult:
        """Lowest generalized-cost routes between cities.

        Parameters
        ----------
        land_cost, sea_cost : cost per km on land / at sea (only their ratio and the
            port cost relative to them matter for route choice).
        port_cost : cost per port call (each sea leg = 2 port calls).
        origins, destinations : optional lists of city names or ids ("ISO3:City").
        closed_passages : sea passages to close, e.g. ["suez", "babalmandab"].
        closed_regions : whole sea regions to close, e.g. ["red_sea"] (see SEA_REGIONS)
            or (lat_min, lon_min, lat_max, lon_max) boxes.
        max_port_lane_km : if set, ports farther than this from the nearest sea lane
            (river/lake ports such as Manaus, ~1,080 km) get no sea connection and act
            as land-only terminals. Default None keeps every port connected.
        """
        if min(land_cost, sea_cost) <= 0 or port_cost < 0:
            raise ValueError("land_cost and sea_cost must be > 0, port_cost >= 0")
        a, b, tau = float(land_cost), float(sea_cost), float(port_cost)
        oi = self._select(origins, self._city)
        di = self._select(destinations, self._city)
        P = self._port
        L = self.L
        S = self.sea_matrix(closed_passages, sea_edge_multiplier, closed_regions)
        if max_port_lane_km is not None:
            far = self.terminals["sea_km_to_lane"].to_numpy()[P] > float(max_port_lane_km)
            S = S.copy(); S[far, :] = np.inf; S[:, far] = np.inf; S[far, far] = 0.0

        # (2) port-to-port single-leg options, then closure
        Lpp = L[np.ix_(P, P)]
        land_c = a * Lpp
        sea_c = 2 * tau + b * S
        use_sea = sea_c < land_c
        D = np.where(use_sea, sea_c, land_c)
        Lk = np.where(use_sea, 0.0, Lpp)
        Sk = np.where(use_sea, S, 0.0)
        Nl = use_sea.astype(np.int32)
        np.fill_diagonal(D, 0.0); np.fill_diagonal(Lk, 0.0); np.fill_diagonal(Sk, 0.0); np.fill_diagonal(Nl, 0)
        D[~np.isfinite(D)] = np.inf
        D, Lk, Sk, Nl = floyd_warshall(D, Lk, Sk, Nl, use_numba)

        # (3) origin -> exit port (land) then port network
        Lop = L[np.ix_(oi, P)]
        A = a * Lop
        zeros_op = np.zeros_like(Lop); n_op = np.zeros(Lop.shape, np.int32)
        C1, LC1, SC1, NC1 = minplus(A, np.where(np.isfinite(Lop), Lop, 0.0), zeros_op, n_op,
                                    D, Lk, Sk, Nl, use_numba)
        # ... then entry port -> destination (land)
        Lpd = L[np.ix_(P, di)]
        B = a * Lpd
        C2, LC2, SC2, NC2 = minplus(C1, LC1, SC1, NC1, B, np.where(np.isfinite(Lpd), Lpd, 0.0),
                                    np.zeros_like(Lpd), np.zeros(Lpd.shape, np.int32), use_numba)
        # pure land alternative
        Lod = L[np.ix_(oi, di)]
        direct = a * Lod
        take = direct <= C2
        cost = np.where(take, direct, C2)
        land_km = np.where(take, Lod, LC2)
        sea_km = np.where(take, 0.0, SC2)
        legs = np.where(take, 0, NC2)
        same = oi[:, None] == di[None, :]
        cost[same] = 0; land_km[same] = 0; sea_km[same] = 0; legs[same] = 0
        unreachable = ~np.isfinite(cost)
        land_km[unreachable] = np.nan; sea_km[unreachable] = np.nan
        t = self.terminals
        return RouteResult(t.iloc[oi].reset_index(drop=True), t.iloc[di].reset_index(drop=True),
                           cost, land_km, sea_km, legs,
                           {"land_cost": a, "sea_cost": b, "port_cost": tau,
                            "closed_passages": list(closed_passages or []),
                            "closed_regions": list(closed_regions or []),
                            "max_port_lane_km": max_port_lane_km})

    def land_km(self, origins=None, destinations=None) -> pd.DataFrame:
        """Pure road distances (km) between cities (NaN when no road connection)."""
        oi = self._select(origins, self._city); di = self._select(destinations, self._city)
        M = self.L[np.ix_(oi, di)].copy(); M[~np.isfinite(M)] = np.nan
        t = self.terminals
        return pd.DataFrame(M, index=t["source_id"].to_numpy()[oi], columns=t["source_id"].to_numpy()[di])
