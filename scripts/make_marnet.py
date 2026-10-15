"""Convert searoute-py's MARNET adjacency (searoute/data/marnet_dict.py, v1.6.0)
into the compact array file shipped with Arteries (src/arteries/data/marnet.npz).

Cleaning (documented in docs/METHODS.md §5):
  * longitudes are normalized to [-180, 180); nodes at +180 and -180 with the
    same latitude are merged, so the zero-length 'pacific_ocean' joins vanish;
  * searoute's antimeridian seam edges ('segment', 'segment2') run straight along
    the 180th meridian; those overlapping 64.5-71.5N cross Chukotka / Wrangel
    Island (land) and are removed, the remaining open-ocean seam edges are kept;
  * parallel duplicates are collapsed; edge lengths are recomputed as exact
    WGS84 geodesics instead of searoute's 0.1-km-rounded haversine.
usage: python make_marnet.py path/to/marnet_dict.py out.npz
"""
import importlib.util, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from arteries.geodesy import geodesic_km

spec = importlib.util.spec_from_file_location("marnet_dict", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def norm(lon):
    lon = float(lon)
    while lon >= 180.0: lon -= 360.0
    while lon < -180.0: lon += 360.0
    return lon

ids, lat, lon = {}, [], []
def nid(p):
    k = (round(p[1] * 1e6), round(norm(p[0]) * 1e6))
    if k not in ids:
        ids[k] = len(lat); lat.append(k[0] / 1e6); lon.append(k[1] / 1e6)
    return ids[k]

edges = {}
dropped = 0
for a, nbrs in m.edge_list.items():
    for b, attr in nbrs.items():
        pas = attr.get("passage") or ""
        if pas in ("segment", "segment2"):
            lo, hi = sorted((a[1], b[1]))
            if lo < 71.5 and hi > 64.5:
                dropped += 1; continue
            pas = ""
        if pas == "pacific_ocean":
            pas = ""
        u, v = nid(a), nid(b)
        if u == v: continue
        key = (min(u, v), max(u, v))
        if key not in edges or (pas and not edges[key]):
            edges[key] = pas
lat = np.array(lat); lon = np.array(lon)
uv = np.array(list(edges.keys()), dtype=np.int64)
pas = np.array(list(edges.values()), dtype="U16")
km = geodesic_km(lat[uv[:, 0]], lon[uv[:, 0]], lat[uv[:, 1]], lon[uv[:, 1]])
np.savez_compressed(sys.argv[2], lat=lat, lon=lon, u=uv[:, 0], v=uv[:, 1], km=km, passage=pas)
print(f"nodes={lat.size} edges={uv.shape[0]} dropped_seam_directed={dropped} total_km={km.sum():,.0f}")
