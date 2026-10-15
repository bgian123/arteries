"""Figure: New Jersey road network before and after exact simplification.

usage: python plot_new_jersey.py <work_dir_with_countries/USA.npz> <states-10m.json> <out_dir> [cities.csv]
The NJ outline is the US Census 1:10m state boundary (us-atlas 3.0.1 TopoJSON).
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MPath
from arteries.build import graphops as G

work, topo, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
cities_csv = sys.argv[4] if len(sys.argv) > 4 else None
out.mkdir(parents=True, exist_ok=True)

t = json.load(open(topo)); sc, tl = t["transform"]["scale"], t["transform"]["translate"]
arcs = [np.cumsum(np.array(a), axis=0) * sc + tl for a in t["arcs"]]
def ring(ix):
    pts = []
    for i in ix:
        a = arcs[i] if i >= 0 else arcs[~i][::-1]
        pts.append(a if not pts else a[1:])
    return np.vstack(pts)
g = [x for x in t["objects"]["states"]["geometries"] if x["properties"]["name"] == "New Jersey"][0]
outline = ring(g["arcs"][0])
poly = MPath(outline)

z = np.load(work / "countries" / "USA.npz")
keys, u, v, w = z["keys"], z["u"], z["v"], z["w"]
lat, lon = G.unpack_keys(keys)
box = (lon > -75.7) & (lon < -73.8) & (lat > 38.85) & (lat < 41.4)
cand = np.flatnonzero(box[u] & box[v])
mid = np.column_stack([(lon[u[cand]] + lon[v[cand]]) / 2, (lat[u[cand]] + lat[v[cand]]) / 2])
inside = cand[poly.contains_points(mid, radius=0.004)]
u, v, w = u[inside], v[inside], w[inside]
used = np.unique(np.r_[u, v])
re = np.full(keys.size, -1); re[used] = np.arange(used.size)
k = keys[used]; uu, vv = re[u], re[v]; la, lo = lat[used], lon[used]

# protect nodes where the network leaves NJ (edges outside the clip) + NJ city anchors
deg_full = G.degrees(keys.size, z["u"], z["v"])
deg_nj = G.degrees(used.size, uu, vv)
protected = deg_full[used] != deg_nj
if cities_csv:
    import pandas as pd
    c = pd.read_csv(cities_csv); c = c[(c.iso3 == "USA") & (c.year <= 2026)]
    c = c.sort_values("year").groupby(["city", "lat", "lon"]).tail(1)
    c = c[(c.population >= 1e5) & poly.contains_points(c[["lon", "lat"]].to_numpy())]
    from scipy.spatial import cKDTree
    _, j = cKDTree(np.column_stack([lo, la])).query(c[["lon", "lat"]].to_numpy())
    protected[j] = True
k2, u2, v2, w2, p2, st = G.simplify(k, uu, vv, w, protected, drop_unprotected_components=False)
la2, lo2 = G.unpack_keys(k2)

INK, MUTED, ACCENT, BG = "#1f2328", "#8b949e", "#c2410c", "#ffffff"
coslat = np.cos(np.radians(40.1))
def frame(ax):
    ax.set_aspect(1 / coslat); ax.set_axis_off()
    ax.plot(outline[:, 0], outline[:, 1], color="#d0d7de", lw=0.8, zorder=0)
    ax.set_xlim(-75.65, -73.85); ax.set_ylim(38.88, 41.40)

def raw_panel(ax, lw=0.45):
    seg = np.stack([np.column_stack([lo[uu], la[uu]]), np.column_stack([lo[vv], la[vv]])], 1)
    ax.add_collection(LineCollection(seg, lw=lw, color=INK, capstyle="round"))
    frame(ax)

# 1) single hero figure
fig, ax = plt.subplots(figsize=(5.2, 7.4), dpi=300, facecolor=BG)
raw_panel(ax, 0.4)
ax.set_title(f"New Jersey road network\n", fontsize=12.5, weight="bold", color=INK, loc="left", pad=2)
ax.text(0.0, 1.0, f"{used.size:,} nodes · {uu.size:,} edges · {w.sum():,.0f} km of road (OpenStreetMap)",
        transform=ax.transAxes, fontsize=7.6, color=MUTED, va="bottom")
fig.savefig(out / "nj_roads_raw.png", dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig)

# 2) before / after
fig, axs = plt.subplots(1, 2, figsize=(9.6, 7.2), dpi=250, facecolor=BG)
raw_panel(axs[0], 0.35)
axs[0].scatter(lo, la, s=0.15, color=INK, lw=0, alpha=0.6)
seg2 = np.stack([np.column_stack([lo2[u2], la2[u2]]), np.column_stack([lo2[v2], la2[v2]])], 1)
axs[1].add_collection(LineCollection(seg2, lw=0.45, color=MUTED))
axs[1].scatter(lo2, la2, s=2.2, color=ACCENT, lw=0, zorder=3)
frame(axs[1])
axs[0].set_title(f"Before: every OSM vertex\n{used.size:,} nodes · {uu.size:,} edges", fontsize=10, color=INK, loc="left")
axs[1].set_title(f"After: junctions only (exact distances)\n{k2.size:,} nodes · {u2.size:,} edges", fontsize=10, color=INK, loc="left")
fig.text(0.01, 0.015, "Edges carry summed WGS84 length; chains of degree-2 vertices and dead-end spurs removed. "
         f"Node reduction {1 - k2.size / used.size:.1%}.", fontsize=7.5, color=MUTED)
fig.savefig(out / "nj_before_after_simplification.png", dpi=250, bbox_inches="tight", facecolor=BG)
print(json.dumps({"nodes_raw": int(used.size), "edges_raw": int(uu.size), "km": float(w.sum()),
                  "nodes_simplified": int(k2.size), "edges_simplified": int(u2.size)}))
