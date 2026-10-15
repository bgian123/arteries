#!/usr/bin/env python3
"""Download road segments from the OpenStreetMap Overpass API in the same format
as the existing country pickles (gzipped pickle of a set of
((lat1, lon1), (lat2, lon2)) tuples, coordinates rounded to 6 decimals, both
directions stored).

Two uses
--------
1) Fill countries that are missing or empty in initial_graphs/:

    python fetch_overpass.py missing --roads-dir initial_graphs --out-dir initial_graphs

   (Eritrea, Niger, Nauru, Tuvalu failed in the original download; 32 files hold
   an empty set, including PRI, HKG, MAC, REU, GLP, MTQ, GUF, PYF, NCL, GUM; TWN
   looks truncated. Pass --iso to choose explicitly, e.g. --iso NER ERI TWN.)

2) Download the *_link ramp roads for every country so interchanges are
   represented by real OSM topology rather than by inserted crossing junctions:

    python fetch_overpass.py links --roads-dir initial_graphs --out-dir links

   then build with  --set insert_crossings=false  and point the build at a
   directory containing both sets of files (ingest reads every
   <iso>_roads.pkl.gz and, if present, <iso>_links.pkl.gz).

Road classes: set --classes to the classes used for the original download
(default: motorway|trunk|primary|secondary). Run from a machine whose network
can reach overpass-api.de (your laptop or a cluster login node); the Claude
workspace could not.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
KNOWN_MISSING = ["ERI", "NER", "NRU", "TUV"]
SUSPECT = ["TWN"]


def overpass(query: str, timeout: int = 900) -> dict:
    data = urllib.parse.urlencode({"data": query}).encode()
    last = None
    for attempt in range(6):
        url = ENDPOINTS[attempt % len(ENDPOINTS)]
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "arteries-build/0.1"})
            with urllib.request.urlopen(req, timeout=timeout + 60) as r:
                return json.load(r)
        except Exception as e:  # rate limits, timeouts, gateway errors
            last = e
            wait = 30 * (attempt + 1)
            print(f"    {url}: {e}; retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Overpass failed: {last}")


def area_filter(iso3: str) -> str:
    return f'area["ISO3166-1:alpha3"="{iso3}"]["boundary"="administrative"]->.a;'


def bbox_of(iso3: str):
    q = f'[out:json][timeout:300];{area_filter(iso3)}rel(pivot.a);out bb;'
    d = overpass(q, 300)
    b = [e["bounds"] for e in d.get("elements", []) if "bounds" in e]
    if not b:
        return None
    return (min(x["minlat"] for x in b), min(x["minlon"] for x in b),
            max(x["maxlat"] for x in b), max(x["maxlon"] for x in b))


def fetch(iso3: str, regex: str, tiles: int = 1) -> set:
    segs = set()
    boxes = [None]
    if tiles > 1:
        bb = bbox_of(iso3)
        if bb:
            s, w, n, e = bb
            dy, dx = (n - s) / tiles, (e - w) / tiles
            boxes = [(s + i * dy, w + j * dx, s + (i + 1) * dy, w + (j + 1) * dx)
                     for i in range(tiles) for j in range(tiles)]
    for k, box in enumerate(boxes):
        bbox = "" if box is None else "(%.6f,%.6f,%.6f,%.6f)" % box
        q = f'[out:json][timeout:900][maxsize:4000000000];{area_filter(iso3)}way["highway"~"{regex}"](area.a){bbox};out geom;'
        d = overpass(q)
        for el in d.get("elements", []):
            g = el.get("geometry") or []
            for a, b in zip(g[:-1], g[1:]):
                p = (round(a["lat"], 6), round(a["lon"], 6))
                r = (round(b["lat"], 6), round(b["lon"], 6))
                if p != r:
                    segs.add((p, r)); segs.add((r, p))
        if box is not None:
            print(f"    tile {k+1}/{len(boxes)}: {len(segs):,} segments so far", flush=True)
    return segs


def save(segs: set, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb") as fh:
        pickle.dump(segs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["missing", "links"])
    ap.add_argument("--roads-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iso", nargs="*")
    ap.add_argument("--classes", default="motorway|trunk|primary|secondary")
    ap.add_argument("--tiles", type=int, default=1, help="split each country into tiles x tiles boxes")
    a = ap.parse_args()
    roads = Path(a.roads_dir); out = Path(a.out_dir)
    if a.iso:
        isos = [x.upper() for x in a.iso]
    elif a.mode == "missing":
        empty = [p.name.split("_")[0].upper() for p in roads.glob("*_roads.pkl.gz") if p.stat().st_size < 64]
        isos = sorted(set(KNOWN_MISSING + SUSPECT + empty))
    else:
        isos = sorted(p.name.split("_")[0].upper() for p in roads.glob("*_roads.pkl.gz") if p.stat().st_size >= 64)
    cls = a.classes
    if a.mode == "missing":
        regex = f"^({cls})(_link)?$"; suffix = "roads"
    else:
        regex = f"^({cls})_link$"; suffix = "links"
    print(f"{a.mode}: {len(isos)} countries, highway~{regex}")
    log = []
    for iso in isos:
        path = out / f"{iso.lower()}_{suffix}.pkl.gz"
        if path.exists() and path.stat().st_size > 64 and a.mode == "links":
            print(f"{iso}: exists, skip"); continue
        t = time.time()
        try:
            segs = fetch(iso, regex, a.tiles)
        except Exception as e:
            if a.tiles == 1:
                print(f"{iso}: whole-country query failed ({e}); retrying in 4x4 tiles")
                segs = fetch(iso, regex, 4)
            else:
                raise
        save(segs, path)
        print(f"{iso}: {len(segs)//2:,} segments in {time.time()-t:.0f}s -> {path}", flush=True)
        log.append({"iso3": iso, "segments": len(segs) // 2, "seconds": round(time.time() - t)})
        time.sleep(5)  # be polite to the public API
    (out / f"fetch_{a.mode}_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    sys.exit(main())
