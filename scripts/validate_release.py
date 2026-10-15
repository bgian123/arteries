"""Run all independent correctness checks on a finished build.
usage: python validate_release.py WORK_DIR VERSION"""
import json, sys
from pathlib import Path
import numpy as np
import arteries
from arteries.build import validate as V

work, ver = Path(sys.argv[1]), sys.argv[2]
net = arteries.load(work / "release" / f"arteries_core_v{ver}.npz")
import pandas as pd
t = pd.read_csv(work / "terminals.csv")
net.terminals["node"] = t["node"].to_numpy()
rep = {
    "simplification_exactness": V.check_simplification(work, net.L),
    "metric": V.check_metric(net.L, t["lat"].to_numpy(), t["lon"].to_numpy()),
    "multimodal_exactness": V.check_multimodal(work, net),
}
print(json.dumps(rep, indent=2))

# hard pass/fail so the Slurm job (and the e-mail) reports FAILED on a bad build
problems = []
se = rep["simplification_exactness"]
if not se["same_reachability"] or se["max_rel_diff"] > 1e-5:
    problems.append("simplified land matrix differs from unsimplified-graph Dijkstra")
m = rep["metric"]
if not (m["diag_zero"] and m["symmetric"]) or m["max_triangle_violation_km(float32 rounding)"] > 1e-2:
    problems.append("land matrix is not a metric")
if m["min_road_over_geodesic"] is not None and m["min_road_over_geodesic"] < 0.999:
    problems.append(f"a road distance is shorter than the geodesic (ratio {m['min_road_over_geodesic']:.4f})")
for r in rep["multimodal_exactness"]:
    if not r["same_reachability"] or r["max_rel_diff"] > 1e-5:
        problems.append(f"route_costs differs from full multimodal Dijkstra at weights {r['weights']}")
print("VALIDATION " + ("PASSED" if not problems else "FAILED:\n  - " + "\n  - ".join(problems)))
sys.exit(1 if problems else 0)
