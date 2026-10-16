# Arteries

**Calculate lowest-cost distances between the world's cities (population > 100,000) over a real infrastructure network, with land kilometers, sea kilometers, and port calls priced to your desired specifications.**

![New Jersey road network before and after simplification](https://raw.githubusercontent.com/bgian123/arteries/main/docs/figures/nj_before_after_simplification.png)

Modeling distance is important. Economists tend to rely on great-circle (as-the-crow-flies) distances because of their numerical simplicity. The caveat is that a majority of goods do not travel in clean arcs to their destinations. They move by road to a port, by sea along shipping lanes and through chokepoints, and by road again. Arteries computes that route for every pair of cities with at least 100,000 people.

You choose what a road kilometer, a sea kilometer, and a port call cost. The package efficiently returns the cheapest route's generalized cost, together with the land and sea kilometers it uses and its number of sea legs.

Results are exact: they equal Dijkstra's algorithm on the full global multimodal graph for any choice of costs. They are also very fast. Routing all 2,075 × 2,075 city pairs takes about 20 seconds on a laptop, a major improvement over previous options.

```python
import arteries

net = arteries.load()                       # downloads the network data once (~21 MB)

res = net.route_costs(
    land_cost=5.0,                          # cost per road km (relative to sea)
    sea_cost=1.0,                           # cost per sea km
    port_cost=1000.0,                       # cost per port call (loading or unloading)
    closed_regions=["red_sea"],             # e.g. the 2023-24 Red Sea disruption
)
df = res.to_frame()      # origin, destination, cost, land_km, sea_km, total_km, sea_legs
```

* **Exact.** The answer equals Dijkstra on the full multimodal graph for any weights. Proof and tests: [docs/METHODS.md](https://github.com/bgian123/arteries/blob/main/docs/METHODS.md).
* **Fast.** Routing all 2,075 × 2,075 city pairs takes about 20 seconds, because the expensive land shortest paths are precomputed once and every new weighting only reweights a small terminal graph.
* **Decomposed.** Every result reports land km, sea km, and the number of sea legs of the cheapest route.
* **Scenarios.** You can close whole sea regions (`closed_regions=["red_sea"]`, `"persian_gulf"`, `"black_sea"`, `"baltic_sea"`, or any lat/lon box), turn river and lake ports into land-only terminals (`max_port_lane_km`), close chokepoints (`suez`, `panama`, `babalmandab`, `malacca`, `ormuz`, `gibraltar`, `bosporus`, `dardanelles`, `sunda`, `bering`, `chili`, `south_africa`, `northwest`), or rescale individual sea lanes with `sea_edge_multiplier`.

Documentation: https://blaizegiangiulio.com/arteries/

## Install

```bash
pip install arteries            # numpy, scipy, pandas
pip install "arteries[fast]"    # + numba (strongly recommended)
```

Data is cached in `~/.cache/arteries`; set the `ARTERIES_DATA` environment variable to use another folder.

## What is in the network

| Layer | Source | Size after processing |
|---|---|---|
| Roads | OpenStreetMap main road classes, 236 countries | Simplified to junctions only (−99 % nodes) |
| Sea lanes | MARNET via searoute-py | 9,690 nodes, 15,947 lanes |
| Ports | IMF PortWatch | 1,804 ports (181 offshore oil terminals excluded) |
| Cities | UN WUP + Brinkhoff | 2,075 cities with ≥ 100,000 people (latest year ≤ 2026) |

## Rebuilding the data

The build pipeline (`python -m arteries.build ...`) and the Slurm scripts (`slurm/`)
reproduce the release from the raw extracts. See [docs/METHODS.md](https://github.com/bgian123/arteries/blob/main/docs/METHODS.md)
for every step, the exactness arguments, validation results and resource use.

## Licenses and attribution

* **Code:** MIT.
* **Road data:** © OpenStreetMap contributors, ODbL 1.0. The distributed distance data is a derivative database and is shared under ODbL.
* **Maritime network:** searoute-py (Apache-2.0), from Eurostat SeaRoute / ORNL shipping lanes.
* **Ports:** IMF PortWatch.
* **Cities:** UN World Urbanization Prospects and citypopulation.de (Brinkhoff).

## Citation

Giangiulio, B. (2026). *Arteries: exact multimodal transport distances between world cities.* Python package version 0.1.0.
