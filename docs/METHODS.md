# Arteries: build methodology, exactness arguments and computational record

*Version 0.1.0 · October 2026 · Blaize Giangiulio*

This document records every step that turns the raw inputs into the data shipped with the
`arteries` Python package. For each step it gives the algorithm, why the step leaves
shortest-path distances unchanged (or exactly how it changes them), the parameters, and
measured cost. Section 9 lists independent checks that are run on every build.

---

## 0. Notation

* $G=(N,E,w)$ is an undirected graph with edge lengths $w_e>0$ in km.
* $d_G(a,b)$ is the shortest-path distance in $G$, with $d_G(a,b)=\infty$ when $a$ and $b$ are not connected.
* **Terminals** $\mathcal T=\mathcal C\cup\mathcal P$: the cities $\mathcal C$ and ports $\mathcal P$. These are the only nodes whose pairwise distances are ever needed.
* A node is **protected** if it, or a terminal attached to it, must survive simplification.

---

## 1. Inputs

| Input | Content | Notes |
|---|---|---|
| `initial_graphs/<iso3>_roads.pkl.gz` | 236 files. Each is a gzipped pickle of a `set` of segments `((lat1, lon1), (lat2, lon2))`, stored in both directions, at 6 decimals | Overpass extracts of the main road classes; no way IDs or road classes. 32 files hold an empty set; ERI, NER, NRU and TUV failed to download; TWN looks truncated |
| `cities/cities_merged.csv` | 316k city-year rows for 6,633 cities, 1950–2035 (UN + Brinkhoff) | — |
| `ports/PortWatch_ports_database.csv` | 1,985 ports | 181 offshore oil terminals (every port whose `portname` contains "Offshore") are excluded, leaving **1,804 ports**. The three Guinean ports at the end of the file (Dapilon, Katougouma, Taressa) are real ports and are kept |
| MARNET maritime graph | 9,708 nodes and 15,970 lanes, from `searoute` 1.6.0 (Apache-2.0) | Built from Eurostat SeaRoute / ORNL global shipping lanes. Cleaned as described in §5 |

**City set.** For every `(iso3, city, lat, lon)`, the pipeline keeps the latest observation with
`year <= 2026`, then keeps cities whose population at that date is at least 100,000. This
gives **2,075 cities**. The UN series carries projections to 2035, and capping the year at
2026 keeps the selection from depending on forecasts. City IDs are `ISO3:City`; a name that
repeats within a country gets `#2`, `#3`, … (for example the two US Concords).

---

## 2. Stage 1: ingest (one array task per country)

1. **Reading the pickles.** A Numba-compiled interpreter of the pickle opcode stream
   (`build/pickle_reader.py`) writes coordinates directly into NumPy arrays. Peak memory
   is about 32 bytes per segment, against roughly 300 bytes with `pickle.load`.
   - It handles exactly the opcodes CPython emits for a set of 2-tuples of 2-tuples of floats (protocols 4–5).
   - Any other opcode makes it fall back to `pickle.load`.
   - Tests confirm it returns the same segment set as `pickle.load` on synthetic files and on SMR, EST, FIN and NLD.
   - The USA file (378 MB, 20.7 M stored segments) parses in 7 s with 1.6 GB peak memory.
2. **Node identity.** Coordinates are quantized to integer micro-degrees and packed into an
   `int64` key: (lat × 10⁶) ≪ 32 | (lon × 10⁶). Identical keys are the same node. This
   also merges the same OSM node appearing in two country files, which is how border
   crossings are joined.
3. **Crossing junctions** (`build/crossings.py`).
   - **Why it is needed.** The extracts contain carriageways but not `*_link` ramps. Two roads that meet at a grade-separated interchange therefore share no node.
   - **Evidence.** On Denmark, the E45 crosses every radial road around Aarhus without a shared node. Before this step, the road distance from Odense to Aarhus was 556 km, a detour through Aalborg. After it, the distance is 152 km.
   - **Method.** The pipeline inserts a node at every *proper* crossing of two segments (their interiors intersect and they share no endpoint) and splits both segments there.
   - **Exactness of the test.** Intersection is invariant under affine maps, so the test runs in (lon, lat) with exact orientation predicates.
   - **Each pair is tested once.** Candidate pairs come from a uniform grid (0.005°). A pair is tested only in the grid cell that holds the lower-left corner of the overlap of the two bounding boxes. Both segments are always registered in that cell.
   - **Caveat.** Where one road crosses another with no interchange, this adds a junction that doesn't exist, which can only shorten routes. The bias disappears if link roads are downloaded (`scripts/fetch_overpass.py links`) and the build is run with `insert_crossings=false`.
   - **Measured counts.** 46,315 crossings for the USA, 273 for Denmark, 1,854 for the Netherlands.
4. **Edge lengths.** Each segment's length is computed on the WGS84 ellipsoid with the local metric

   $$ds^2=\big(M(\varphi)\,d\varphi\big)^2+\big(N(\varphi)\cos\varphi\,d\lambda\big)^2,\qquad
   M=\frac{a(1-e^2)}{(1-e^2\sin^2\varphi)^{3/2}},\; N=\frac{a}{(1-e^2\sin^2\varphi)^{1/2}},$$

   evaluated at the mid-latitude. Here $M$ and $N$ are the meridional and prime-vertical radii of curvature.
   - **Accuracy.** The relative error against Karney's exact geodesic is $O((d/R)^2)$. The test-suite measures it below $10^{-5}$ for segments up to about 1.5 km.
   - **Typical segment.** The median segment is 37 m and the 99th percentile is 0.6 km (USA).
   - **Longer links.** City and port connectors, gap links and sea lanes use the exact Karney geodesic (`pyproj`).
5. **Canonical edges.** Edges are oriented $u<v$, self-loops are dropped, and duplicate edges are collapsed to the minimum length.

**Output:** `countries/<ISO3>.npz` and `stats/ingest_<ISO3>.json`. The JSON records node and edge counts, km, number of components, dead ends and segment-length quantiles.

---

## 3. Stage 2: merge and gap repair (single large-memory job)

**Merge.** All country graphs are concatenated, and node keys are made globally unique with
`np.unique`. A key that appears in two or more country files is a shared border node.
`country_land_adjacency.csv` counts these shared nodes per country pair; it is a by-product
that documents which borders actually connect.

**Gap repair.** Data gaps split the road network into many small components. The pipeline
joins **dead ends** to the nearest node of a *different* component within `bridge_km = 2 km`,
using Borůvka rounds:

* *(a)* dead ends of non-giant components are linked to the exact nearest node of the giant component (k=1 KD-tree query on a tree of giant-component nodes);
* *(b)* dead ends of non-giant components are linked to the nearest node of another non-giant component (k=32 nearest, keeping the first node with a different component label);
* *(c)* dead ends of the giant component are linked to the exact nearest node of a non-giant component.

In each round, every non-giant component keeps only its shortest candidate link (the Borůvka
step). This guarantees that a round never adds a link inside a component and adds at most
#components − 1 links. Three rounds are run. Link lengths are exact geodesics.

| Region | Components before repair | Components after repair | Links added | Mean link length |
|---|---|---|---|---|
| USA | 419 | 48 | 393 | 0.15 km (57 km in total) |
| 9-country test region | 113 | 33 | 92 | — |

Repair deliberately does **not** connect components more than 2 km apart. Islands are reached
by sea (§6), not by invented bridges.

---

## 4. Stage 3: terminals and snapping

* **Cities** attach to the nearest road node within `city_snap_km = 25 km`.
* **Ports** attach within `port_snap_km = 80.47 km` (50 miles, the threshold used in the earlier pipeline).
* **Snapping prefers "major" components** (at least 10 km of road), so a city is not attached to a 200 m fragment that happens to be closest. If no major component is in range, snapping falls back to any component.
* **Connector length** is the exact geodesic. The connector becomes a land edge, so it is costed at the land rate.

Measured on the USA: the median city connector is 0.18 km and the maximum is 1.8 km. Two
ports did not snap: "Dutch Harbor" (Aleutians) and a port PortWatch labels "Sydney, USA",
which looks like a source-data issue to check.

---

## 5. Stage 4: exact simplification (single large-memory job)

Protected nodes = the attachment nodes of all terminals. The pipeline repeats the following
until the edge count stops changing.

1. **Drop unprotected components.** A component with no protected node cannot lie on a path between terminals, so removing it is exact.
2. **Contract degree-2 chains.** Define an *anchor* as a node that is protected or has degree ≠ 2.
   - The non-anchor nodes induce a union of simple paths and cycles.
   - Each path is attached to exactly two anchors $a,b$ by two boundary edges. The test-suite asserts this and the pipeline raises an error if it ever fails.
   - The path is replaced by one edge $(a,b)$ whose weight is the sum of the path's edge weights.
   - **Exactness.** Every $a$–$b$ walk through the chain uses the whole chain, because its interior nodes have no other neighbours. So for any anchors $x,y$, $d_{G'}(x,y)=d_G(x,y)$.
   - **Cycles and self-loops.** Pure cycles have no anchors and are dropped. A chain whose two ends are the same anchor becomes a self-loop and is dropped. Neither can lie on a shortest path between two different nodes.
   - **Parallel edges.** These are collapsed to the minimum, which is exact.
   - **Implementation.** The step is fully vectorized. Chains are labelled with `scipy.sparse.csgraph.connected_components` on the non-anchor subgraph, and chain lengths are summed with `bincount`.
3. **Prune leaves.** Unprotected degree-1 nodes are removed until none remain.
   - **Exactness.** A degree-1 node $x\neq s,t$ cannot be an interior vertex of a simple $s$–$t$ path.
   - **Implementation.** A single O(n+m) queue sweep in Numba. It is equivalent to iterating the one-shot rule to a fixed point; that iteration needed 2,253 rounds on the test region.
4. **Compact** the node index, so later iterations cost O(current size).

Several iterations are needed because collapsing parallel chains (dual carriageways, ladders)
lowers degrees and creates new chains. The USA took 16 iterations, the test region 32.

| Region | Nodes | Edges | Time / memory |
|---|---|---|---|
| USA | 10,342,034 → 114,765 (−98.9 %) | 10,440,133 → 199,432 | 10 s, 1.4 GB |
| Test region (9 countries) | 1,090,062 → 27,192 | — | — |
| New Jersey (figure) | 220,868 → 3,234 | 223,458 → 5,574 | — |

**Verification.** Dijkstra on the *unsimplified* graph matches the packaged matrix to within float32 rounding. The largest relative difference is 5.9 × 10⁻⁸ (§9).

---

## 6. Stage 5: assembly, islands, ports at sea

1. **Terminal nodes.** Each terminal becomes its own node, attached by its connector edge. Because a terminal node is a leaf, no other route can pass through it.
2. **Making every city able to reach a port.**
   - Consider each connected land component that contains a city but no port.
   - Let $d_L$ be the geodesic distance from the component to the nearest node of a port-bearing component, and $d_S$ the distance to the nearest sea-lane node.
   - If $d_S<d_L$, the component is an island: the node nearest the sea becomes a **synthetic port**.
   - Otherwise, a **synthetic land link** of length $d_L$ joins it to the port-bearing network. Typical cases are a country with no road file (Niger) or a city beyond the snapping radius.
   - The pass is repeated until no such component is left. Every synthetic object is listed in `synthetic_land_links.csv` and in `terminals.csv` (`kind = port_synthetic`), with its distances, so it can be audited.
3. **Ports on the sea graph** (`sea.snap_points_to_sea`).
   - Each port is projected orthogonally onto the nearest sea lane, chosen among the lanes incident to its 24 nearest sea nodes.
   - The port then links to both ends of that lane with lengths $d(\text{port},\pi)+d(\pi,u)$ and $d(\text{port},\pi)+d(\pi,v)$.
   - Ports projecting onto the same lane also link directly, with length $d_A+d_B+|s_A-s_B|$, where $s$ is the position along the lane.
   - This is the same as inserting the projection points as nodes on the lane, without changing the shared sea graph.
   - Measured on the USA: the median port-to-lane distance is 7 km and the 99th percentile is 88 km (inland river ports).
4. **Cleaning of the MARNET graph** (`scripts/make_marnet.py`):
   - longitudes are normalized to [−180, 180), and the +180/−180 seam nodes are merged, which removes searoute's zero-length `pacific_ocean` joins;
   - the antimeridian seam edges that run along the 180th meridian across Chukotka and Wrangel Island (64.5–71.5° N) are removed;
   - duplicate edges are collapsed;
   - lengths are recomputed as exact geodesics (searoute rounds haversine lengths to 0.1 km).

   Result: 9,690 nodes, 15,947 lanes, a single connected component. The 13 passage tags
   (suez, panama, babalmandab, malacca, ormuz, gibraltar, bosporus, dardanelles, sunda,
   bering, chili, south_africa, northwest) are kept so users can close them. As in
   searoute, `northwest` is closed by default.

---

## 7. Stage 6: the terminal land-distance matrix (array job)

For each terminal $t$, single-source Dijkstra (`scipy.sparse.csgraph.dijkstra`) is run on the
assembled land graph, and the distances to all terminals are stored.

* **Size.** $|\mathcal T|\approx 3{,}900$, so the matrix $L$ has about 15 M entries, stored as float32 (61 MB).
* **Parallelism.** Sources are split into chunks of 64, one array task per chunk. Within a task, `fork()`-ed workers share the CSR graph copy-on-write.
* **Cost.** On the USA graph (115 k nodes), one source takes about 10 ms. The global simplified graph is expected to be around 1–2 M nodes, which makes this stage minutes of wall time.
* **Terminal bookkeeping.** Packed node keys are negative for southern latitudes, so whether a terminal snapped to a road is recorded in an explicit `snapped` flag, never inferred from the sign of its key. The first global build used a sign test, which silently disconnected every southern-hemisphere terminal; the validation step caught it.
* **Packaging checks.** Every row is present; no row file is older than the graph (stale files are rejected); and $L$ is symmetric to 10⁻⁶ km (measured: 2 × 10⁻¹¹ km).

---

## 8. Query-time algebra (the `arteries` package)

**Cost of a route.** Let $\alpha$ be the cost per land km, $\beta$ the cost per sea km and
$\tau$ the cost per port call, where each sea leg makes two port calls. The cost of a route is

$$c=\alpha\,\text{km}_\text{land}+\beta\,\text{km}_\text{sea}+\tau\,(\#\text{port calls}).$$

**Claim.** The minimum-cost route between cities $i$ and $j$ in the full multimodal graph is

$$c^*(i,j)=\min\Big\{\alpha L_{ij},\;\min_{p,q\in\mathcal P}\big[\alpha L_{ip}+D_{pq}+\alpha L_{qj}\big]\Big\},$$

where $D$ is the shortest-path closure (Floyd–Warshall) over ports of the one-step costs
$A_{pq}=\min\{\alpha L_{pq},\,2\tau+\beta S_{pq}\}$, and $S$ is the port-to-port sea distance with the chosen passages closed.

**Proof.**
1. Any route alternates maximal land segments and maximal sea segments. Mode changes happen only at ports, which are the only nodes with both land and sea edges.
2. Every maximal land segment runs between two terminals: the city at each end, or the port where the mode changes. Its cheapest version costs $\alpha$ times the land-only shortest distance, which is $\alpha L$.
3. Every maximal sea segment runs port-to-port and costs at least $2\tau+\beta S_{pq}$. Equality holds because $S$ is the sea shortest path.
4. The route's cost is therefore the cost of a walk in the complete graph on $\mathcal T$ with these edge costs. The cheapest such walk is given by the closure above.
5. Conversely, every walk in that graph corresponds to a real route.

Hence the formula is exact. The decomposition $L\to D\to$ (city, port) → (city, city)
is a pair of min-plus products. These are computed in parallel with Numba, which skips
unreachable intermediate nodes; NumPy fallbacks give bit-identical results, which the
test-suite checks.

**Reported km.** Alongside the cost, the companion matrices track land km, sea km and the
number of sea legs of the arg-min route. They are updated whenever the incumbent route is
replaced, so the reported km always belong to the cost-minimizing route.

**Verification.** The formula is compared with Dijkstra on the full multimodal graph (roads + connectors + sea lanes, weighted as above) for random city samples under weights
(1,1,0), (4,1,100) and (1,3,25). The largest relative difference is 5 × 10⁻⁸ (float32 storage of $L$).

**Cost of a query.** About 3 × 10¹⁰ elementary min/add operations for all city pairs with all ports, taking seconds on a laptop with Numba. A new set of closed passages also requires one Dijkstra per port on the roughly 11 k-node sea graph (about 1 s); the result is cached per passage set.

---

## 9. Validation (run automatically by `07_package.sbatch` → `validation_report.txt`)

| Check | Test region (9 countries) | USA |
|---|---|---|
| Unsimplified-graph Dijkstra vs. packaged $L$ (12 random sources): largest relative difference | 5.4e-8 | 5.9e-8 |
| Same pairs reachable in both | yes | yes |
| Triangle inequality on 200 k random triples: largest violation | 0 | 0 |
| Road km / geodesic km: minimum | 1.06 | 1.03 |
| Road km / geodesic km: median, pairs over 50 km | 1.24 | 1.15 |
| Full multimodal Dijkstra vs. `route_costs`: largest relative difference | 4e-8 | 5e-8 |

The US median circuity of 1.15 sits in the usual 1.1–1.3 range reported for road networks.

**Spot checks** against typical driving distances (my reference values, approximate):

| Route | Arteries | Typical driving distance |
|---|---|---|
| Helsinki–Tampere | 177.7 km | about 178 km |
| Rotterdam–Antwerp | 103.8 km | about 100 km |
| Copenhagen–Aarhus | 309 km | about 305 km |
| NYC–Philadelphia | 141.5 km | about 150 km |
| Chicago–Houston | 1,715 km | about 1,740 km |
| NYC–LA | 4,406 km | about 4,450 km |

The slight undershoot is consistent with the inserted crossing junctions and with snapping
cities to their nearest main road.

---

## 10. Cluster execution and resources

Slurm scripts are in `slurm/` and settings in `slurm/config.sh`.

**Resource requests (Picotte).** Every job runs on the free-tier partition `def-sm` under account `lebowfreeprj`, following `picotte_slurm_guide.md`.
- Each job asks for a whole def-sm node's worth of resources: 48 cores and all node memory (`--mem=0`), with a 24 h wall time. Every stage is expected to need minutes.
- Arrays are not throttled; the guide reports no per-user job limits.
- The ingest array reads a static manifest (`countries.txt`) instead of a directory listing.
- The logs folder is created before `sbatch`.
- Failure e-mails go to bg654@drexel.edu; the final package job also e-mails on success.

**Measured resource use** (2 vCPU / 7 GB workstation):

| Step | USA alone | World estimate (about 8× USA) |
|---|---|---|
| ingest | 63 s, 2.0 GB | 1–2 min per country (array) |
| merge + gap repair | 48 s, 2.2 GB | 10–20 min, 20–40 GB |
| terminals | 30 s, 1.7 GB | 5 min |
| simplify | 11 s, 1.4 GB | 2–5 min, 15–30 GB |
| assemble | 1 s | 1–5 min |
| landmatrix | 4 s | minutes (array) |

The whole build fits comfortably in the requested resources.

**Reproducibility.**
- `work/config.json` stores every parameter.
- Each stage writes `stats/<stage>.json` (counts, quantiles, host, versions, Slurm job IDs) and a timestamped log with peak RSS in `logs/`.
- All outputs are written to a temporary file and then renamed, so a killed job never leaves a partial file.
- Stages can be re-run individually. Re-running assemble invalidates the land-matrix rows automatically.

---

## 11. Parameters

| Name | Default | Meaning |
|---|---|---|
| `city_min_population` | 100,000 | Population threshold for cities |
| `city_max_year` | 2026 | Use the latest observation up to this year |
| `insert_crossings` | true | Add a junction where two segments cross (link roads absent) |
| `crossing_cell_deg` | 0.005 | Grid cell size for crossing detection (affects speed only) |
| `bridge_km` / `bridge_rounds` | 2 km / 3 | Gap repair |
| `city_snap_km` | 25 | City → road snapping radius |
| `port_snap_km` | 80.47 | Port → road snapping radius (50 mi) |
| `exclude_offshore_ports` | true | Drop the 181 offshore terminals |
| `sssp_chunk` | 64 | Sources per land-matrix array task |

Change a parameter with `python -m arteries.build <stage> --work W --set key=value`, then re-run that stage and every later one.

---

## 12. Known limitations and how to remove them

1. **Missing or empty countries.** ERI, NER, NRU and TUV failed; 32 files are empty (PRI, HKG, MAC, REU, GLP, MTQ, GUF, PYF, NCL, GUM, …); TWN looks truncated. Cities there are connected through synthetic links or synthetic ports, flagged in the outputs. **Fix:** `scripts/fetch_overpass.py missing` (Overpass was unreachable from the build workstation).
2. **Link roads absent.** Crossing junctions stand in for interchanges. **Fix:** `fetch_overpass.py links`, then build with `insert_crossings=false`.
3. **No road class or speed.** Land cost is per km only. Road-class weights would need class tags in the extracts.
4. **Ferries are not land edges.** Short straits (Messina, Øresund before its bridge, …) are crossed as sea legs between ports, which is what the cost model is for.
5. **MARNET resolution.** Lanes are coarse near coasts. Ports up to about 100 km inland (river ports) attach with a long sea connector, costed at the sea rate.
6. **Float32 storage.** $L$ is stored at about 1e-7 relative precision, which is irrelevant at km scale.

---

## 13. Results of the global build (Picotte, 15 October 2026)

**Network size by stage**

| Stage | Result |
|---|---|
| Raw global road graph | 75,784,991 nodes · 76,594,221 edges · 5.36 M km of road; 286 country pairs share border nodes |
| Gap repair | 6,329 → 1,064 components; 5,929 links averaging 0.17 km (1,011 km in total; longest 2.0 km) |
| Simplified graph | 1,042,247 nodes (−98.6 %) · 1,717,761 edges · 4.66 M km kept; 63 iterations |
| Terminals | 3,879 = 2,075 cities + 1,804 ports; 0 synthetic ports |
| Distance matrix | 3,879 × 3,879; 35.8 % of terminal pairs connected by road |

**Cities without a road within 25 km (6).** These connect through a synthetic land link (§6), each listed in `synthetic_land_links.csv`:

| City | Country | Link length | Reason |
|---|---|---|---|
| Asmara | ERI | 65 km | Eritrea's road download failed |
| Niamey | NER | 132 km | Niger's road download failed |
| Zinder | NER | 109 km | Niger's road download failed |
| Honiara | SLB | 3 km | Linked to its port |
| Ibanda | UGA | 48 km | — |
| Masindi | UGA | 49 km | — |

The other synthetic links are Nablus (24 km, the West Bank network is isolated in the extracts) and two Philippine island components (25 km each).

**Ports with no road within 80 km (46).** These are mostly Pacific islands, LNG and oil terminals, Arctic Russian ports and Caspian ports. They keep their sea connection and act as sea-only terminals.

**Ports far from a sea lane.** MARNET has no inland waterways, so river and lake ports attach by a straight sea connector:
- 91 ports are more than 100 km from a lane;
- the largest are Amazon ports (Coari 1,350 km, Manaus 1,079 km, Iquitos 752 km), Paraguay-river ports, Lake Ladoga/Onega ports, Saimaa (Finland) and Dudinka;
- `route_costs(max_port_lane_km=...)` turns these into land-only terminals for robustness checks.

**Validation** (all checks passed):
- unsimplified-graph Dijkstra vs. the packaged matrix: largest relative difference 5.9e-8, and the same pairs are reachable;
- triangle-inequality violations: 0;
- road km / geodesic km: minimum 0.99999996; 5th / 50th / 95th percentiles 1.18 / 1.37 / 2.35;
- full multimodal Dijkstra vs. `route_costs`: largest relative difference 4.6e-8 under all three weightings.

**Road spot checks:**

| Route | Arteries |
|---|---|
| Paris–Berlin | 1,046 km |
| Madrid–Warsaw | 2,817 km |
| Beijing–Shanghai | 1,176 km |
| Delhi–Mumbai | 1,356 km |
| Sydney–Melbourne | 857 km |
| Johannesburg–Cape Town | 1,395 km |
| Shanghai–Rotterdam (all road) | 11,019 km |

**Sea spot checks** (land cost 4, sea cost 1):

| Route and scenario | Port cost 150 | Port cost 1,000 |
|---|---|---|
| Shanghai–Rotterdam, open | 19,567 sea km, 1 leg (via Suez) | same route |
| Suez + Bab-el-Mandeb passages closed | Ships still call inside the Red Sea and bridge both straits by land (Aden→Mokha, Eilat→Ashkelon): 627 land km + 19,360 sea km, 3 legs | Cape of Good Hope: 25,003 sea km, 1 leg |
| `closed_regions=["red_sea"]` | Persian Gulf land bridge: 1,553 land km, 2 legs | Cape of Good Hope: 25,003 sea km, 1 leg |

For Red Sea counterfactuals, close the region, not just the two passages. The port cost governs whether land bridges compete with the Cape route, so report results across a range of port costs.

**Measured resource use** (def-sm, 24 cores):

| Stage | Wall time | Peak memory |
|---|---|---|
| ingest | ~1 min per country | 2.1 GB |
| merge + gap repair | 5.7 min | 16 GB |
| terminals | 4 min | 11.4 GB |
| simplify | 1.8 min | 9.4 GB |
| assemble | 2 s | — |
| landmatrix | ~1 s per 64 sources | — |
| package + validation | under 1 min | — |

Querying all city pairs on a 2-core laptop with Numba:
- sea matrix: 2.9 s;
- full 2,075 × 2,075 `route_costs`: 19 s (23 s on the first call, including compilation).
