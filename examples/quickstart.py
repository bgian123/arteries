"""Quickstart on the small demo build (Baltics, Finland, Denmark, Benelux).
With the full release, replace the path by  net = arteries.load()."""
from pathlib import Path
import arteries

net = arteries.load(Path(__file__).with_name("demo_baltic_benelux_core.npz"))
print(f"{len(net.cities)} cities, {len(net.ports)} ports")

# 1) pure road distances
print(net.land_km(["Helsinki", "Rīga", "Amsterdam"], ["Tallinn", "Vilnius", "Antwerpen"]).round(1))

# 2) land 4x as costly per km as sea, 150 per port call
res = net.route_costs(land_cost=4, sea_cost=1, port_cost=150)
df = res.to_frame()
print(df[df.origin == "Helsinki"].sort_values("cost").head(8).to_string(index=False))

# 3) scenario: close a chokepoint (here Gibraltar; the Arctic Northwest Passage stays closed)
res_closed = net.route_costs(land_cost=4, sea_cost=1, port_cost=150, closed_passages=["northwest", "gibraltar"])
