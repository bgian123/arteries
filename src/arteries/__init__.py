"""Arteries: lowest-cost multimodal (road + sea) distances between world cities.

    import arteries
    net = arteries.load()
    res = net.route_costs(land_cost=3, sea_cost=1, port_cost=200)
    df = res.to_frame()
"""
from .network import SEA_REGIONS, Network, RouteResult, load
from .sea import DEFAULT_CLOSED, PASSAGES

__version__ = "0.1.0"
__all__ = ["load", "Network", "RouteResult", "PASSAGES", "SEA_REGIONS", "DEFAULT_CLOSED", "__version__"]
