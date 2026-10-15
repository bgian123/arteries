"""Distance computations on the WGS84 ellipsoid.

Two regimes are used throughout Arteries:

* **Short segments** (individual OSM road segments, typically < 1 km):
  ``segment_length_km`` uses the local ellipsoidal metric
  ``ds^2 = (M(phi) dphi)^2 + (N(phi) cos(phi) dlambda)^2`` evaluated at the
  mid-latitude. M and N are the WGS84 meridional and prime-vertical radii of
  curvature. Relative error versus the exact (Karney) geodesic is O((d/R)^2),
  i.e. < 1e-6 for 5 km segments and < 1e-4 for 50 km segments.

* **Long connectors** (city/port snapping edges, sea polylines, bridges):
  ``geodesic_km`` uses pyproj's implementation of Karney (2013) when pyproj is
  installed (exact to ~15 nm), and falls back to the haversine formula on the
  WGS84 mean radius (error <= 0.5 %).

All inputs are decimal degrees; all outputs are kilometers.
"""
from __future__ import annotations

import numpy as np

# WGS84 constants
WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
MEAN_RADIUS_KM = 6371.0088  # IUGG mean radius R1 = (2a + b) / 3

try:  # optional, exact geodesics
    from pyproj import Geod as _Geod

    _GEOD = _Geod(ellps="WGS84")
except Exception:  # pragma: no cover - optional dependency
    _GEOD = None


def segment_length_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized WGS84 length of short segments (km)."""
    lat1 = np.asarray(lat1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    phi = np.radians(0.5 * (lat1 + lat2))
    dphi = np.radians(lat2 - lat1)
    dlon = lon2 - lon1
    dlon = (dlon + 180.0) % 360.0 - 180.0  # antimeridian-safe
    dlam = np.radians(dlon)
    s2 = np.sin(phi) ** 2
    w = np.sqrt(1.0 - WGS84_E2 * s2)
    n_rad = WGS84_A_KM / w  # prime vertical radius of curvature
    m_rad = WGS84_A_KM * (1.0 - WGS84_E2) / w**3  # meridional radius
    return np.hypot(m_rad * dphi, n_rad * np.cos(phi) * dlam)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized great-circle distance on the mean-radius sphere (km)."""
    p1 = np.radians(np.asarray(lat1, dtype=np.float64))
    p2 = np.radians(np.asarray(lat2, dtype=np.float64))
    dl = np.radians(np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, dtype=np.float64))
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * MEAN_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def geodesic_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Exact WGS84 geodesic distance (Karney) if pyproj is present, else haversine."""
    if _GEOD is None:
        return haversine_km(lat1, lon1, lat2, lon2)
    lat1, lon1, lat2, lon2 = (np.asarray(x, dtype=np.float64) for x in (lat1, lon1, lat2, lon2))
    _, _, d = _GEOD.inv(lon1, lat1, lon2, lat2)
    return np.asarray(d) / 1000.0


def polyline_length_km(lats, lons) -> float:
    """Length of a polyline as the sum of its (exact) geodesic pieces."""
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    if lats.size < 2:
        return 0.0
    return float(np.sum(geodesic_km(lats[:-1], lons[:-1], lats[1:], lons[1:])))


def to_unit_xyz(lat, lon) -> np.ndarray:
    """Unit-sphere Cartesian coordinates for KD-tree queries (chord metric)."""
    p = np.radians(np.asarray(lat, dtype=np.float64))
    l = np.radians(np.asarray(lon, dtype=np.float64))
    c = np.cos(p)
    return np.column_stack([c * np.cos(l), c * np.sin(l), np.sin(p)])


def km_to_chord(km: float) -> float:
    """Convert an arc length (km, mean sphere) to unit-sphere chord length."""
    return 2.0 * np.sin(min(km / MEAN_RADIUS_KM, np.pi) / 2.0)


def chord_to_km(chord) -> np.ndarray:
    return 2.0 * MEAN_RADIUS_KM * np.arcsin(np.clip(np.asarray(chord) / 2.0, 0.0, 1.0))
