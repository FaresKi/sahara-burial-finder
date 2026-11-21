"""Deterministic common 10 m grid per AOI (EPSG:32631).

Contract for ticket 09 (tile grid): the modern grid origin is the AOI's lon/lat bounding box,
densified and projected to the AOI CRS, then snapped OUTWARD to multiples of the pixel size
(10 m) in absolute UTM coordinates. Top-left = (floor(xmin/10)*10, ceil(ymax/10)*10);
width/height = ceil(extent / 10). Because the snap is to absolute multiples of 10 m, any
tile grid whose origin and tile steps are multiples of 10 m is pixel-aligned with every
modern layer. Same AOI + same settings => identical grid.
"""

from __future__ import annotations

import math

import shapely
from rasterio.crs import CRS
from rasterio.transform import Affine

from sbf.core.aoi import load_aois
from sbf.core.config import Settings, load_settings
from sbf.core.crs import transform_geometry

Grid = tuple[Affine, int, int, str]


def grid_from_bounds(
    bounds_lonlat: tuple[float, float, float, float], crs: str, res: float = 10.0
) -> Grid:
    """Grid covering a lon/lat bbox in `crs`, snapped outward to multiples of `res`."""
    box = shapely.box(*bounds_lonlat)
    dense = shapely.segmentize(box, 0.005)  # ~500 m: keeps the projected bounds tight
    xmin, ymin, xmax, ymax = transform_geometry(dense, "EPSG:4326", crs).bounds
    eps = 1e-9
    x0 = math.floor(xmin / res + eps) * res
    y1 = math.ceil(ymax / res - eps) * res
    width = math.ceil((xmax - x0) / res - eps)
    height = math.ceil((y1 - ymin) / res - eps)
    return Affine(res, 0.0, x0, 0.0, -res, y1), int(width), int(height), crs


def modern_grid(aoi_id: str, settings: Settings | None = None) -> Grid:
    """`(transform, width, height, crs)` of the AOI's common modern grid.

    `crs` is the EPSG string from settings (e.g. "EPSG:32631"); resolution is
    `settings.modern_resolution_m` (10 m).
    """
    s = settings or load_settings()
    aois = load_aois(s)
    row = aois[aois["aoi_id"] == aoi_id]
    if row.empty:
        raise KeyError(f"unknown AOI {aoi_id!r}; known: {sorted(aois['aoi_id'])}")
    return grid_from_bounds(
        tuple(row.geometry.iloc[0].bounds), s.crs(aoi_id), float(s.modern_resolution_m)
    )


def grid_bounds(grid: Grid) -> tuple[float, float, float, float]:
    """(left, bottom, right, top) of a grid."""
    t, w, h, _ = grid
    return (t.c, t.f + t.e * h, t.c + t.a * w, t.f)


def grid_profile(grid: Grid) -> dict:
    """rasterio profile fragment (crs, transform, width, height) for a grid."""
    t, w, h, crs = grid
    return {"crs": CRS.from_string(crs), "transform": t, "width": w, "height": h}
