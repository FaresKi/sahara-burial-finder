"""CRS helpers. Geographic data is EPSG:4326 (lon, lat); metric work is per-AOI UTM."""

from __future__ import annotations

from pyproj import Transformer
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from sbf.core.config import Settings, load_settings

WGS84 = "EPSG:4326"


def transform_geometry(geom: BaseGeometry, src: str, dst: str) -> BaseGeometry:
    """Reproject a shapely geometry (always_xy: lon/lat or x/y order)."""
    if src == dst:
        return geom
    t = Transformer.from_crs(src, dst, always_xy=True)
    return transform(t.transform, geom)


def aoi_crs(aoi_id: str, settings: Settings | None = None) -> str:
    """Metric CRS of an AOI from settings."""
    return (settings or load_settings()).crs(aoi_id)


def lonlat_to_xy(lon: float, lat: float, crs: str) -> tuple[float, float]:
    """WGS84 lon/lat to projected x/y in `crs`."""
    return Transformer.from_crs(WGS84, crs, always_xy=True).transform(lon, lat)


def xy_to_lonlat(x: float, y: float, crs: str) -> tuple[float, float]:
    """Projected x/y in `crs` to WGS84 lon/lat."""
    return Transformer.from_crs(crs, WGS84, always_xy=True).transform(x, y)


def buffer_point_m(lon: float, lat: float, radius_m: float, crs: str) -> BaseGeometry:
    """Circle of radius_m metres around a lon/lat point, returned in `crs`."""
    x, y = lonlat_to_xy(lon, lat, crs)
    return Point(x, y).buffer(radius_m)
