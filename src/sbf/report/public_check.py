"""Leak guard for outputs/public (SPEC section 9.4, ARCHITECTURE restricted outputs).

Scans every text file under `outputs/public` for coordinate-like content and fails unless it
matches something already public: a reference feature (geometry, refined or not), an AOI vertex
or a catalogue footprint vertex, within a small tolerance. Three detectors:

* decimal-degree pairs: adjacent latitude and longitude tokens (>= 3 decimals, inside an AOI's
  lat/lon range, plus margin);
* UTM pairs: adjacent easting and northing tokens inside an AOI's EPSG:32631 extent (plus margin);
* tile IDs (`REG-64-00012-00034`), which encode the grid position: each must intersect a
  reference feature (plus a margin).

Violations carry file, line and kind only. The offending values are never printed (printing a
leaked coordinate would be a second leak).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from shapely import wkt
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from sbf.core.aoi import load_aois, load_reference_features
from sbf.core.config import Settings, load_settings
from sbf.core.crs import lonlat_to_xy, xy_to_lonlat

TEXT_SUFFIXES = {".md", ".csv", ".json", ".html", ".txt", ".geojson", ".yaml", ".yml", ".wkt"}
PAIR_GAP = 14  # max characters (no digits) between the two numbers of a pair
DEG_TOL = 0.003  # about 300 m
LATLON_MARGIN = 0.05  # degrees added to AOI ranges
UTM_MARGIN = 2000.0  # metres added to AOI UTM extents
TILE_MARGIN_M = 64.0

_NUM = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.])")
_TILE = re.compile(r"\b([A-Z]{2,5}-\d{2,4}-\d{5}-\d{5})\b")


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    kind: str  # latlon_pair | utm_pair | tile_id

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: {self.kind} not matching a public reference"


@dataclass
class _Ranges:
    aoi: str
    lon: tuple[float, float]
    lat: tuple[float, float]
    crs: str
    x: tuple[float, float]
    y: tuple[float, float]


def _aoi_ranges(settings: Settings) -> list[_Ranges]:
    out = []
    for r in load_aois(settings).itertuples():
        lon0, lat0, lon1, lat1 = r.geometry.bounds
        crs = settings.crs(r.aoi_id)
        corners = [lonlat_to_xy(lo, la, crs) for lo in (lon0, lon1) for la in (lat0, lat1)]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        out.append(
            _Ranges(
                r.aoi_id,
                (lon0 - LATLON_MARGIN, lon1 + LATLON_MARGIN),
                (lat0 - LATLON_MARGIN, lat1 + LATLON_MARGIN),
                crs,
                (float(np.min(xs)) - UTM_MARGIN, float(np.max(xs)) + UTM_MARGIN),
                (float(np.min(ys)) - UTM_MARGIN, float(np.max(ys)) + UTM_MARGIN),
            )
        )
    return out


class PublicReferences:
    """What is allowed to appear: reference geometries, AOI vertices, footprint vertices."""

    def __init__(self, settings: Settings):
        self.s = settings
        refs = load_reference_features(settings)
        self.ref_geoms: list[BaseGeometry] = [g for g in refs.geometry if g is not None]
        pts: list[tuple[float, float]] = []
        for g in load_aois(settings).geometry:
            pts += _vertices(g)
        cov = settings.outputs_public / "coverage.csv"
        if cov.exists():
            import pandas as pd

            df = pd.read_csv(cov, usecols=lambda c: c == "footprint_wkt")
            for w in df.get("footprint_wkt", []):
                if isinstance(w, str) and w:
                    try:
                        pts += _vertices(wkt.loads(w))
                    except Exception:  # noqa: BLE001, S112 - malformed WKT: not a reference
                        continue
        self.vertices = np.array(pts) if pts else np.zeros((0, 2))

    def lonlat_ok(self, lon: float, lat: float) -> bool:
        p = Point(lon, lat)
        if any(g.distance(p) <= DEG_TOL for g in self.ref_geoms):
            return True
        if len(self.vertices):
            d = np.hypot(self.vertices[:, 0] - lon, self.vertices[:, 1] - lat)
            return bool(d.min() <= DEG_TOL)
        return False

    def tile_ok(self, tile_id: str) -> bool:
        from sbf.features.tiling import tile_geometries

        try:
            tiles = tile_geometries([tile_id], settings=self.s)
        except Exception:  # noqa: BLE001 - unparsable id: not an allowed tile
            return False
        geom = tiles.geometry.iloc[0]
        crs = str(tiles.crs)
        for g in self.ref_geoms:
            from sbf.core.crs import transform_geometry

            if geom.distance(transform_geometry(g, "EPSG:4326", crs)) <= TILE_MARGIN_M:
                return True
        return False


def _is_dec(tok: str) -> bool:
    """Decimal-degree-like token: has a fractional part of >= 3 digits."""
    return "." in tok and len(tok.split(".")[1]) >= 3


def _vertices(g: BaseGeometry) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for part in getattr(g, "geoms", [g]):
        if hasattr(part, "exterior"):
            out += [(x, y) for x, y, *_ in part.exterior.coords]
            for ring in part.interiors:
                out += [(x, y) for x, y, *_ in ring.coords]
        elif hasattr(part, "coords"):
            out += [(x, y) for x, y, *_ in part.coords]
    return out


def _scan_text(
    name: str, text: str, ranges: list[_Ranges], refs: PublicReferences
) -> list[Violation]:
    found: list[Violation] = []
    seen: set[tuple[int, str]] = set()
    lines = text.splitlines() or [text]
    # windows are applied on the whole text so that pairs split over lines/cells are caught
    offsets = np.cumsum([0] + [len(x) + 1 for x in lines])

    def line_of(pos: int) -> int:
        return int(np.searchsorted(offsets, pos, side="right"))

    def add(pos: int, kind: str) -> None:
        key = (line_of(pos), kind)
        if key not in seen:
            seen.add(key)
            found.append(Violation(name, line_of(pos), kind))

    toks = [(m.start(), m.end(), m.group(1)) for m in _NUM.finditer(text)]
    i = 0
    while i < len(toks) - 1:
        s0, e0, a = toks[i]
        s1, _e1, b = toks[i + 1]
        gap = text[e0:s1]
        if len(gap) > PAIR_GAP or any(ch.isdigit() for ch in gap):
            i += 1  # only adjacent numbers form a coordinate pair
            continue
        va, vb = float(a), float(b)
        consumed = False
        for rg in ranges:
            if _is_dec(a) and _is_dec(b):
                for lon, lat in ((va, vb), (vb, va)):
                    if rg.lon[0] <= lon <= rg.lon[1] and rg.lat[0] <= lat <= rg.lat[1]:
                        consumed = True
                        if not refs.lonlat_ok(lon, lat):
                            add(s0, "latlon_pair")
            for x, y in ((va, vb), (vb, va)):
                if rg.x[0] <= x <= rg.x[1] and rg.y[0] <= y <= rg.y[1]:
                    consumed = True
                    lon, lat = xy_to_lonlat(x, y, rg.crs)
                    if not refs.lonlat_ok(lon, lat):
                        add(s0, "utm_pair")
        i += 2 if consumed else 1  # a matched pair consumes both numbers (no cross-pairing)
    for m in _TILE.finditer(text):
        if not refs.tile_ok(m.group(1)):
            add(m.start(), "tile_id")
    return found


def check_public(settings: Settings | None = None) -> list[Violation]:
    """Scan outputs/public; return violations (empty list = clean)."""
    s = settings or load_settings()
    root = s.outputs_public
    if not root.exists():
        return []
    ranges = _aoi_ranges(s)
    refs = PublicReferences(s)
    out: list[Violation] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = p.read_text(errors="strict")
        except UnicodeDecodeError:
            continue
        out += _scan_text(str(p.relative_to(root)), text, ranges, refs)
    return out
