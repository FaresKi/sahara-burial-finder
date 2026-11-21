"""Deterministic tile grids (ARCHITECTURE.md "Tile grid").

Grid origin rule
----------------
Everything is in the AOI's metric CRS (EPSG:32631). The origin is the south-west corner of the
AOI bounding box (WGS84 polygon densified, projected to EPSG:32631) snapped DOWN to a multiple
of 128 m. 128 is a multiple of both tile sizes (64, 128) and of the 32 m / 64 m steps used at
50 % overlap, so grids of different sizes share lattice points. The 10 m modern raster grid
(``sbf.ingest.modern``) snaps to 10 m on its own and is deliberately NOT aligned with the tile
lattice: ``tile_windows`` rounds each tile to the nearest pixel offset and returns chips of one
fixed shape (``round(size_m / pixel_size)``).

Indexing: ``col`` increases east and ``row`` increases north from the origin. Tile ``(col, row)``
spans ``[ox + col*step, ox + col*step + size] x [oy + row*step, oy + row*step + size]`` with
``step = size_m * (1 - overlap)``. The grid covers the whole AOI bounding box.

``tile_id = "{aoi}-{size}-{col:05d}-{row:05d}"``. D8: only IDs and counts are ever logged.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio.windows import Window

from sbf.core.aoi import load_aois
from sbf.core.config import Settings, load_settings
from sbf.core.crs import transform_geometry

ORIGIN_SNAP_M = 128
_ID_RE = re.compile(r"^(?P<aoi>[A-Za-z0-9]+)-(?P<size>\d+)-(?P<col>\d{5,})-(?P<row>\d{5,})$")


def make_tile_id(aoi_id: str, size_m: int, col: int, row: int) -> str:
    """Format a tile ID: ``"{aoi}-{size}-{col:05d}-{row:05d}"``."""
    return f"{aoi_id}-{int(size_m)}-{col:05d}-{row:05d}"


def parse_tile_id(tile_id: str) -> tuple[str, int, int, int]:
    """Inverse of `make_tile_id`: returns ``(aoi_id, size_m, col, row)``."""
    m = _ID_RE.match(tile_id)
    if not m:
        raise ValueError(f"malformed tile_id {tile_id!r}")
    return m["aoi"], int(m["size"]), int(m["col"]), int(m["row"])


def aoi_bounds_m(
    aoi_id: str, settings: Settings | None = None
) -> tuple[float, float, float, float]:
    """AOI bounding box (minx, miny, maxx, maxy) in the AOI metric CRS."""
    s = settings or load_settings()
    aois = load_aois(s).set_index("aoi_id")
    if aoi_id not in aois.index:
        raise KeyError(f"unknown AOI {aoi_id!r}; known: {sorted(aois.index)}")
    geom = shapely.segmentize(aois.loc[aoi_id, "geometry"], 0.01)  # curved edges in UTM
    return transform_geometry(geom, "EPSG:4326", s.crs(aoi_id)).bounds


def grid_origin(aoi_id: str, settings: Settings | None = None) -> tuple[float, float]:
    """Tile lattice origin: AOI SW corner snapped down to a multiple of 128 m."""
    minx, miny, _, _ = aoi_bounds_m(aoi_id, settings)
    return (
        math.floor(minx / ORIGIN_SNAP_M) * ORIGIN_SNAP_M,
        math.floor(miny / ORIGIN_SNAP_M) * ORIGIN_SNAP_M,
    )


def _step(size_m: float, overlap: float) -> float:
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    return size_m * (1 - overlap)


def make_grid(
    aoi_id: str,
    size_m: int,
    overlap: float,
    settings: Settings | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    """Tile grid for an AOI; columns ``tile_id, aoi_id, size_m, geometry`` (metric CRS).

    ``bounds`` (minx, miny, maxx, maxy, metric CRS) optionally restricts the grid to tiles
    fully inside that box (e.g. a raster footprint); IDs are identical to the full grid's.
    Deterministic: same inputs give the same IDs in the same (row-major, col fastest) order.
    """
    s = settings or load_settings()
    crs = s.crs(aoi_id)
    ox, oy = grid_origin(aoi_id, s)
    _, _, maxx, maxy = aoi_bounds_m(aoi_id, s)
    step = _step(size_m, overlap)
    ncol = math.ceil((maxx - ox - size_m) / step) + 1
    nrow = math.ceil((maxy - oy - size_m) / step) + 1
    c0, c1, r0, r1 = 0, ncol - 1, 0, nrow - 1
    if bounds is not None:
        bx0, by0, bx1, by1 = bounds
        c0 = max(c0, math.ceil((bx0 - ox) / step - 1e-9))
        r0 = max(r0, math.ceil((by0 - oy) / step - 1e-9))
        c1 = min(c1, math.floor((bx1 - size_m - ox) / step + 1e-9))
        r1 = min(r1, math.floor((by1 - size_m - oy) / step + 1e-9))
    cols = np.arange(c0, c1 + 1)
    rows = np.arange(r0, r1 + 1)
    if cols.size == 0 or rows.size == 0:
        return gpd.GeoDataFrame(
            {"tile_id": [], "aoi_id": [], "size_m": np.array([], dtype=int)}, geometry=[], crs=crs
        )
    cc, rr = np.meshgrid(cols, rows)
    cc, rr = cc.ravel(), rr.ravel()
    x0 = ox + cc * step
    y0 = oy + rr * step
    geoms = shapely.box(x0, y0, x0 + size_m, y0 + size_m)
    ids = [make_tile_id(aoi_id, size_m, int(c), int(r)) for c, r in zip(cc, rr, strict=True)]
    return gpd.GeoDataFrame(
        {"tile_id": ids, "aoi_id": aoi_id, "size_m": int(size_m)}, geometry=geoms, crs=crs
    )


def tile_geometries(
    tile_ids: Sequence[str], overlap: float | None = None, settings: Settings | None = None
) -> gpd.GeoDataFrame:
    """Rebuild tile polygons from IDs alone (no full grid needed). Same columns as `make_grid`."""
    s = settings or load_settings()
    ov = s.overlap if overlap is None else overlap
    parsed = [parse_tile_id(t) for t in tile_ids]
    if not parsed:
        return make_grid_empty(s)
    aois = {p[0] for p in parsed}
    if len(aois) != 1:
        raise ValueError("tile_geometries needs tiles of a single AOI")
    aoi = aois.pop()
    ox, oy = grid_origin(aoi, s)
    size = np.array([p[1] for p in parsed], dtype=float)
    step = size * (1 - ov)
    x0 = ox + np.array([p[2] for p in parsed]) * step
    y0 = oy + np.array([p[3] for p in parsed]) * step
    return gpd.GeoDataFrame(
        {"tile_id": list(tile_ids), "aoi_id": aoi, "size_m": size.astype(int)},
        geometry=shapely.box(x0, y0, x0 + size, y0 + size),
        crs=s.crs(aoi),
    )


def make_grid_empty(settings: Settings) -> gpd.GeoDataFrame:
    """Empty grid frame with the standard columns."""
    return gpd.GeoDataFrame(
        {"tile_id": [], "aoi_id": [], "size_m": np.array([], dtype=int)},
        geometry=[],
        crs="EPSG:32631",
    )


def apply_exclusions(
    grid: gpd.GeoDataFrame, exclusions_gdf: gpd.GeoDataFrame | None
) -> gpd.GeoDataFrame:
    """Return a copy of ``grid`` with a bool ``excluded`` column (tile intersects any exclusion).

    ``exclusions_gdf`` may be in any CRS (reprojected to the grid's); None/empty excludes nothing.
    """
    out = grid.copy()
    out["excluded"] = False
    if exclusions_gdf is None or len(exclusions_gdf) == 0 or len(out) == 0:
        return out
    ex = exclusions_gdf[~exclusions_gdf.geometry.isna() & ~exclusions_gdf.geometry.is_empty]
    if ex.crs is None:
        raise ValueError("exclusions have no CRS")
    ex = ex.to_crs(out.crs)
    union = shapely.union_all(ex.geometry.values)
    shapely.prepare(union)
    out["excluded"] = shapely.intersects(out.geometry.values, union)
    return out


def tile_windows(
    grid: gpd.GeoDataFrame,
    raster_path: str | Path,
    band: int | Sequence[int] = 1,
    min_valid_frac: float = 0.0,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(tile_id, chip)`` for each grid tile overlapping the raster.

    Raster must be north-up and in the grid CRS (EPSG:32631); no reprojection. Chips are
    float32 with nodata/masked/outside-raster pixels set to NaN, of fixed shape
    ``(n, n)`` (``band`` an int) or ``(bands, n, n)`` (``band`` a sequence), where
    ``n = round(size_m / pixel_size)``. Partial tiles are NaN-padded. A tile is skipped when it has
    no valid pixel or its valid fraction is below ``min_valid_frac``.
    """
    with rasterio.open(raster_path) as src:
        t = src.transform
        if t.b != 0 or t.d != 0 or t.e >= 0:
            raise ValueError("tile_windows requires a north-up raster")
        px, py = t.a, -t.e
        indexes = [band] if isinstance(band, int) else list(band)
        for tile_id, geom in zip(grid["tile_id"], grid.geometry, strict=True):
            x0, y0, x1, y1 = geom.bounds
            n_c, n_r = round((x1 - x0) / px), round((y1 - y0) / py)
            col_off = round((x0 - t.c) / px)
            row_off = round((t.f - y1) / py)
            if (
                col_off >= src.width
                or row_off >= src.height
                or col_off + n_c <= 0
                or row_off + n_r <= 0
            ):
                continue
            win = Window(col_off, row_off, n_c, n_r)
            inside = (
                col_off >= 0
                and row_off >= 0
                and col_off + n_c <= src.width
                and row_off + n_r <= src.height
            )
            data = src.read(indexes, window=win, masked=True, boundless=not inside)
            arr = np.ma.filled(data.astype("float32"), np.nan)
            valid = float(np.isfinite(arr).all(axis=0).mean())
            if valid == 0 or valid < min_valid_frac:
                continue
            yield tile_id, (arr[0] if isinstance(band, int) else arr)
