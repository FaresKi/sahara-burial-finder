"""Validity ("support") mask of a georeferenced frame.

A warp is only trustworthy near GCP support: away from the GCPs a polynomial/TPS extrapolates
(a frame-wide fit scored 4 m on checkpoints yet was ~85 m off where GCPs were sparse). The mask is
1 where the warped pixel maps back to a frame pixel within `D` (default `SUPPORT_DIST_FRAC` = 10%
of the frame/strip diagonal, frame px) of the convex hull of the GCPs, else 0. It is written on
exactly the grid of `{ID}_georef.tif` as `{ID}_support.tif` (uint8 COG, nodata 0). Strips with a
fallback model (no GCPs of their own) contribute nothing. Change detection reads it with
`sbf.georef.registry.load_support`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import rasterio
import rasterio.shutil
from rasterio.windows import Window
from shapely import contains_xy
from shapely.geometry import MultiPoint

from sbf.georef.models import SUPPORT_DIST_FRAC, UTM_CRS
from sbf.georef.warp import Grid, block_source_coords, open_tiled_output


def support_polygon(px, extent, dist_frac: float = SUPPORT_DIST_FRAC):
    """Convex hull of the GCPs (frame px) buffered by `dist_frac` x diagonal of `extent`;
    None when there are no GCPs."""
    px = np.asarray(px, float)
    if len(px) == 0:
        return None
    x0, y0, x1, y1 = extent
    d = dist_frac * float(np.hypot(x1 - x0, y1 - y0))
    return MultiPoint(px).convex_hull.buffer(d)


def grid_of(transform, shape: tuple[int, int]) -> Grid:
    return Grid(transform.c, transform.f, transform.a, int(shape[1]), int(shape[0]))


def frame_support_fn(model, shape, poly, *, step: int = 16) -> Callable:
    """Block function (grid, r0, c0, nr, nc) -> bool mask for a single frame-wide model."""
    from sbf.georef.fit import invert_on_lattice

    inv = invert_on_lattice(model, shape)
    h, w = shape[:2]

    def fn(grid: Grid, r0: int, c0: int, nr: int, nc: int) -> np.ndarray:
        sx, sy = block_source_coords(inv, grid, r0, c0, nr, nc, step)
        inside = (sx >= 0) & (sx <= w) & (sy >= 0) & (sy <= h)
        return inside & contains_xy(poly, sx, sy)

    return fn


def write_support(
    path: str | Path,
    grid: Grid,
    block_fn: Callable,
    *,
    crs: str = UTM_CRS,
    block: int = 1024,
    prov: dict | None = None,
) -> Path:
    """Stream `block_fn` over `grid` into a uint8 COG (1 = supported, 0 = not)."""
    path = Path(path)
    tmp = path.with_name(f".{path.stem}_tiled.tif")
    with open_tiled_output(tmp, grid, np.uint8, 0, crs) as dst:
        for r0, c0, nr, nc in grid.blocks(block):
            m = block_fn(grid, r0, c0, nr, nc).astype("uint8")
            dst.write(m, 1, window=Window(c0, r0, nc, nr))
    rasterio.shutil.copy(tmp, path, driver="COG", compress="DEFLATE")
    tmp.unlink(missing_ok=True)
    if prov:
        from sbf.core.provenance import write_prov

        write_prov(path, **prov)
    return path
