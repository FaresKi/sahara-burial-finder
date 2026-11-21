"""Chip extraction around a point from a stack layer (used by the detectability notebook)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from sbf.core.crs import lonlat_to_xy
from sbf.ingest.modern.process import read_manifest, stack_dir


def extract_chip(
    path: str | Path, x: float, y: float, half_width_m: float, bands: tuple[int, ...] = (1,)
) -> np.ndarray | None:
    """(n, h, w) float32 chip centred on projected (x, y); None if the point is off the raster.

    Parts of the window outside the raster are NaN-padded.
    """
    with rasterio.open(path) as src:
        if not (
            src.bounds.left <= x <= src.bounds.right and src.bounds.bottom <= y <= src.bounds.top
        ):
            return None
        row, col = src.index(x, y)
        hw = round(half_width_m / abs(src.transform.a))
        win = Window(col - hw, row - hw, 2 * hw + 1, 2 * hw + 1)
        data = src.read(list(bands), window=win, boundless=True, fill_value=np.nan)
    return data.astype(np.float32)


def stretch(a: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Percentile stretch to [0, 1] for display; NaN preserved."""
    v = a[np.isfinite(a)]
    if v.size == 0:
        return np.zeros_like(a)
    p0, p1 = np.percentile(v, [lo, hi])
    return np.clip((a - p0) / (p1 - p0 + 1e-12), 0, 1)


def feature_chips(
    aoi_id: str, lon: float, lat: float, half_width_m: float = 1000.0, settings=None
) -> dict[str, np.ndarray | None]:
    """Chip per layer for a lon/lat point; S2 layers return 3-band RGB (B4, B3, B2)."""
    m = read_manifest(aoi_id, settings)
    if m is None:
        return {}
    x, y = lonlat_to_xy(lon, lat, m["crs"])
    out: dict[str, np.ndarray | None] = {}
    for name, ent in m["layers"].items():
        bands = (3, 2, 1) if name.startswith("s2_") else (1,)
        out[name] = extract_chip(
            stack_dir(aoi_id, settings) / ent["file"], x, y, half_width_m, bands
        )
    return out
