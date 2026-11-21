"""Inspection chips around a reference feature, for a human to digitise from."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import from_bounds

from sbf.core.config import Settings, load_settings
from sbf.core.crs import transform_geometry
from sbf.ingest.modern.chips import stretch
from sbf.ingest.refine.core import RefinementError

_M_PER_DEG = 111_320.0


def export_feature_chip(
    feature_id: str,
    raster_path: str | Path,
    out_png: str | Path,
    half_size_m: float = 1500.0,
    center_lonlat: tuple[float, float] | None = None,
    settings: Settings | None = None,
) -> Path:
    """Render a PNG chip (scale bar, original point marked) around the current geometry.

    BASE has no geometry: pass `center_lonlat`, labelled "human-supplied centre".
    """
    from sbf.core.aoi import load_reference_features

    s = settings or load_settings()
    cur = load_reference_features(s).set_index("feature_id")
    if feature_id not in cur.index:
        raise RefinementError(f"unknown feature_id {feature_id!r}")
    row = cur.loc[feature_id]
    orig = load_reference_features(s, refined=False).set_index("feature_id").loc[feature_id]
    geom = row["geometry"]
    has_geom = geom is not None and not geom.is_empty
    if center_lonlat is not None:
        lon, lat = center_lonlat
        label = "human-supplied centre"
    elif has_geom:
        lon, lat = geom.centroid.x, geom.centroid.y
        label = "refined geometry" if row["refined"] else "original geometry"
    else:
        raise RefinementError(
            f"{feature_id} has no geometry; supply --center-lon/--center-lat (human-supplied)"
        )

    with rasterio.open(raster_path) as src:
        crs = src.crs.to_string()
        cx, cy = transform_geometry(_pt(lon, lat), "EPSG:4326", crs).coords[0]
        geo = src.crs.is_geographic
        hx = half_size_m / (_M_PER_DEG * math.cos(math.radians(lat))) if geo else half_size_m
        hy = half_size_m / _M_PER_DEG if geo else half_size_m
        l, b, r, t = cx - hx, cy - hy, cx + hx, cy + hy
        if (
            r < src.bounds.left
            or l > src.bounds.right
            or t < src.bounds.bottom
            or b > src.bounds.top
        ):
            raise RefinementError("chip centre is outside the raster")
        win = from_bounds(l, b, r, t, src.transform)
        nb = src.count
        bands = [1, 2, 3] if nb >= 3 else [1]
        data = src.read(bands, window=win, boundless=True, fill_value=np.nan).astype("float32")
    img = np.moveaxis(stretch(data), 0, -1)
    if img.shape[-1] == 1:
        img = img[..., 0]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, extent=(l, r, b, t), cmap="gray", interpolation="nearest")
    if orig["geometry"] is not None and not orig["geometry"].is_empty:
        ox, oy = transform_geometry(orig["geometry"].centroid, "EPSG:4326", crs).coords[0]
        ax.plot(ox, oy, "+", color="red", ms=16, mew=2, label="original point")
    if row["refined"] and has_geom and geom.geom_type == "Polygon":
        gx, gy = transform_geometry(geom, "EPSG:4326", crs).exterior.xy
        ax.plot(gx, gy, color="yellow", lw=1.5, label="current polygon")
    if center_lonlat is not None:
        ax.plot(cx, cy, "o", mfc="none", mec="cyan", ms=14, label="human-supplied centre")
    bar_m = 500.0 if half_size_m >= 1000 else 100.0
    bw = bar_m / (_M_PER_DEG * math.cos(math.radians(lat))) if geo else bar_m
    x0, y0 = l + 0.05 * (r - l), b + 0.05 * (t - b)
    ax.plot([x0, x0 + bw], [y0, y0], color="white", lw=4)
    ax.text(x0, y0 + 0.02 * (t - b), f"{bar_m:.0f} m", color="white", fontsize=9)
    ax.set_title(
        f"{feature_id} ({row['precision']}) - {label}\n{Path(raster_path).name}", fontsize=9
    )
    ax.legend(loc="upper right", fontsize=8)
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def _pt(lon: float, lat: float):
    from shapely.geometry import Point

    return Point(lon, lat)
