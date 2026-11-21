"""Reference imagery on the metric grid: modern S2 stack (default), DEM hillshade, or user VHR.

Never fetches Esri/Google tiles (ToS). A VHR GeoTIFF supplied via `--reference` is the
human's responsibility to be licensed for this use.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds

from sbf.core.config import Settings, load_settings
from sbf.georef.models import UTM_CRS
from sbf.ingest.modern.process import slope_degrees, stack_dir

PAN_BANDS = ("B2", "B3", "B4", "B8")


@dataclass
class ReferenceLayer:
    name: str
    array: np.ndarray  # (h, w) float32 gray, NaN = nodata
    transform: Affine
    crs: str = UTM_CRS
    sources: list[str] = field(default_factory=list)
    terrain: dict[str, np.ndarray] = field(default_factory=dict)  # slope / roughness on same grid

    @property
    def resolution_m(self) -> float:
        return abs(self.transform.a)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        h, w = self.array.shape
        return (
            self.transform.c,
            self.transform.f + self.transform.e * h,
            self.transform.c + self.transform.a * w,
            self.transform.f,
        )

    def mask_stack(self) -> dict[str, np.ndarray]:
        return {"s2": self.array, **self.terrain}


def hillshade(dem: np.ndarray, res: float, azimuth: float = 315.0, altitude: float = 45.0):
    """Classic Horn-style hillshade in [0, 1] (NaN preserved)."""
    d = dem.astype("float64")
    dzdy, dzdx = np.gradient(d, res)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    az, alt = np.radians(360.0 - azimuth + 90.0), np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    out = np.clip(hs, 0, 1).astype("float32")
    out[~np.isfinite(d)] = np.nan
    return out


def _window(src, bounds) -> Window | None:
    if bounds is None:
        return None
    w = from_bounds(*bounds, transform=src.transform)
    c0, r0 = math.floor(w.col_off), math.floor(w.row_off)
    c1, r1 = math.ceil(w.col_off + w.width), math.ceil(w.row_off + w.height)
    return Window(c0, r0, c1 - c0, r1 - r0).intersection(Window(0, 0, src.width, src.height))


def _read(path: Path, bands: list[int], bounds) -> tuple[np.ndarray, Affine]:
    with rasterio.open(path) as src:
        win = _window(src, bounds)
        arr = src.read(bands, window=win, masked=True).astype("float32").filled(np.nan)
        tr = src.window_transform(win) if win is not None else src.transform
    return arr, tr


def load_stack_reference(
    aoi_id: str,
    settings: Settings | None = None,
    *,
    mode: str = "pan",
    bounds_xy: tuple[float, float, float, float] | None = None,
) -> ReferenceLayer:
    """Reference from `data/processed/modern/{AOI}` (ticket 05). `mode`: `pan` (mean of
    B2,B3,B4,B8), a band name such as `B4`, or `hillshade` (from `dem`). `bounds_xy` crops."""
    s = settings or load_settings()
    sd = stack_dir(aoi_id, s)
    mf = sd / "stack.json"
    if not mf.exists():
        raise FileNotFoundError(f"modern stack for {aoi_id} not built ({mf}); run build-stack")
    layers = json.loads(mf.read_text())["layers"]
    srcs: list[str] = []
    terrain: dict[str, np.ndarray] = {}
    if mode == "hillshade":
        arr, tr = _read(sd / layers["dem"]["file"], [1], bounds_xy)
        gray = hillshade(arr[0], abs(tr.a))
        srcs.append(str(sd / layers["dem"]["file"]))
        name = "dem_hillshade"
    else:
        info = layers["s2_median"]
        names = [b["name"] for b in info["bands"]]
        want = list(PAN_BANDS) if mode == "pan" else [mode]
        idx = [names.index(b) + 1 for b in want if b in names]
        if not idx:
            raise ValueError(f"S2 bands {want} not in stack ({names})")
        arr, tr = _read(sd / info["file"], idx, bounds_xy)
        gray = np.nanmean(arr, axis=0).astype("float32") if len(idx) > 1 else arr[0]
        srcs.append(str(sd / info["file"]))
        name = "s2_median_" + ("pan" if mode == "pan" else mode)
    for key, layer in (("slope", "dem_slope"), ("roughness", "dem_roughness")):
        if layer in layers:
            t, _ = _read(sd / layers[layer]["file"], [1], bounds_xy)
            if t.shape[1:] == gray.shape:
                terrain[key] = t[0]
    if "slope" not in terrain and "dem" in layers:
        d, _ = _read(sd / layers["dem"]["file"], [1], bounds_xy)
        if d.shape[1:] == gray.shape:
            terrain["slope"] = slope_degrees(d[0], abs(tr.a))
    if mode == "hillshade":
        terrain = {}  # mask rule needs S2 texture; hillshade runs unmasked
    return ReferenceLayer(name, gray, tr, UTM_CRS, srcs, terrain)


def load_file_reference(
    path: str | Path,
    *,
    crs: str = UTM_CRS,
    bounds_xy: tuple[float, float, float, float] | None = None,
) -> ReferenceLayer:
    """A user-supplied (licensed) GeoTIFF, band 1, warped on the fly to `crs`."""
    path = Path(path)
    with (
        rasterio.open(path) as src,
        WarpedVRT(src, crs=crs, resampling=rasterio.enums.Resampling.bilinear) as vrt,
    ):
        win = _window(vrt, bounds_xy)
        arr = vrt.read(1, window=win, masked=True).astype("float32").filled(np.nan)
        tr = vrt.window_transform(win) if win is not None else vrt.transform
    return ReferenceLayer(f"file:{path.name}", arr, tr, crs, [str(path)])
