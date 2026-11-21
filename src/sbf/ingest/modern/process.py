"""Pure local post-processing: co-registration, terrain derivatives, stack building.

Nothing here touches the network or Earth Engine.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage

from sbf.core.config import Settings, load_settings
from sbf.core.provenance import read_prov
from sbf.core.raster import write_cog
from sbf.ingest.modern.grid import Grid, grid_bounds, grid_profile, modern_grid
from sbf.ingest.modern.layers import LAYERS, NODATA_RAW, PRODUCTS, LayerSpec

ROUGHNESS_WINDOW_PX = 5
PALSAR_CAL_DB = -83.0  # gamma0 [dB] = 10*log10(DN^2) + CF, CF = -83 (JAXA)


# --------------------------------------------------------------------------- paths


def raw_dir(aoi_id: str, settings: Settings | None = None) -> Path:
    return (settings or load_settings()).data_raw / "modern" / aoi_id


def stack_dir(aoi_id: str, settings: Settings | None = None) -> Path:
    return (settings or load_settings()).data_processed / "modern" / aoi_id


def manifest_path(aoi_id: str, settings: Settings | None = None) -> Path:
    return stack_dir(aoi_id, settings) / "stack.json"


def read_manifest(aoi_id: str, settings: Settings | None = None) -> dict | None:
    """Parsed stack.json, or None if the stack has not been built."""
    p = manifest_path(aoi_id, settings)
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------------- co-registration


def reproject_to_grid(
    src_path: str | Path,
    grid: Grid,
    *,
    bands: tuple[int, ...] | None = None,
    resampling: str = "bilinear",
    src_nodata: float | None = NODATA_RAW,
) -> np.ndarray:
    """Resample selected 1-based `bands` of a raster onto `grid`; returns (n, h, w) float32.

    Nodata pixels (the file's nodata, else `src_nodata`) become NaN.
    """
    transform, width, height, crs = grid
    with rasterio.open(src_path) as src:
        idx = list(bands or range(1, src.count + 1))
        nd = src.nodata if src.nodata is not None else src_nodata
        out = np.full((len(idx), height, width), np.nan, dtype=np.float32)
        for i, b in enumerate(idx):
            reproject(
                source=rasterio.band(src, b),
                destination=out[i],
                src_nodata=nd,
                dst_transform=transform,
                dst_crs=crs,
                dst_nodata=np.nan,
                resampling=Resampling[resampling],
            )
    return out


# --------------------------------------------------------------------------- terrain


def slope_degrees(dem: np.ndarray, xres: float, yres: float | None = None) -> np.ndarray:
    """Slope in degrees from central differences (one-sided at edges). NaN propagates."""
    yres = xres if yres is None else yres
    d = dem.astype(np.float64)
    dzdy, dzdx = np.gradient(d, abs(yres), abs(xres))
    out = np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)
    out[~np.isfinite(d)] = np.nan
    return out


def terrain_ruggedness_index(dem: np.ndarray) -> np.ndarray:
    """Riley TRI: sqrt(sum of squared differences to the 8 neighbours). Border/NaN-adjacent = NaN."""
    d = dem.astype(np.float64)
    padded = np.pad(d, 1, constant_values=np.nan)
    h, w = d.shape
    acc = np.zeros_like(d)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            acc += (nb - d) ** 2
    out = np.sqrt(acc)
    return out.astype(np.float32)  # NaN where any neighbour (or itself) is NaN/outside


def local_std(arr: np.ndarray, window: int = ROUGHNESS_WINDOW_PX) -> np.ndarray:
    """NaN-aware standard deviation in a square `window` (odd, pixels)."""
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    a = arr.astype(np.float64)
    valid = np.isfinite(a)
    if not valid.any():
        return np.full(a.shape, np.nan, dtype=np.float32)
    shift = np.nanmean(a)  # improves conditioning of E[x^2] - E[x]^2
    x = np.where(valid, a - shift, 0.0)
    frac = ndimage.uniform_filter(valid.astype(np.float64), window, mode="constant")
    m1 = ndimage.uniform_filter(x, window, mode="constant")
    m2 = ndimage.uniform_filter(x * x, window, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        var = m2 / frac - (m1 / frac) ** 2
    out = np.sqrt(np.clip(var, 0.0, None))
    out[~valid | (frac <= 0)] = np.nan
    return out.astype(np.float32)


def palsar_to_db(arr: np.ndarray) -> tuple[np.ndarray, str]:
    """Convert PALSAR DN to gamma0 dB: 10*log10(DN^2) - 83. Returns (array, step description).

    If the data already looks like dB (non-positive maximum) it is returned unchanged.
    """
    a = arr.astype(np.float64)
    finite = a[np.isfinite(a)]
    if finite.size and finite.max() <= 0:
        return arr.astype(np.float32), "palsar: values already <= 0, assumed dB; no conversion"
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.where(a > 0, a, np.nan) ** 2) + PALSAR_CAL_DB
    return db.astype(np.float32), f"palsar: DN -> dB, 10*log10(DN^2){PALSAR_CAL_DB:+g}"


# --------------------------------------------------------------------------- stack


def _layer_array(spec: LayerSpec, raw_path: Path, grid: Grid) -> tuple[np.ndarray, list[str]]:
    steps = [
        f"reproject {raw_path.name} bands {list(spec.bands)} to common grid ({spec.resampling})"
    ]
    arr = reproject_to_grid(raw_path, grid, bands=spec.bands, resampling=spec.resampling)
    res = abs(grid[0].a)
    if spec.kind == "palsar_db":
        arr, step = palsar_to_db(arr)
        steps.append(step)
    elif spec.kind == "slope":
        arr = slope_degrees(arr[0], res, res)[None]
        steps.append("slope_degrees: np.gradient central differences on common-grid DEM")
    elif spec.kind == "roughness":
        arr = local_std(arr[0], ROUGHNESS_WINDOW_PX)[None]
        steps.append(
            f"local_std: NaN-aware std of elevation, {ROUGHNESS_WINDOW_PX}x{ROUGHNESS_WINDOW_PX} px"
        )
    return arr, steps


def _missing_reasons(rdir: Path) -> dict[str, str]:
    rep = rdir / "export_report.json"
    if not rep.exists():
        return {}
    try:
        results = json.loads(rep.read_text()).get("results", [])
    except json.JSONDecodeError:
        return {}
    return {
        r["product"]: r.get("error") or r.get("status", "") for r in results if r["status"] != "ok"
    }


def build_stack(
    aoi_id: str,
    *,
    settings: Settings | None = None,
    grid: Grid | None = None,
    layers: list[str] | None = None,
) -> dict:
    """Build COG layers + `stack.json` for an AOI from `data/raw/modern/{AOI}/`.

    Missing raw products are skipped and listed under `missing` (never synthesised).
    `grid` overrides `modern_grid(aoi_id)` (used by tests with tiny grids). Returns the manifest.
    """
    s = settings or load_settings()
    grid = grid or modern_grid(aoi_id, s)
    transform, width, height, crs = grid
    rdir, odir = raw_dir(aoi_id, s), stack_dir(aoi_id, s)
    odir.mkdir(parents=True, exist_ok=True)
    reasons = _missing_reasons(rdir)
    names = layers or list(LAYERS)
    unknown = set(names) - set(LAYERS)
    if unknown:
        raise KeyError(f"unknown layers: {sorted(unknown)}")

    entries: dict[str, dict] = {}
    missing: dict[str, str] = {}
    for name in names:
        spec = LAYERS[name]
        raw = rdir / f"{spec.product}.tif"
        if not raw.exists():
            missing[name] = reasons.get(spec.product, f"raw product {spec.product}.tif not found")
            continue
        arr, steps = _layer_array(spec, raw, grid)
        try:
            rp = read_prov(raw)
            source, source_id = rp["source"], rp["source_id"]
            acq = rp.get("acquisition_date")
        except FileNotFoundError:
            source, source_id, acq = "unknown (no raw provenance sidecar)", spec.product, None
        out = odir / f"{name}.tif"
        prof = {**grid_profile(grid), "nodata": float("nan")}
        write_cog(
            out,
            arr,
            prof,
            prov={
                "source": source,
                "source_id": source_id,
                "acquisition_date": acq,
                "steps": steps,
                "inputs": [str(raw)],
            },
        )
        entries[name] = {
            "file": out.name,
            "product": spec.product,
            "bands": [
                {"index": i + 1, "name": n, "meaning": m}
                for i, (n, m) in enumerate(zip(spec.band_names, spec.band_meaning, strict=True))
            ],
            "units": spec.units,
            "dtype": str(arr.dtype),
            "nodata": "NaN",
            "source_id": source_id,
            "resampling": spec.resampling,
            "native_resolution_m": PRODUCTS[spec.product].native_resolution_m,
            "derived_from": spec.derived_from,
        }

    left, bottom, right, top = grid_bounds(grid)
    manifest = {
        "aoi_id": aoi_id,
        "crs": crs,
        "resolution_m": abs(transform.a),
        "transform": list(transform)[:6],
        "width": width,
        "height": height,
        "bounds": [left, bottom, right, top],
        "created": datetime.now(UTC).isoformat(),
        "layers": entries,
        "missing": missing,
    }
    (odir / "stack.json").write_text(json.dumps(manifest, indent=2))
    return manifest
