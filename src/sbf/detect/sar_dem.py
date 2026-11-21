"""SAR and DEM anomaly detectors (SPEC Phase 4, detector 4): ``sar_z`` and ``dem_relief``.

``sar_z``
    Pixel-level local z-score of backscatter against a ring (annulus) background that excludes
    the centre: ``z = (x - mu_ring) / max(sd_ring, min_sd)``. ``mu/sd`` are mean/std, or with
    ``robust=True`` median / 1.4826*MAD. Values are expected in **dB**. The modern stack layers
    (``palsar_hv``, ``palsar_hh``, ``s1_vv_mean``, ``s1_vh_mean``, ``s1_*_std``) are already dB
    (``stack.json`` units ``dB``); pass ``units="linear"`` (or a manifest entry with units
    ``linear``) to convert with ``10*log10(x)`` (non-positive -> NaN).
``dem_relief``
    High-pass micro-relief residual ``DEM - smooth(DEM)`` (Gaussian with ``sigma_m``, or a disk
    median of radius ``sigma_m``), in metres. Also ``dem_roughness`` as a local-z variant.

Tile score = aggregate (default 95th percentile) of the tail-transformed per-pixel values of the
pixels inside each tile (`sbf.features.tiling.tile_windows`).

Caveat (GLO-30): the Copernicus DEM is ~30 m native, resampled to 10 m; relief with wavelength
below ~30 m is not resolvable, whatever the pixel size. SAR layers have a coarser native
resolution too (PALSAR 25 m).

Whole-AOI rasters are processed in blocks with a halo equal to the filter radius (identical to
whole-array processing); the per-pixel result goes to a temporary GeoTIFF and tiles are read
back from it. Nodata/NaN pixels are excluded from backgrounds (normalised convolution) and a
pixel is only scored if enough of its background is valid. Labels are never read (D7); only
counts and tile IDs are logged (D8).
"""

from __future__ import annotations

import json
import logging
import tempfile
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from numpy.lib.stride_tricks import sliding_window_view
from rasterio.windows import Window
from scipy import ndimage

from sbf.core.config import Settings, load_settings
from sbf.detect.scores import _prepared, validate_scores, write_scores
from sbf.features.tiling import make_grid, tile_windows

log = logging.getLogger(__name__)

SAR_LAYERS = (
    "palsar_hv",
    "palsar_hh",
    "palsar_hv_multi",
    "palsar_hh_multi",
    "s1_vv_mean",
    "s1_vh_mean",
    "s1_vv_std",
    "s1_vh_std",
)
DEM_LAYERS = ("dem", "dem_roughness")
GLO30_NOTE = (
    "GLO-30 DEM is ~30 m native (resampled to 10 m): micro-relief with wavelength "
    "below ~30 m is not resolvable"
)
_MAX_STRIP_ELEMS = 6_000_000


# --------------------------------------------------------------------------- helpers


def db_from_linear(arr: np.ndarray) -> np.ndarray:
    """``10*log10(x)`` for positive finite ``x``; everything else becomes NaN."""
    a = np.asarray(arr, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            np.isfinite(a) & (a > 0), 10.0 * np.log10(np.where(a > 0, a, 1.0)), np.nan
        ).astype("float32")


def annulus_footprint(inner_px: float, outer_px: float) -> np.ndarray:
    """Boolean ring footprint: ``inner_px < distance <= outer_px`` (pixel-centre distances)."""
    if outer_px <= inner_px or outer_px < 1:
        raise ValueError("need outer radius > inner radius and outer >= 1 pixel")
    r = int(np.ceil(outer_px))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    d = np.hypot(yy, xx)
    return (d > inner_px) & (d <= outer_px)


def disk_footprint(radius_px: float) -> np.ndarray:
    r = int(np.ceil(radius_px))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return np.hypot(yy, xx) <= radius_px


def _robust_window_stats(
    arr: np.ndarray, fp: np.ndarray, min_frac: float, want_mad: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel nanmedian, 1.4826*MAD and valid count over footprint ``fp`` (strips of rows)."""
    k = fp.shape[0]
    r = k // 2
    h, w = arr.shape
    padded = np.pad(arr.astype("float32"), r, constant_values=np.nan)
    windows = sliding_window_view(padded, (k, k))
    n_fp = int(fp.sum())
    med = np.full((h, w), np.nan, dtype="float32")
    mad = np.full((h, w), np.nan, dtype="float32")
    cnt = np.zeros((h, w), dtype="int32")
    rows = max(1, _MAX_STRIP_ELEMS // max(1, w * n_fp))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i0 in range(0, h, rows):
            win = windows[i0 : i0 + rows][..., fp]  # (rows, w, n_fp) copy
            c = np.isfinite(win).sum(-1)
            m = np.nanmedian(win, axis=-1)
            ok = c >= max(1, min_frac * n_fp)
            sl = slice(i0, i0 + win.shape[0])
            cnt[sl] = c
            med[sl] = np.where(ok, m, np.nan)
            if want_mad:
                s = 1.4826 * np.nanmedian(np.abs(win - m[..., None]), axis=-1)
                mad[sl] = np.where(ok, s, np.nan)
    return med, mad, cnt


def local_z(
    arr: np.ndarray,
    inner_px: float,
    outer_px: float,
    robust: bool = False,
    min_sd: float = 1e-6,
    min_valid_frac: float = 0.3,
) -> np.ndarray:
    """Local z-score of every pixel against its ring background (centre excluded).

    NaN in -> NaN out; a pixel is NaN also when fewer than ``min_valid_frac`` of its ring pixels
    are valid. ``min_sd`` floors the background spread so flat backgrounds do not explode.
    """
    x = np.asarray(arr, dtype="float64")
    fp = annulus_footprint(inner_px, outer_px)
    if robust:
        med, sd, _ = _robust_window_stats(x, fp, min_valid_frac, want_mad=True)
        mu = med.astype("float64")
        sd = sd.astype("float64")
    else:
        kern = fp.astype("float64")
        valid = np.isfinite(x)
        x0 = np.where(valid, x, 0.0)
        n = ndimage.correlate(valid.astype("float64"), kern, mode="constant", cval=0.0)
        s1 = ndimage.correlate(x0, kern, mode="constant", cval=0.0)
        s2 = ndimage.correlate(x0 * x0, kern, mode="constant", cval=0.0)
        ok = n >= max(1.0, min_valid_frac * fp.sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(ok, s1 / n, np.nan)
            var = np.where(ok, s2 / n - mu * mu, np.nan)
        sd = np.sqrt(np.maximum(var, 0.0))
    z = (x - mu) / np.maximum(sd, min_sd)
    return z.astype("float32")


def smooth_nan(
    arr: np.ndarray, sigma_px: float, method: str = "gaussian", min_valid_frac: float = 0.5
) -> np.ndarray:
    """NaN-aware smoothing: normalised Gaussian convolution, or disk median of radius sigma_px."""
    x = np.asarray(arr, dtype="float64")
    if method == "gaussian":
        valid = np.isfinite(x)
        num = ndimage.gaussian_filter(
            np.where(valid, x, 0.0), sigma_px, mode="constant", truncate=4.0
        )
        den = ndimage.gaussian_filter(
            valid.astype("float64"), sigma_px, mode="constant", truncate=4.0
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den >= min_valid_frac, num / den, np.nan)
        return out.astype("float32")
    if method == "median":
        med, _, _ = _robust_window_stats(x, disk_footprint(sigma_px), min_valid_frac, False)
        return med
    raise ValueError(f"unknown smoothing method {method!r} (gaussian | median)")


def relief_residual(
    arr: np.ndarray, sigma_px: float, method: str = "gaussian", min_valid_frac: float = 0.5
) -> np.ndarray:
    """High-pass micro-relief ``arr - smooth(arr)``; NaN where the centre or background is invalid."""
    return (
        np.asarray(arr, dtype="float32") - smooth_nan(arr, sigma_px, method, min_valid_frac)
    ).astype("float32")


def _halo_for_gaussian(sigma_px: float) -> int:
    return int(np.ceil(4.0 * sigma_px)) + 1


def _blocks(
    h: int, w: int, block: int, halo: int
) -> Iterator[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    """Yield core (r0, r1, c0, c1) and halo-padded (R0, R1, C0, C1) boxes covering the array."""
    for r0 in range(0, h, block):
        for c0 in range(0, w, block):
            r1, c1 = min(h, r0 + block), min(w, c0 + block)
            yield (
                (r0, r1, c0, c1),
                (max(0, r0 - halo), min(h, r1 + halo), max(0, c0 - halo), min(w, c1 + halo)),
            )


def blockwise(
    arr: np.ndarray, fn: Callable[[np.ndarray], np.ndarray], halo: int, block: int = 512
) -> np.ndarray:
    """Apply a local filter ``fn`` block by block with a ``halo``; equals ``fn(arr)``."""
    h, w = arr.shape
    out = np.full((h, w), np.nan, dtype="float32")
    for (r0, r1, c0, c1), (R0, R1, C0, C1) in _blocks(h, w, block, halo):
        res = fn(arr[R0:R1, C0:C1])
        out[r0:r1, c0:c1] = res[r0 - R0 : r1 - R0, c0 - C0 : c1 - C0]
    return out


def aggregator(agg: str) -> Callable[[np.ndarray], float]:
    """``p<N>`` percentile (e.g. ``p95``), ``max``, ``mean`` or ``median`` of a 1-D array."""
    a = agg.lower()
    if a.startswith("p") and a[1:].replace(".", "", 1).isdigit():
        q = float(a[1:])
        if not 0 <= q <= 100:
            raise ValueError("percentile must be in [0, 100]")
        return lambda v: float(np.percentile(v, q))
    simple = {"max": np.max, "mean": np.mean, "median": np.median}
    if a in simple:
        f = simple[a]
        return lambda v: float(f(v))
    raise ValueError(f"unknown aggregate {agg!r} (pNN | max | mean | median)")


def _tail(chip: np.ndarray, tail: str) -> np.ndarray:
    if tail == "both":
        return np.abs(chip)
    if tail == "high":
        return chip
    if tail == "low":
        return -chip
    raise ValueError("tail must be both | high | low")


# --------------------------------------------------------------------------- raster pipeline


def _manifest_entry(aoi_id: str, layer: str, settings: Settings) -> tuple[Path | None, dict]:
    d = settings.data_processed / "modern" / aoi_id
    mp = d / "stack.json"
    if not mp.exists():
        return None, {}
    m = json.loads(mp.read_text())
    e = m.get("layers", {}).get(layer)
    if e is None:
        return None, {}
    return d / e["file"], e


def resolve_layer(
    aoi_id: str, layer: str, raster: str | Path | None, settings: Settings
) -> tuple[Path, dict]:
    """Raster path (explicit, else from ``data/processed/modern/{AOI}/stack.json``) + entry."""
    path, entry = _manifest_entry(aoi_id, layer, settings)
    if raster is not None:
        return Path(raster), entry
    if path is None:
        raise FileNotFoundError(
            f"no --raster given and layer {layer!r} is not in the stack manifest for {aoi_id}"
        )
    return path, entry


def _filter_to_raster(
    src_path: Path,
    out_path: Path,
    fn: Callable[[np.ndarray], np.ndarray],
    halo: int,
    block: int,
    band: int,
    preprocess: Callable[[np.ndarray], np.ndarray] | None,
) -> None:
    """Run ``fn`` on halo-padded windows of ``src_path`` and write the cores to ``out_path``."""
    with rasterio.open(src_path) as src:
        prof = src.profile | {
            "count": 1,
            "dtype": "float32",
            "nodata": np.nan,
            "tiled": False,
            "compress": "deflate",
        }
        for k in ("blockxsize", "blockysize", "photometric", "interleave"):
            prof.pop(k, None)
        with rasterio.open(out_path, "w", **prof) as dst:
            for (r0, r1, c0, c1), (R0, R1, C0, C1) in _blocks(src.height, src.width, block, halo):
                a = (
                    src.read(band, window=Window(C0, R0, C1 - C0, R1 - R0), masked=True)
                    .astype("float32")
                    .filled(np.nan)
                )
                if preprocess is not None:
                    a = preprocess(a)
                res = fn(a)[r0 - R0 : r1 - R0, c0 - C0 : c1 - C0]
                dst.write(res.astype("float32"), 1, window=Window(c0, r0, c1 - c0, r1 - r0))


def _score_raster(
    path: Path,
    aoi_id: str,
    size_m: int,
    detector: str,
    layer: str,
    agg: str,
    tail: str,
    min_valid_frac: float,
    settings: Settings,
) -> pd.DataFrame:
    with rasterio.open(path) as src:
        b = src.bounds
    grid = make_grid(
        aoi_id, size_m, settings.overlap, settings, bounds=(b.left, b.bottom, b.right, b.top)
    )
    f = aggregator(agg)
    ids, scores = [], []
    for tid, chip in tile_windows(grid, path, 1, min_valid_frac):
        v = _tail(chip[np.isfinite(chip)], tail)
        ids.append(tid)
        scores.append(f(v))
    log.info(
        "%s %s/%s size=%d: %d tiles in grid, %d scored",
        detector,
        aoi_id,
        layer,
        size_m,
        len(grid),
        len(ids),
    )
    if len(ids) < 3:
        raise ValueError(f"only {len(ids)} usable tiles; need at least 3")
    df = pd.DataFrame(
        {
            "tile_id": ids,
            "aoi_id": aoi_id,
            "size_m": int(size_m),
            "detector": detector,
            "layer": layer,
            "score": scores,
        }
    )
    return validate_scores(_prepared(df))


def _pixel_size(path: Path) -> float:
    with rasterio.open(path) as src:
        t = src.transform
    if t.b != 0 or t.d != 0 or t.e >= 0 or not np.isclose(t.a, -t.e):
        raise ValueError("raster must be north-up with square pixels")
    return float(t.a)


def sar_z_raster(
    raster_path: str | Path,
    out_path: str | Path,
    inner_m: float = 30.0,
    outer_m: float = 120.0,
    robust: bool = False,
    units: str = "db",
    band: int = 1,
    min_sd: float = 0.25,
    min_valid_frac: float = 0.3,
    block: int = 1024,
) -> None:
    """Write the per-pixel local-z raster of a backscatter layer (blockwise, with halo)."""
    path = Path(raster_path)
    px = _pixel_size(path)
    inner_px, outer_px = inner_m / px, outer_m / px
    annulus_footprint(inner_px, outer_px)  # validate
    pre = db_from_linear if units == "linear" else None
    if units not in ("db", "linear"):
        raise ValueError("units must be db | linear")
    _filter_to_raster(
        path,
        Path(out_path),
        lambda a: local_z(a, inner_px, outer_px, robust, min_sd, min_valid_frac),
        int(np.ceil(outer_px)),
        block,
        band,
        pre,
    )


def run_sar_z(
    aoi_id: str,
    layer: str,
    raster_path: str | Path | None = None,
    size_m: int = 128,
    inner_m: float = 30.0,
    outer_m: float = 120.0,
    robust: bool = False,
    agg: str = "p95",
    tail: str = "both",
    units: str | None = None,
    band: int = 1,
    min_sd: float = 0.25,
    min_valid_frac: float = 0.5,
    block: int = 1024,
    write: bool = True,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """``sar_z`` score table for one backscatter layer (tile = ``agg`` of local z per tile).

    ``units`` defaults to the manifest entry's units (``dB`` -> dB) else dB. ``tail`` selects
    ``both`` (|z|), ``high`` (bright anomalies) or ``low`` (dark). Logs counts only (D8).
    """
    s = settings or load_settings()
    path, entry = resolve_layer(aoi_id, layer, raster_path, s)
    u = (units or str(entry.get("units", "dB"))).lower()
    u = "linear" if u.startswith("lin") else "db"
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "z.tif"
        sar_z_raster(path, zpath, inner_m, outer_m, robust, u, band, min_sd, 0.3, block)
        df = _score_raster(zpath, aoi_id, size_m, "sar_z", layer, agg, tail, min_valid_frac, s)
    if write:
        write_scores(df, s)
    return df


def dem_relief_raster(
    raster_path: str | Path,
    out_path: str | Path,
    sigma_m: float = 50.0,
    method: str = "gaussian",
    band: int = 1,
    min_valid_frac: float = 0.5,
    block: int = 1024,
    roughness: bool = False,
    inner_m: float = 30.0,
    outer_m: float = 120.0,
    min_sd: float = 0.01,
) -> None:
    """Write the per-pixel micro-relief residual (m), or with ``roughness`` its local robust z."""
    path = Path(raster_path)
    px = _pixel_size(path)
    if roughness:
        inner_px, outer_px = inner_m / px, outer_m / px
        _filter_to_raster(
            path,
            Path(out_path),
            lambda a: local_z(a, inner_px, outer_px, True, min_sd, 0.3),
            int(np.ceil(outer_px)),
            block,
            band,
            None,
        )
        return
    sigma_px = sigma_m / px
    halo = _halo_for_gaussian(sigma_px) if method == "gaussian" else int(np.ceil(sigma_px)) + 1
    _filter_to_raster(
        path,
        Path(out_path),
        lambda a: relief_residual(a, sigma_px, method, min_valid_frac),
        halo,
        block,
        band,
        None,
    )


def run_dem_relief(
    aoi_id: str,
    raster_path: str | Path | None = None,
    layer: str = "dem",
    size_m: int = 128,
    sigma_m: float = 50.0,
    method: str = "gaussian",
    agg: str = "p95",
    tail: str = "both",
    band: int = 1,
    inner_m: float = 30.0,
    outer_m: float = 120.0,
    min_valid_frac: float = 0.5,
    block: int = 1024,
    write: bool = True,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """``dem_relief`` score table. ``layer="dem"``: high-pass residual in metres;
    ``layer="dem_roughness"``: robust local-z anomaly of the roughness layer (ring background).
    """
    if layer not in DEM_LAYERS:
        raise ValueError(f"layer must be one of {DEM_LAYERS}")
    s = settings or load_settings()
    path, _ = resolve_layer(aoi_id, layer, raster_path, s)
    rough = layer == "dem_roughness"
    if sigma_m < 30 and not rough:
        log.warning("sigma_m=%.0f m is below the ~30 m GLO-30 native resolution", sigma_m)
    log.warning(GLO30_NOTE)
    with tempfile.TemporaryDirectory() as tmp:
        rpath = Path(tmp) / "relief.tif"
        dem_relief_raster(path, rpath, sigma_m, method, band, 0.5, block, rough, inner_m, outer_m)
        df = _score_raster(
            rpath,
            aoi_id,
            size_m,
            "dem_relief",
            layer,
            agg,
            "high" if rough else tail,
            min_valid_frac,
            s,
        )
    if write:
        write_scores(df, s)
    return df
