"""Texture baseline detector: chips -> texture features -> background model -> score table."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from sbf.core.config import Settings, load_settings
from sbf.detect.background import fit_score
from sbf.detect.scores import _prepared, validate_scores, write_scores
from sbf.features.preprocess import preprocess_array, robust_range
from sbf.features.texture import texture_features
from sbf.features.tiling import make_grid, tile_windows

log = logging.getLogger(__name__)
DETECTOR = "texture"


def _raster_bounds(path) -> tuple[float, float, float, float]:
    with rasterio.open(path) as src:
        b = src.bounds
    return b.left, b.bottom, b.right, b.top


def run_texture_detector(
    aoi_id: str,
    layer: str,
    raster_path: str | Path,
    size_m: int = 64,
    method: str = "iforest",
    band: int = 1,
    preprocess: str = "none",
    min_valid_frac: float = 0.5,
    seed: int = 0,
    write: bool = True,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Score every tile of the raster for texture anomaly; return (and optionally write) the table.

    The raster must be north-up in the AOI CRS. Tiles with less than ``min_valid_frac`` valid
    pixels are dropped. ``preprocess`` (``none|clahe|lcn``, SPEC section 7) is applied to the whole
    band (held in memory) before chipping. The background model is fitted on all tiles of this
    raster (unsupervised; the detector never loads reference labels, D7).
    Logs only counts (D8).
    """
    s = settings or load_settings()
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(raster_path)
        if preprocess != "none":
            with rasterio.open(src_path) as src:
                arr = src.read(band, masked=True).astype("float32").filled(np.nan)
                profile = src.profile | {"count": 1, "dtype": "float32", "nodata": np.nan}
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
            profile.update(tiled=False, compress=None)
            src_path = Path(tmp) / "pre.tif"
            with rasterio.open(src_path, "w", **profile) as dst:
                dst.write(preprocess_array(arr, preprocess), 1)
            band = 1
        with rasterio.open(src_path) as src:
            stride = max(1, int(max(src.width, src.height) // 2048))
            sample = src.read(band, masked=True)[::stride, ::stride].astype("float32")
            lo, hi = robust_range(sample.filled(np.nan))
        grid = make_grid(aoi_id, size_m, s.overlap, s, bounds=_raster_bounds(src_path))
        feats = texture_features(tile_windows(grid, src_path, band, min_valid_frac), lo, hi)
    log.info(
        "texture %s/%s size=%d: %d tiles in grid, %d scored",
        aoi_id,
        layer,
        size_m,
        len(grid),
        len(feats),
    )
    if len(feats) < 3:
        raise ValueError(f"only {len(feats)} usable tiles; need at least 3")
    df = pd.DataFrame(
        {
            "tile_id": feats.index,
            "aoi_id": aoi_id,
            "size_m": int(size_m),
            "detector": DETECTOR,
            "layer": layer,
            "score": fit_score(feats.to_numpy(), method, seed=seed),
        }
    )
    df = validate_scores(_prepared(df))
    if write:
        write_scores(df, s)
    return df
