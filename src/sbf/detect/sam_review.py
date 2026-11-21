"""SAM2 review polygons for the top-K tiles of a detector (human review aid, NOT scoring).

Outputs are RESTRICTED: written only through ``restricted_path`` and encrypted when a key is
available. Only counts are logged (D8). The SAM2 checkpoint is never downloaded automatically:
fetching the weights is a human step.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import geopandas as gpd
import numpy as np
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import shape

from sbf.core.config import Settings, load_settings
from sbf.core.restricted import encrypt_if_key, restricted_path

log = logging.getLogger(__name__)
CHECKPOINT_ENV = "SBF_SAM2_CHECKPOINT"


class Segmenter(Protocol):
    def segment(self, chip: np.ndarray) -> list[np.ndarray]:
        """Boolean masks (h, w) for a 2-D float chip (NaN allowed)."""
        ...


class FakeSegmenter:
    """Threshold segmenter for tests: one mask of pixels above ``threshold``."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def segment(self, chip: np.ndarray) -> list[np.ndarray]:
        m = np.nan_to_num(chip, nan=-np.inf) > self.threshold
        return [m] if m.any() else []


class Sam2Segmenter:
    """SAM2 automatic mask generator. Checkpoint path from ``$SBF_SAM2_CHECKPOINT``."""

    def __init__(self, checkpoint: str | None = None, config: str = "sam2_hiera_s.yaml"):
        ckpt = checkpoint or os.environ.get(CHECKPOINT_ENV)
        if not ckpt or not Path(ckpt).is_file():
            raise RuntimeError(
                f"SAM2 checkpoint not found (set {CHECKPOINT_ENV} to a local file). Downloading "
                "the SAM2 weights is a human step; this code never downloads them."
            )
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except ImportError as e:
            raise RuntimeError("the `sam2` package is not installed (human step)") from e
        self._gen = SAM2AutomaticMaskGenerator(build_sam2(config, ckpt, device="cpu"))

    def segment(self, chip: np.ndarray) -> list[np.ndarray]:
        v = np.nan_to_num(chip, nan=float(np.nanmedian(chip)) if np.isfinite(chip).any() else 0)
        lo, hi = np.percentile(v, [2, 98])
        u8 = (np.clip((v - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)
        rgb = np.repeat(u8[..., None], 3, axis=2)
        return [m["segmentation"].astype(bool) for m in self._gen.generate(rgb)]


ChipReader = Callable[[tuple[float, float, float, float]], "np.ndarray | None"]


def sam_review_path(aoi: str, detector: str, layer: str, size_m: int, settings: Settings) -> Path:
    return restricted_path(
        "sam_review", f"{aoi}_{detector}_{layer}_{int(size_m)}m.gpkg", settings=settings
    )


def top_tiles(scores, k: int = 50):
    """Top-k rows by score (ties by tile_id)."""
    return scores.sort_values(["score", "tile_id"], ascending=[False, True]).head(k)


def run_sam_review(
    scores,
    segmenter: Segmenter,
    read_chip: ChipReader,
    aoi: str,
    detector: str,
    layer: str,
    size_m: int,
    top_k: int = 50,
    settings: Settings | None = None,
) -> Path | None:
    """Segment the chips of the top-K tiles; write polygons to the restricted gpkg.

    ``read_chip(bounds)`` returns a 2-D array covering the tile bounds (metric CRS) or None.
    Returns the output path (the encrypted copy if a key exists), or None with no polygons.
    """
    from sbf.features.tiling import tile_geometries

    s = settings or load_settings()
    sel = top_tiles(scores, top_k)
    tiles = tile_geometries(list(sel["tile_id"]), s.overlap, s).set_index("tile_id")
    crs = s.crs(aoi)
    rows, geoms = [], []
    for rank, (_, r) in enumerate(sel.iterrows(), start=1):
        b = tuple(tiles.geometry[r["tile_id"]].bounds)
        chip = read_chip(b)
        if chip is None or chip.ndim != 2:
            continue
        tr = from_bounds(*b, chip.shape[1], chip.shape[0])
        for mask in segmenter.segment(chip):
            for geom, val in features.shapes(mask.astype("uint8"), mask=mask, transform=tr):
                if val:
                    geoms.append(shape(geom))
                    rows.append({"tile_id": r["tile_id"], "rank": rank, "score": float(r["score"])})
    log.info("sam review: %d tiles, %d polygons", len(sel), len(geoms))
    if not geoms:
        return None
    path = sam_review_path(aoi, detector, layer, size_m, s)
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)
    gdf.to_file(path, driver="GPKG")
    return encrypt_if_key(path)
