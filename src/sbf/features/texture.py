"""Texture statistics per chip: GLCM contrast/entropy, local variance, edge density."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import canny, graycomatrix, graycoprops

FEATURE_NAMES = ["glcm_contrast", "glcm_entropy", "local_var", "edge_density"]
GLCM_LEVELS = 16


def chip_texture(chip: np.ndarray, levels: int = GLCM_LEVELS) -> np.ndarray:
    """Four texture features (``FEATURE_NAMES`` order) of one 2-D chip scaled to ~[0, 1].

    NaN pixels are filled with the chip mean. An all-NaN chip returns NaNs.
    """
    a = np.asarray(chip, dtype="float64")
    finite = np.isfinite(a)
    if not finite.any():
        return np.full(len(FEATURE_NAMES), np.nan)
    a = np.where(finite, a, a[finite].mean())
    q = np.clip((a * levels).astype(int), 0, levels - 1).astype("uint8")
    glcm = graycomatrix(
        q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=levels, symmetric=True, normed=True
    )
    contrast = float(graycoprops(glcm, "contrast").mean())
    p = glcm[:, :, 0, :]
    entropy = float(-(p * np.log2(p, where=p > 0, out=np.zeros_like(p))).sum(axis=(0, 1)).mean())
    mean = ndimage.uniform_filter(a, 3)
    local_var = float(np.maximum(ndimage.uniform_filter(a * a, 3) - mean**2, 0).mean())
    edge = float(canny(a, sigma=1.0).mean())
    return np.array([contrast, entropy, local_var, edge])


def texture_features(
    chips: Iterable[tuple[str, np.ndarray]], lo: float | None = None, hi: float | None = None
) -> pd.DataFrame:
    """Feature matrix indexed by ``tile_id`` with columns ``FEATURE_NAMES``.

    If ``lo``/``hi`` are given, chips are scaled to [0, 1] with that single global range
    (use ``sbf.features.preprocess.robust_range`` over the layer so tiles stay comparable);
    otherwise chips must already be in [0, 1]. Rows with non-finite features are dropped.
    """
    ids, rows = [], []
    for tile_id, chip in chips:
        if lo is not None and hi is not None:
            chip = np.clip((chip - lo) / (hi - lo), 0.0, 1.0)
        ids.append(tile_id)
        rows.append(chip_texture(chip))
    df = pd.DataFrame(rows, index=pd.Index(ids, name="tile_id"), columns=FEATURE_NAMES)
    return df[np.isfinite(df).all(axis=1)]
