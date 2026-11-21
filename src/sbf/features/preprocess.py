"""Contrast preprocessing for saturated desert scans (SPEC section 7).

Bright sand clips in film scans, so local contrast normalisation is applied before texture
features or embeddings. All functions take 2-D float arrays (NaN = nodata, preserved) and return
float32 arrays of the same shape.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import exposure

METHODS = ("none", "clahe", "lcn")


def robust_range(array: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    """Percentile (lo, hi) of the finite values; guards against a degenerate range."""
    v = array[np.isfinite(array)]
    if v.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(v, [lo, hi])
    return (float(a), float(b)) if b > a else (float(a), float(a) + 1.0)


def to_unit(array: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linearly map [lo, hi] to [0, 1], clipped. NaN preserved."""
    return np.clip((array.astype("float32") - lo) / (hi - lo), 0.0, 1.0)


def clahe(array: np.ndarray, clip_limit: float = 0.02, tiles: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation; output in [0, 1].

    Input is first min-max scaled by its 1-99 percentile (so clipped/saturated sand does not
    dominate). ``tiles`` is the approximate number of contextual regions per axis.
    """
    a = np.asarray(array, dtype="float32")
    mask = np.isfinite(a)
    if not mask.any():
        return a.copy()
    lo, hi = robust_range(a, 1, 99)
    u = to_unit(np.where(mask, a, np.nanmedian(a)), lo, hi)
    kernel = max(2, max(a.shape) // tiles)
    out = exposure.equalize_adapthist(u, kernel_size=kernel, clip_limit=clip_limit).astype(
        "float32"
    )
    out[~mask] = np.nan
    return out


def local_contrast_normalise(array: np.ndarray, sigma: float = 16.0) -> np.ndarray:
    """(x - local mean) / local std with Gaussian windows (NaN-aware). Zero-mean output."""
    a = np.asarray(array, dtype="float64")
    mask = np.isfinite(a)
    if not mask.any():
        return a.astype("float32")
    w = ndimage.gaussian_filter(mask.astype("float64"), sigma)
    w = np.where(w > 1e-6, w, 1.0)
    x = np.where(mask, a, 0.0)
    mu = ndimage.gaussian_filter(x, sigma) / w
    var = ndimage.gaussian_filter(x * x, sigma) / w - mu**2
    out = (x - mu) / np.sqrt(np.maximum(var, 0) + 1e-6)
    out[~mask] = np.nan
    return out.astype("float32")


def preprocess_array(array: np.ndarray, method: str = "clahe") -> np.ndarray:
    """Dispatch on ``method`` in ``METHODS`` (``none`` returns a float32 copy)."""
    if method == "none":
        return np.asarray(array, dtype="float32").copy()
    if method == "clahe":
        return clahe(array)
    if method == "lcn":
        return local_contrast_normalise(array)
    raise ValueError(f"unknown preprocess method {method!r}; choose from {METHODS}")
