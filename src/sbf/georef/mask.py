"""Step 3: stable-terrain mask (SPEC section 7: never use dunes for GCPs).

Single documented rule, `stable_terrain_mask`:

    dune_like = (S2 local std over ~200 m is inside DUNE_TEXTURE_BAND)
                AND (DEM slope >= DUNE_SLOPE_MIN_DEG, or DEM roughness >= DUNE_ROUGHNESS_MIN_M
                     when no slope layer is supplied)
    stable    = finite(S2) AND NOT dilate(dune_like, DUNE_BUFFER_PX)

Rationale: dune fields have moderate, regular S2 texture (ripples/slip faces) together with
real relief; flat sand sheets have low texture, bedrock/regs/wadis/roads have high or very low
texture with little regular relief at 30 m. Thresholds are heuristics for S2 surface
reflectance (0-1) and Copernicus GLO-30 slope and MUST be calibrated on the first real chips
(see notes/PHASE_3_NOTES.md). Flat sand sheets stay "stable" but carry few features anyway.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from sbf.ingest.modern.process import local_std


@dataclass(frozen=True)
class MaskParams:
    texture_window_px: int = 21  # 210 m at 10 m
    dune_texture_band: tuple[float, float] = (0.008, 0.045)  # reflectance std
    dune_slope_min_deg: float = 3.0
    dune_roughness_min_m: float = 1.5
    dune_buffer_px: int = 2


DEFAULT_PARAMS = MaskParams()
MIN_STABLE_FRACTION = 0.05  # below this the mask is discarded by the pipeline (warning)


def stable_terrain_mask(
    stack: Mapping[str, np.ndarray], params: MaskParams = DEFAULT_PARAMS
) -> np.ndarray:
    """Boolean mask, True = usable for GCP matching. Keys: `s2` (required), `slope`, `roughness`."""
    s2 = np.asarray(stack["s2"], dtype="float32")
    texture = local_std(s2, params.texture_window_px)
    lo, hi = params.dune_texture_band
    with np.errstate(invalid="ignore"):
        textured = (texture >= lo) & (texture <= hi)
        if stack.get("slope") is not None:
            relief = np.asarray(stack["slope"]) >= params.dune_slope_min_deg
        elif stack.get("roughness") is not None:
            relief = np.asarray(stack["roughness"]) >= params.dune_roughness_min_m
        else:
            relief = np.zeros(s2.shape, dtype=bool)  # no DEM: texture alone is not enough
    dune = textured & relief
    if params.dune_buffer_px > 0 and dune.any():
        dune = ndimage.binary_dilation(dune, iterations=params.dune_buffer_px)
    return np.isfinite(s2) & ~dune
