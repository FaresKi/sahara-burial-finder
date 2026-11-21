"""Raster output: always Cloud Optimized GeoTIFF, always with a provenance sidecar."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import rasterio
import rasterio.shutil

from sbf.core.provenance import write_prov


def write_cog(
    path: str | Path,
    array: np.ndarray,
    profile: dict,
    *,
    prov: dict,
) -> Path:
    """Write `array` ((h, w) or (bands, h, w)) as a COG plus `<path>.prov.json`.

    `profile` needs at least `crs` and `transform`; dtype/count/height/width are derived
    from the array. `prov` holds the kwargs of `write_prov` (source, source_id,
    acquisition_date, steps, optional inputs).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = array[np.newaxis] if array.ndim == 2 else array
    prof = {
        k: v for k, v in profile.items() if k not in {"driver", "blockxsize", "blockysize", "tiled"}
    }
    prof.update(
        driver="GTiff",
        count=arr.shape[0],
        height=arr.shape[1],
        width=arr.shape[2],
        dtype=str(arr.dtype),
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_tif = Path(tmp) / "src.tif"
        with rasterio.open(tmp_tif, "w", **prof) as dst:
            dst.write(arr)
        rasterio.shutil.copy(tmp_tif, path, driver="COG", compress="DEFLATE")
    write_prov(path, **prov)
    return path
