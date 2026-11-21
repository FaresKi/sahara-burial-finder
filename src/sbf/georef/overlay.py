"""GZ overlay figure: georeferenced chip around each GZ1-GZ4 inside the frame.

Only public reference positions (IAEA ground zeros) are drawn; offsets are a human-measured
placeholder in the report, never computed here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sbf.core.crs import lonlat_to_xy
from sbf.georef.models import UTM_CRS
from sbf.georef.warp import WarpFile, WarpResult

GZ_IDS = ("GZ1", "GZ2", "GZ3", "GZ4")
CHIP_HALF_M = 500.0


def gz_in_frame(warped: WarpResult | WarpFile, gz) -> dict[str, tuple[float, float]]:
    """{feature_id: (col, row)} of GZ points whose pixel is inside the valid warped frame."""
    out: dict[str, tuple[float, float]] = {}
    inv = ~warped.transform
    h, w = warped.shape
    for _, f in gz.iterrows():
        if f["feature_id"] not in GZ_IDS or f.geometry is None or f.geometry.is_empty:
            continue
        c0 = f.geometry.centroid  # refined features are polygons; points are their own centroid
        x, y = lonlat_to_xy(c0.x, c0.y, UTM_CRS)
        c, r = inv @ (x, y)
        if 0 <= c < w and 0 <= r < h:
            v = warped.window(int(r), int(r) + 1, int(c), int(c) + 1)[0, 0]
            if np.isfinite(v) and v != warped.nodata:
                out[f["feature_id"]] = (c, r)
    return out


def gz_overlay(
    warped: WarpResult | WarpFile, gz, entity_id: str, out_dir: Path
) -> tuple[Path | None, list[str]]:
    """Save `{entity_id}_gz.png`; returns (path or None when no GZ is inside, ids inside)."""
    inside = gz_in_frame(warped, gz)
    if not inside:
        return None, []
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(inside)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2), squeeze=False)
    half = int(CHIP_HALF_M / warped.resolution_m)
    for ax, (fid, (c, r)) in zip(axes[0], inside.items(), strict=True):
        r0, r1 = max(int(r) - half, 0), min(int(r) + half, warped.shape[0])
        c0, c1 = max(int(c) - half, 0), min(int(c) + half, warped.shape[1])
        chip = warped.window(r0, r1, c0, c1).astype("float32")
        chip = np.where(chip == warped.nodata, np.nan, chip)
        ax.imshow(chip, cmap="gray", extent=(c0, c1, r1, r0))
        ax.plot(c, r, "+", color="#d62728", ms=18, mew=2)
        ax.set_title(f"{fid} (IAEA position)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{entity_id}: georeferenced chip, +/- {CHIP_HALF_M:.0f} m", fontsize=10)
    fig.tight_layout()
    png = out_dir / f"{entity_id}_gz.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png, list(inside)
