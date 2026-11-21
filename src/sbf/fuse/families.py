"""Score families, validation-derived weights and tile-level fusion (SPEC Phase 6).

``S = sum_f w_f * family_score_f`` where ``family_score_f`` is, per tile, the max (or mean)
``score_norm`` over the tables mapped to family ``f``. Tiles lacking a family are renormalised
over the families present (weight > 0); ``n_families`` records how many contributed.
Weights come from blind-test validation only (SPEC: no tuning on unseen data).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FAMILIES = ("hist_change", "hist_anomaly", "modern_optical", "SAR_L", "SAR_C", "DEM", "prior")
MODERN_PREFIXES = ("s2_", "s1_", "palsar_", "dem")
_EMBED_TEXTURE = {"texture", "embed_iforest", "embed_knn", "embed_maha"}
_CHANGE = {"change_embed", "change_diff"}


def family_of(detector: str, layer: str) -> str | None:
    """Map a (detector, layer) score table to a fusion family, or None (ignored, e.g. random)."""
    lay = layer.lower()
    if detector == "prior":
        return "prior"
    if detector in _CHANGE:
        return "hist_change" if lay.startswith("pair_") else None
    if detector == "dem_relief":
        return "DEM"
    if detector == "sar_z":
        if lay.startswith("palsar_"):
            return "SAR_L"
        if lay.startswith("s1_"):
            return "SAR_C"
        return None
    if detector in _EMBED_TEXTURE:
        if lay.startswith("s2_"):
            return "modern_optical"
        if lay.startswith(MODERN_PREFIXES) or lay.startswith("pair_"):
            return None
        return "hist_anomaly"
    return None


def add_family(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` with a ``family`` column (rows without a family are dropped)."""
    out = df.copy()
    out["family"] = [
        family_of(d, lay) for d, lay in zip(out["detector"], out["layer"], strict=True)
    ]
    return out[out["family"].notna()].reset_index(drop=True)


def weights_from_metrics(metrics: pd.DataFrame) -> dict[str, float]:
    """Family weights from ``harness.evaluate`` output (validation only).

    ``weight_f = max(0, best(AP - random_AP) within f) / sum over families``. Families that do
    not beat random (or have NaN metrics) get 0. All-zero when nothing beats random.
    """
    m = add_family(metrics)
    m = m[np.isfinite(m["ap"]) & np.isfinite(m["random_ap"])] if len(m) else m
    gain = dict.fromkeys(FAMILIES, 0.0)
    for fam, g in m.groupby("family"):
        gain[fam] = max(0.0, float((g["ap"] - g["random_ap"]).max()))
    total = sum(gain.values())
    return {f: (v / total if total > 0 else 0.0) for f, v in gain.items()}


def weights_table_md(weights: Mapping[str, float], source: str) -> str:
    """Markdown weights table (no coordinates; safe for outputs/public)."""
    lines = [
        "# Fusion weights",
        "",
        f"Source: {source}",
        "",
        "| family | weight |",
        "|---|---|",
    ]
    lines += [f"| {f} | {weights.get(f, 0.0):.3f} |" for f in FAMILIES]
    return "\n".join(lines) + "\n"


def family_scores(scores: pd.DataFrame, size_m: int, mode: str = "max") -> pd.DataFrame:
    """Wide frame (index tile_id, columns = families present) of per-tile family scores."""
    if mode not in ("max", "mean"):
        raise ValueError("mode must be 'max' or 'mean'")
    df = add_family(scores[scores["size_m"] == size_m])
    if df.empty:
        return pd.DataFrame(index=pd.Index([], name="tile_id"))
    wide = df.groupby(["tile_id", "family"])["score_norm"].agg(mode).unstack("family")
    return wide.reindex(columns=[f for f in FAMILIES if f in wide.columns])


def fuse(fam: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    """Fused score per tile: columns ``S`` and ``n_families`` (families present with w > 0).

    Tiles with no weighted family present are dropped.
    """
    cols = [c for c in fam.columns if weights.get(c, 0.0) > 0]
    if not cols:
        raise ValueError("no family with positive weight is present in the scores")
    w = np.array([weights[c] for c in cols], dtype=float)
    vals = fam[cols].to_numpy(dtype=float)
    present = ~np.isnan(vals)
    wsum = (present * w).sum(axis=1)
    num = (np.nan_to_num(vals) * w).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(wsum > 0, num / wsum, np.nan)
    out = pd.DataFrame({"S": s, "n_families": present.sum(axis=1)}, index=fam.index)
    return out[out["S"].notna()]
