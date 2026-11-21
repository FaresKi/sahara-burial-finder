"""Blind-test scoring harness (SPEC Phase 4, D7).

This is the ONLY module allowed to load reference labels. Detectors never see them.

Definitions (per aoi, detector, layer, tile size):
- Footprint of a reference feature: polygon geometry as is; point buffered to a circle of area
  ``expected_area_km2`` (point itself, i.e. its single tile, if the area is null). Features
  with null geometry are skipped. Tiles are positive if they intersect any footprint of the AOI.
- Excluded tiles (intersecting the exclusions mask) are removed BEFORE ranking.
- ``recall_at_K``: fraction of covered reference features (those with at least one scored,
  non-excluded positive tile) having a positive tile within the top K tiles. Feature-level,
  since one 1 km2 feature spans many tiles. ``tile_recall_at_K`` is the tile-level variant.
- ``ap``: tile-level average precision (positives vs all other tiles; label noise acknowledged).
- Random baseline: ``random_recall_at_K`` is the exact expectation (hypergeometric, per feature),
  ``random_tile_recall_at_K = K/N``, ``random_ap`` approximated by the positive prevalence.
- ``approximate_labels`` is True when any covered feature is not of precision ``iaea``.
Ties are broken by tile_id for determinism. Output holds counts/metrics only, no geometry (D8).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from sbf.core.aoi import load_reference_features
from sbf.core.config import Settings, load_settings
from sbf.core.crs import buffer_point_m, transform_geometry
from sbf.detect.exclusions import load_exclusions, review_status
from sbf.detect.scores import validate_scores
from sbf.features.tiling import apply_exclusions, tile_geometries

log = logging.getLogger(__name__)
DEFAULT_KS = (10, 50, 100)
_GROUP = ["aoi_id", "detector", "layer", "size_m"]


# ----------------------------------------------------------------------------- pure metrics
def average_precision(labels_ranked: Sequence[bool] | np.ndarray) -> float:
    """AP of a ranked 0/1 list (best first): mean of precision@rank over the positives."""
    y = np.asarray(labels_ranked, dtype=bool)
    if not y.any():
        return float("nan")
    hits = np.cumsum(y)
    return float((hits[y] / (np.flatnonzero(y) + 1)).mean())


def tile_recall_at_k(labels_ranked: Sequence[bool] | np.ndarray, k: int) -> float:
    """Positives in the top k / all positives."""
    y = np.asarray(labels_ranked, dtype=bool)
    return float(y[:k].sum() / y.sum()) if y.any() else float("nan")


def feature_recall_at_k(best_ranks: Sequence[int], k: int) -> float:
    """Fraction of features whose best (1-based) positive-tile rank is within k."""
    r = np.asarray(best_ranks)
    return float((r <= k).mean()) if len(r) else float("nan")


def random_feature_recall_at_k(n_tiles: int, tiles_per_feature: Sequence[int], k: int) -> float:
    """Expected feature-level recall@k for a random ranking of ``n_tiles`` tiles."""
    if not len(tiles_per_feature):
        return float("nan")
    k = min(k, n_tiles)
    return float(np.mean([hypergeom.sf(0, n_tiles, m, k) for m in tiles_per_feature]))


# ----------------------------------------------------------------------------- labels
def reference_footprints(aoi_id: str, reference: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    """Footprints (metric ``crs``) of the AOI's reference features; null geometries skipped.

    Columns: ``feature_id``, ``precision``, ``geometry``.
    """
    ref = reference[
        (reference["aoi_id"] == aoi_id) & reference.geometry.notna() & ~reference.geometry.is_empty
    ]
    rows = []
    for _, r in ref.iterrows():
        g = r.geometry
        if g.geom_type == "Point":
            area = r.get("expected_area_km2")
            if area is not None and np.isfinite(area) and area > 0:
                radius = float(np.sqrt(area * 1e6 / np.pi))
                geom = buffer_point_m(g.x, g.y, radius, crs)
            else:
                geom = transform_geometry(g, reference.crs.to_string(), crs)
        else:
            geom = transform_geometry(g, reference.crs.to_string(), crs)
        rows.append({"feature_id": r["feature_id"], "precision": r["precision"], "geometry": geom})
    return gpd.GeoDataFrame(rows, columns=["feature_id", "precision", "geometry"], crs=crs)


def _evaluate_group(g: pd.DataFrame, hits: pd.DataFrame, precision: dict, ks) -> dict:
    g = g.sort_values(["score", "tile_id"], ascending=[False, True], kind="stable")
    rank = pd.Series(np.arange(1, len(g) + 1), index=g["tile_id"].values)
    is_pos = g["tile_id"].isin(hits["tile_id"]).to_numpy()
    h = hits[hits["tile_id"].isin(rank.index)]
    if h.empty:
        per_feat = pd.DataFrame({"n": [], "best": []}, index=pd.Index([], name="feature_id"))
    else:
        per_feat = (
            h.assign(rank=rank.loc[h["tile_id"]].to_numpy())
            .groupby("feature_id")
            .agg(n=("tile_id", "nunique"), best=("rank", "min"))
        )
    out = {
        "n_tiles": len(g),
        "n_pos_tiles": int(is_pos.sum()),
        "n_features": len(per_feat),
        "approximate_labels": bool(any(precision.get(f) != "iaea" for f in per_feat.index)),
    }
    for k in ks:
        out[f"recall_at_{k}"] = feature_recall_at_k(per_feat["best"], k)
        out[f"random_recall_at_{k}"] = random_feature_recall_at_k(len(g), per_feat["n"], k)
        out[f"tile_recall_at_{k}"] = tile_recall_at_k(is_pos, k)
        out[f"random_tile_recall_at_{k}"] = min(1.0, k / len(g))
    out["ap"] = average_precision(is_pos)
    out["random_ap"] = float(is_pos.mean()) if is_pos.any() else float("nan")
    return out


def evaluate(
    scores: pd.DataFrame,
    reference: gpd.GeoDataFrame | None = None,
    exclusions: gpd.GeoDataFrame | None = None,
    settings: Settings | None = None,
    ks: Sequence[int] = DEFAULT_KS,
) -> pd.DataFrame:
    """Blind-test metrics per (aoi, detector, layer, size_m); one row each.

    ``reference`` defaults to ``load_reference_features()`` and ``exclusions`` to
    ``config/exclusions.geojson`` (if present). Pass ``exclusions=gpd.GeoDataFrame()`` to disable.
    Groups with no positive tile get NaN metrics. The ``exclusions_status`` column is
    ``none|draft|reviewed``: results under a draft mask are provisional.
    """
    s = settings or load_settings()
    scores = validate_scores(scores)
    reference = load_reference_features(s) if reference is None else reference
    exclusions = load_exclusions(s) if exclusions is None else exclusions
    status = review_status(exclusions if exclusions is not None and len(exclusions) else None)
    precision = dict(zip(reference["feature_id"], reference["precision"], strict=True))
    rows = []
    for (aoi, size), sub in scores.groupby(["aoi_id", "size_m"]):
        crs = s.crs(aoi)
        ids = sub["tile_id"].unique()
        tiles = apply_exclusions(tile_geometries(list(ids), s.overlap, s), exclusions)
        keep = set(tiles.loc[~tiles["excluded"], "tile_id"])
        foot = reference_footprints(aoi, reference, crs)
        joined = gpd.sjoin(
            tiles[~tiles["excluded"]][["tile_id", "geometry"]],
            foot,
            predicate="intersects",
            how="inner",
        )
        hits = joined[["tile_id", "feature_id"]]
        for (det, layer), g in sub.groupby(["detector", "layer"]):
            n_ex = int((~g["tile_id"].isin(keep)).sum())
            g = g[g["tile_id"].isin(keep)]
            row = {"aoi_id": aoi, "detector": det, "layer": layer, "size_m": int(size)}
            row["n_excluded"] = n_ex
            if len(g):
                row.update(_evaluate_group(g, hits, precision, ks))
            row["exclusions_status"] = status
            rows.append(row)
            log.info("evaluated %s/%s/%s/%s: %d tiles", aoi, det, layer, size, len(g))
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, ks: Sequence[int] = DEFAULT_KS, digits: int = 3) -> str:
    """Compact markdown table (counts and metrics only), with random expectations in brackets."""
    head = ["aoi", "detector", "layer", "size", "tiles", "pos", "feat"]
    head += [f"R@{k} (rand)" for k in ks] + ["AP (rand)", "approx", "mask"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]

    def f(x):
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"

    for _, r in df.iterrows():
        cells = [
            r["aoi_id"],
            r["detector"],
            r["layer"],
            str(r["size_m"]),
            str(r.get("n_tiles", 0)),
            str(r.get("n_pos_tiles", 0)),
            str(r.get("n_features", 0)),
        ]
        cells += [f"{f(r.get(f'recall_at_{k}'))} ({f(r.get(f'random_recall_at_{k}'))})" for k in ks]
        cells += [
            f"{f(r.get('ap'))} ({f(r.get('random_ap'))})",
            "yes" if r.get("approximate_labels") else "no",
            r["exclusions_status"],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
