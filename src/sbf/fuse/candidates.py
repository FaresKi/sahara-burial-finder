"""Candidate polygons, disturbance date brackets and ranking (SPEC Phase 6).

D8: nothing here logs geometry or coordinates; only counts.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from sbf.core.config import Settings, load_settings
from sbf.features.tiling import apply_exclusions, tile_geometries
from sbf.fuse.families import FAMILIES, add_family

log = logging.getLogger(__name__)

# ---- constants (single documented place) ----------------------------------------------------
TAU = 0.8  # anomaly "present" if mean score_norm in polygon >= TAU; family contributes if >= TAU
OPS_START, OPS_END = date(1961, 1, 1), date(1968, 12, 31)
PRE_1960_BEFORE = date(1960, 2, 1)  # earliest frame earlier than this and already anomalous
POST_2025_FROM = date(2025, 1, 1)
LABEL_MULTIPLIER = {
    "ops_1961_1968": 1.5,
    "other": 1.0,
    "pre_1960_natural_or_unknown": 0.7,
    "post_2025_likely_cleanup": 0.3,
}
SCORE_BANDS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001)]


def select_top_tiles(
    fused: pd.DataFrame,
    tile_ids_excluded: Iterable[str] = (),
    top_pct: float = 1.0,
) -> pd.DataFrame:
    """Top ``top_pct`` percent (at least one tile) of non-excluded tiles by S (ties by tile_id)."""
    f = fused[~fused.index.isin(set(tile_ids_excluded))]
    n = max(1, math.ceil(len(f) * top_pct / 100.0)) if len(f) else 0
    f = f.reset_index().sort_values(["S", "tile_id"], ascending=[False, True]).head(n)
    return f.set_index("tile_id")


def build_candidates(
    top: pd.DataFrame,
    fam: pd.DataFrame,
    overlap: float | None = None,
    settings: Settings | None = None,
    weights: Mapping[str, float] | None = None,
) -> tuple[gpd.GeoDataFrame, dict[str, list[str]]]:
    """Union the selected tiles into polygons.

    Returns (candidates GeoDataFrame in the AOI metric CRS, {cand_id: [tile_id, ...]}).
    Columns: cand_id, area_m2, mean_S, max_S, n_tiles, n_families, families (comma list of
    families whose mean score over the polygon's tiles is >= TAU), ``fam_<family>`` (mean family
    score over the tiles, NaN when absent), ``contrib_<family>`` (w_f * mean_f / sum over weighted
    present families; 0 for a present family with zero weight, NaN when absent or no ``weights``),
    geometry.
    """
    s = settings or load_settings()
    cols = ["cand_id", "area_m2", "mean_S", "max_S", "n_tiles", "n_families", "families"]
    cols += [f"fam_{f}" for f in FAMILIES] + [f"contrib_{f}" for f in FAMILIES]
    if top.empty:
        return gpd.GeoDataFrame({c: [] for c in cols}, geometry=[], crs="EPSG:32631"), {}
    tiles = tile_geometries(list(top.index), overlap, s)
    aoi = tiles["aoi_id"].iloc[0]
    polys = list(shapely.get_parts(shapely.union_all(tiles.geometry.values)))
    polys.sort(key=lambda p: (p.bounds[0], p.bounds[1]))
    pts = gpd.GeoSeries(tiles.geometry.representative_point(), crs=tiles.crs)
    rows, members = [], {}
    for i, poly in enumerate(polys):
        ids = tiles.loc[shapely.within(pts.values, poly), "tile_id"].tolist()
        cid = f"{aoi}-C{i + 1:04d}"
        members[cid] = ids
        sub = top.loc[ids]
        fmean = fam.reindex(ids).mean(axis=0, skipna=True)
        fm = {f: float(fmean.get(f, np.nan)) for f in FAMILIES}
        contrib = _contributions(fm, weights)
        rows.append(
            {
                "cand_id": cid,
                "area_m2": float(poly.area),
                "mean_S": float(sub["S"].mean()),
                "max_S": float(sub["S"].max()),
                "n_tiles": len(ids),
                "n_families": int(sub["n_families"].max()),
                "families": ",".join(f for f in FAMILIES if fmean.get(f, 0.0) >= TAU),
                **{f"fam_{f}": fm[f] for f in FAMILIES},
                **{f"contrib_{f}": contrib[f] for f in FAMILIES},
                "geometry": poly,
            }
        )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=tiles.crs)
    log.info("built %d candidate polygons from %d tiles", len(gdf), len(tiles))
    return gdf, members


def _contributions(
    fm: Mapping[str, float], weights: Mapping[str, float] | None
) -> dict[str, float]:
    """Share of each family in the weighted sum behind mean_S (renormalised over present)."""
    nan = float("nan")
    if weights is None:
        return dict.fromkeys(FAMILIES, nan)
    terms = {
        f: weights.get(f, 0.0) * fm[f]
        for f in FAMILIES
        if not math.isnan(fm[f]) and weights.get(f, 0.0) > 0
    }
    tot = sum(terms.values())
    out = dict.fromkeys(FAMILIES, nan)
    for f in FAMILIES:
        if not math.isnan(fm[f]):
            out[f] = terms[f] / tot if f in terms and tot > 0 else 0.0
    return out


def excluded_tile_ids(
    tile_ids: Iterable[str], exclusions: gpd.GeoDataFrame | None, settings: Settings | None = None
) -> set[str]:
    """Tile IDs intersecting the exclusions mask (empty set when no mask)."""
    ids = list(tile_ids)
    if not ids or exclusions is None or len(exclusions) == 0:
        return set()
    s = settings or load_settings()
    t = apply_exclusions(tile_geometries(ids, s.overlap, s), exclusions)
    return set(t.loc[t["excluded"], "tile_id"])


# ---- date bracket ---------------------------------------------------------------------------
def _to_date(v) -> date:
    return pd.to_datetime(v).date()


def _entity_dates(registry: pd.DataFrame) -> dict[str, date]:
    return {
        str(e).lower(): _to_date(d)
        for e, d in zip(registry["entity_id"], registry["acquisition_date"], strict=True)
        if pd.notna(d)
    }


def layer_date(layer: str, ent: Mapping[str, date]) -> date | None:
    """Acquisition date of a historical single-date layer ``<camera>_<entity>`` (longest match)."""
    lay = layer.lower()
    best = None
    for eid in ent:
        if (lay == eid or lay.endswith("_" + eid)) and (best is None or len(eid) > len(best)):
            best = eid
    return ent[best] if best else None


def date_bracket(
    candidate: Iterable[str],
    per_layer_scores: pd.DataFrame,
    registry: pd.DataFrame,
    modern_dates: Mapping[str, date | str] | None = None,
    tau: float = TAU,
) -> dict:
    """Disturbance date bracket and label for one candidate (given by its tile IDs).

    Historical single-date layers (hist_anomaly family) covering the candidate are "present" when
    their mean score_norm over the candidate's tiles (max over detectors) is >= tau.
    Returns dict: first_seen, last_absent_before, bracket, label, n_frames, n_present,
    modern_present (dates).
    """
    ids = set(candidate)
    df = add_family(per_layer_scores[per_layer_scores["tile_id"].isin(ids)])
    ent = _entity_dates(registry)
    frames: dict[date, float] = {}
    for layer, g in df[df["family"] == "hist_anomaly"].groupby("layer"):
        d = layer_date(layer, ent)
        if d is None:
            continue
        sc = float(g.groupby("detector")["score_norm"].mean().max())
        frames[d] = max(sc, frames.get(d, -1.0))
    dates = sorted(frames)
    present = [d for d in dates if frames[d] >= tau]
    first_seen = present[0] if present else None
    absent_before = [d for d in dates if frames[d] < tau and (first_seen is None or d < first_seen)]
    last_absent = max(absent_before) if absent_before else None

    mdates = {k.lower(): _to_date(v) for k, v in (modern_dates or {}).items()}
    modern_present, modern_absent = [], []
    for layer, g in df[df["family"] != "hist_anomaly"].groupby("layer"):
        d = mdates.get(layer.lower())
        if d is None or g["family"].iloc[0] == "prior":
            continue
        (modern_present if float(g["score_norm"].mean()) >= tau else modern_absent).append(d)

    label = _label(dates, first_seen, last_absent, modern_present)
    lo = str(last_absent) if last_absent else "<= earliest frame"
    return {
        "first_seen": first_seen,
        "last_absent_before": last_absent,
        "bracket": f"{lo} .. {first_seen}" if first_seen else "none in historical frames",
        "label": label,
        "n_frames": len(dates),
        "n_present": len(present),
        "modern_present": sorted(modern_present),
    }


def _label(dates, first_seen, last_absent, modern_present) -> str:
    if first_seen is not None:
        if first_seen == dates[0] and dates[0] < PRE_1960_BEFORE:
            return "pre_1960_natural_or_unknown"
        lo = last_absent or date.min
        if OPS_START <= first_seen <= OPS_END or (lo <= OPS_END and first_seen >= OPS_START):
            return "ops_1961_1968"
        return "other"
    if modern_present and all(d >= POST_2025_FROM for d in modern_present):
        return "post_2025_likely_cleanup"
    return "other"


def rank_candidates(cands: gpd.GeoDataFrame, brackets: Mapping[str, dict]) -> gpd.GeoDataFrame:
    """Add bracket columns, multiplier and ``final_rank_score = mean_S * multiplier``; sort + rank."""
    out = cands.copy()
    b = [brackets[c] for c in out["cand_id"]]
    out["label"] = [x["label"] for x in b]
    out["first_seen"] = [str(x["first_seen"] or "") for x in b]
    out["last_absent_before"] = [str(x["last_absent_before"] or "") for x in b]
    out["bracket"] = [x["bracket"] for x in b]
    out["n_present"] = [x.get("n_present", float("nan")) for x in b]
    out["n_frames"] = [x.get("n_frames", float("nan")) for x in b]
    out["multiplier"] = out["label"].map(LABEL_MULTIPLIER)
    out["final_rank_score"] = out["mean_S"] * out["multiplier"]
    out = out.sort_values(["final_rank_score", "cand_id"], ascending=[False, True])
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)
