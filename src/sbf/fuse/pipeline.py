"""Phase 6 orchestration: scores -> fused tiles -> candidates -> restricted outputs + public stats.

Restricted files are written only through ``sbf.core.restricted``. The public summary holds
counts only (no coordinates, no per-candidate areas). D8: logs hold counts only.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd

from sbf.core.config import Settings, load_settings
from sbf.core.restricted import encrypt_if_key, restricted_path
from sbf.fuse.candidates import (
    LABEL_MULTIPLIER,
    SCORE_BANDS,
    TAU,
    build_candidates,
    date_bracket,
    excluded_tile_ids,
    rank_candidates,
    select_top_tiles,
)
from sbf.fuse.families import (
    FAMILIES,
    family_scores,
    fuse,
    weights_from_metrics,
    weights_table_md,
)

log = logging.getLogger(__name__)
RANKING_COLUMNS = [
    "rank", "cand_id", "label", "bracket", "first_seen", "last_absent_before", "mean_S",
    "max_S", "multiplier", "final_rank_score", "area_m2", "n_tiles", "n_families", "families",
    "n_present", "n_frames",
    *[f"fam_{f}" for f in FAMILIES], *[f"contrib_{f}" for f in FAMILIES],
]  # fmt: skip


@dataclass
class FusionResult:
    aoi: str
    size_m: int
    weights: dict[str, float]
    weights_source: str
    ranked: gpd.GeoDataFrame
    n_tiles_fused: int
    paths: list = field(default_factory=list)


def default_size(scores: pd.DataFrame) -> int:
    sizes = set(scores["size_m"].unique())
    if not sizes:
        raise ValueError("no score tables found")
    return 64 if 64 in sizes else 128 if 128 in sizes else min(sizes)


def run_fusion(
    aoi: str,
    scores: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    weights: Mapping[str, float] | None = None,
    metrics: pd.DataFrame | None = None,
    weights_source: str | None = None,
    size_m: int | None = None,
    top_pct: float = 1.0,
    mode: str = "max",
    tau: float = TAU,
    modern_dates: Mapping[str, str] | None = None,
    exclusions: gpd.GeoDataFrame | None = None,
    settings: Settings | None = None,
) -> FusionResult:
    """Compute and rank candidates (no files written; see `write_outputs`)."""
    s = settings or load_settings()
    size = size_m or default_size(scores)
    if weights is None:
        if metrics is None:
            raise ValueError("need either manual weights or validation metrics")
        weights = weights_from_metrics(metrics)
        weights_source = weights_source or "validation metrics (AP above random)"
    else:
        weights_source = weights_source or "manual override file"
    fam = family_scores(scores, size, mode)
    fused = fuse(fam, weights)
    excl = excluded_tile_ids(fused.index, exclusions, s)
    top = select_top_tiles(fused, excl, top_pct)
    cands, members = build_candidates(top, fam, s.overlap, s, weights=weights)
    per_layer = scores[scores["size_m"] == size]
    brackets = {
        cid: date_bracket(ids, per_layer, registry, modern_dates, tau)
        for cid, ids in members.items()
    }
    ranked = rank_candidates(cands, brackets)
    log.info("fusion %s/%d: %d tiles fused, %d candidates", aoi, size, len(fused), len(ranked))
    return FusionResult(aoi, size, dict(weights), weights_source, ranked, len(fused))


def _band_label(lo: float, hi: float) -> str:
    return f"{lo:.1f}-{min(hi, 1.0):.1f}"


def public_summary_md(res: FusionResult, params: Mapping) -> str:
    """Aggregate-only summary: counts per mean-S band and per date-bracket label."""
    r = res.ranked
    lines = [
        f"# Fusion summary: {res.aoi}",
        "",
        "Candidate areas for verification (requires ground verification by competent authorities).",
        "Aggregate counts only; locations are restricted.",
        "",
        f"- tile size: {res.size_m} m; tiles fused: {res.n_tiles_fused}",
        f"- candidates: {len(r)}; selection: top {params.get('top_pct')}% of non-excluded tiles",
        f"- weights source: {res.weights_source}",
        "",
        "## Candidates per mean-score band",
        "",
        "| band | count |",
        "|---|---|",
    ]
    for lo, hi in SCORE_BANDS:
        n = int(((r["mean_S"] >= lo) & (r["mean_S"] < hi)).sum()) if len(r) else 0
        lines.append(f"| {_band_label(lo, hi)} | {n} |")
    lines += ["", "## Candidates per date-bracket label", "", "| label | count |", "|---|---|"]
    for lab in LABEL_MULTIPLIER:
        n = int((r["label"] == lab).sum()) if len(r) else 0
        lines.append(f"| {lab} | {n} |")
    return "\n".join(lines) + "\n"


def _merge_summary(path, aoi: str, section: str) -> str:
    """One public fusion_summary.md for all AOIs: replace this AOI's section, keep the others."""
    head = "# Fusion summary: "
    parts: dict[str, str] = {}
    if path.exists():
        for chunk in path.read_text().split("\n" + head):
            chunk = chunk if chunk.startswith(head) else head + chunk
            if chunk.strip() != head.strip():
                parts[chunk[len(head) :].split("\n", 1)[0].strip()] = chunk.rstrip("\n") + "\n"
    parts[aoi] = section.rstrip("\n") + "\n"
    return "\n".join(parts[k] for k in sorted(parts))


def write_outputs(res: FusionResult, params: Mapping, settings: Settings | None = None) -> list:
    """Write restricted gpkg/ranking/manifest (via sbf.core.restricted) and public md files."""
    s = settings or load_settings()
    paths = []
    gpkg = restricted_path(f"candidates_{res.aoi}.gpkg", settings=s)
    res.ranked.to_file(gpkg, layer="candidates", driver="GPKG")
    ranking = restricted_path("ranking.csv", settings=s)
    cols = [c for c in RANKING_COLUMNS if c in res.ranked.columns]
    out = res.ranked[cols].copy()
    out.insert(0, "aoi_id", res.aoi)
    if ranking.exists():  # one ranking.csv for all AOIs: replace only this AOI's rows
        try:
            prev = pd.read_csv(ranking)
            prev = prev[prev["aoi_id"].astype(str) != res.aoi]
            out = pd.concat([prev, out], ignore_index=True) if len(prev) else out
        except (pd.errors.EmptyDataError, KeyError):
            pass
    out.to_csv(ranking, index=False)
    manifest = restricted_path(f"fusion_manifest_{res.aoi}.json", settings=s)
    manifest.write_text(
        json.dumps(
            {
                "aoi": res.aoi,
                "size_m": res.size_m,
                "weights": res.weights,
                "weights_source": res.weights_source,
                "params": dict(params),
                "n_candidates": len(res.ranked),
            },
            indent=2,
            default=str,
        )
    )
    for p in (gpkg, ranking, manifest):
        paths.append(encrypt_if_key(p))
    pub = s.outputs_public
    pub.mkdir(parents=True, exist_ok=True)
    (pub / "fusion_weights.md").write_text(weights_table_md(res.weights, res.weights_source))
    (pub / "fusion_summary.md").write_text(
        _merge_summary((pub / "fusion_summary.md"), res.aoi, public_summary_md(res, params))
    )
    res.paths = paths
    return paths


__all__ = ["FAMILIES", "FusionResult", "public_summary_md", "run_fusion", "write_outputs"]
