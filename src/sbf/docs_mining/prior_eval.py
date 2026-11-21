"""Does the document prior improve detector recall@K? (ticket 15)

For every existing detector score table of an AOI:

* ``product``: ``combined = score_norm x prior_score_norm``
* ``alpha``:   ``combined = score_norm x (ALPHA + (1 - ALPHA) x prior_score_norm)``, ALPHA = 0.5

``prior_score_norm`` is the rank-normalised tile-mean prior (per tile size). Both are scored
with ``sbf.detect.harness.evaluate`` (same reference features, exclusions, K values) and compared
with the detector alone: output is the table of delta recall@K and delta AP.

Leakage control (the prior used for evaluation is built WITHOUT these claims)
-----------------------------------------------------------------------------
1. Claims with ``describes_reference_feature == True`` (the claim talks about a validation
   feature, e.g. a ground zero).
2. Claims whose own unit-mass surface puts at least ``LEAK_MASS_THRESHOLD`` (2 %) of its mass
   inside the footprint of ANY reference feature (harness footprint: point buffered to
   ``expected_area_km2``). Rationale: every gazetteer anchor is itself a reference feature, so
   "10 km NE of GZ1" is not leakage about GZ1's position per se, but it IS leakage for any
   reference feature 10 km NE of GZ1: the prior would point at the answer. The geometric test
   catches exactly that, whatever the anchor text says. Anchors that are not resolved cannot leak
   spatially (they are not in the prior) and are reported separately.
Excluded claim IDs and reasons are listed in ``outputs/public/prior_eval.md`` (IDs only; no
quotes, no coordinates).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import geometry_mask
from rasterio.transform import Affine

from sbf.core.aoi import load_reference_features
from sbf.core.config import Settings, load_settings
from sbf.detect.harness import DEFAULT_KS, evaluate, reference_footprints
from sbf.detect.scores import rank_normalise, read_scores
from sbf.docs_mining.prior import (
    Claim,
    ResolvedClaim,
    build_prior,
    grid_axes,
    load_accepted_claims,
    resolve_claims,
    resolved_claim_surface,
    tile_prior_table,
)
from sbf.ingest.modern.grid import modern_grid

log = logging.getLogger(__name__)

ALPHA = 0.5
LEAK_MASS_THRESHOLD = 0.02
SKIP_DETECTORS = {"prior", "random"}


def leaking_claims(
    claims: list[ResolvedClaim],
    grid,
    footprints: gpd.GeoDataFrame,
    threshold: float = LEAK_MASS_THRESHOLD,
) -> dict[str, str]:
    """``{extraction_id: reason}`` for usable claims whose own mass on any footprint >= threshold.

    ``footprints`` (columns ``feature_id``, ``geometry``) must be in the grid CRS.
    """
    xs, ys = grid_axes(grid)
    t, _, _, crs = grid
    out: dict[str, str] = {}
    for rc in claims:
        if not rc.usable:
            continue
        surf = resolved_claim_surface(rc, xs, ys, crs)
        if surf is None:
            continue
        rs, cs, w = surf
        wt = t @ Affine.translation(int(cs.start), int(rs.start))
        hits = []
        for fid, geom in zip(footprints["feature_id"], footprints.geometry, strict=True):
            m = ~geometry_mask([geom], out_shape=w.shape, transform=wt, all_touched=True)
            mass = float(w[m].sum())
            if mass >= threshold:
                hits.append(f"{fid} ({mass:.2f})")
        if hits:
            out[rc.claim.extraction_id] = "prior mass on reference feature: " + ", ".join(hits)
    return out


@dataclass
class EvalResult:
    table: pd.DataFrame
    n_accepted: int
    used_ids: list[str] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)  # extraction_id -> reason
    unresolved: dict[str, list[str]] = field(default_factory=dict)  # status -> ids
    note: str = ""


def _combine(df: pd.DataFrame, prior_norm: pd.Series, variant: str) -> pd.DataFrame:
    out = df.copy()
    pn = prior_norm.reindex(out["tile_id"]).to_numpy()
    mult = pn if variant == "product" else ALPHA + (1 - ALPHA) * pn
    out["score"] = out["score_norm"].to_numpy() * mult
    out["score_norm"] = np.nan
    out["score_norm"] = out.groupby(["detector", "layer", "size_m"])["score"].transform(
        rank_normalise
    )
    return out


def _delta_table(before: pd.DataFrame, after: pd.DataFrame, variant: str, ks) -> pd.DataFrame:
    key = ["aoi_id", "detector", "layer", "size_m"]
    m = before.merge(after, on=key, suffixes=("_b", "_a"))
    out = m[key].copy()
    out["variant"] = variant
    out["n_features"] = m["n_features_b"]
    for k in ks:
        out[f"recall_at_{k}_before"] = m[f"recall_at_{k}_b"]
        out[f"recall_at_{k}_after"] = m[f"recall_at_{k}_a"]
        out[f"d_recall_at_{k}"] = m[f"recall_at_{k}_a"] - m[f"recall_at_{k}_b"]
    out["ap_before"], out["ap_after"] = m["ap_b"], m["ap_a"]
    out["d_ap"] = m["ap_a"] - m["ap_b"]
    return out


def evaluate_prior(
    aoi: str,
    detectors: list[str] | None = None,
    settings: Settings | None = None,
    *,
    claims: list[Claim] | None = None,
    scores: pd.DataFrame | None = None,
    grid=None,
    leakage_control: bool = True,
    exclusions: gpd.GeoDataFrame | None = None,
    reference: gpd.GeoDataFrame | None = None,
    ks=DEFAULT_KS,
    write: bool = True,
) -> EvalResult:
    """Evaluate prior x detector against the detector alone; see module docstring.

    ``claims``/``scores``/``grid`` default to the reviewed CSV, the stored score tables and the
    modern 10 m grid. ``leakage_control=False`` is a diagnostic only (never used for reporting).
    """
    s = settings or load_settings()
    claims = load_accepted_claims(s) if claims is None else claims
    grid = grid or modern_grid(aoi, s)
    reference = load_reference_features(s) if reference is None else reference
    crs = grid[3]
    resolved = resolve_claims(claims, aoi, settings=s)
    res = EvalResult(pd.DataFrame(), n_accepted=len(claims))
    for rc in resolved:
        if not rc.usable:
            res.unresolved.setdefault(rc.status, []).append(rc.claim.extraction_id)
    if leakage_control:
        for rc in resolved:
            if rc.claim.describes_reference_feature:
                res.excluded[rc.claim.extraction_id] = "describes_reference_feature"
        foot = reference_footprints(aoi, reference, crs)
        rest = [rc for rc in resolved if rc.claim.extraction_id not in res.excluded]
        res.excluded.update(leaking_claims(rest, grid, foot))
    keep = [rc for rc in resolved if rc.claim.extraction_id not in res.excluded]
    try:
        prior, res.used_ids, _ = build_prior(keep, grid)
    except ValueError:
        res.note = "no usable claims remain after exclusions; nothing evaluated"
        if write:
            write_report(aoi, res, ks, s)
        return res
    if scores is None:
        scores = read_scores(aoi, settings=s)
    scores = scores[~scores["detector"].isin(SKIP_DETECTORS)]
    if detectors is not None:
        scores = scores[scores["detector"].isin(detectors)]
    if scores.empty:
        res.note = "no detector score tables found; nothing evaluated"
        if write:
            write_report(aoi, res, ks, s)
        return res
    ptab = tile_prior_table(prior, grid, scores["tile_id"].unique().tolist(), aoi, s)
    ptab["prior_norm"] = ptab.groupby("size_m")["score"].transform(rank_normalise)
    prior_norm = ptab.set_index("tile_id")["prior_norm"]
    kw = {"reference": reference, "exclusions": exclusions, "settings": s, "ks": ks}
    before = evaluate(scores, **kw)
    parts = [
        _delta_table(before, evaluate(_combine(scores, prior_norm, v), **kw), v, ks)
        for v in ("product", "alpha")
    ]
    res.table = pd.concat(parts, ignore_index=True)
    if write:
        write_report(aoi, res, ks, s)
    return res


def _fmt(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def render_report(aoi: str, res: EvalResult, ks=DEFAULT_KS) -> str:
    """Markdown report: metrics, excluded-claim counts and IDs. No quotes, no coordinates."""
    reasons: dict[str, list[str]] = {}
    for cid, why in sorted(res.excluded.items()):
        reasons.setdefault(why.split(":")[0], []).append(cid)
    lines = [
        f"# Spatial prior evaluation ({aoi})",
        "",
        (
            "Combined scores: `product` = score_norm x prior_norm; "
            f"`alpha` = score_norm x ({ALPHA} + {1 - ALPHA} x prior_norm). Prior built without "
            "leakage-prone claims (see below). Deltas are after minus before."
        ),
        "",
        (
            f"- Accepted claims: {res.n_accepted}; used in prior: {len(res.used_ids)}; "
            f"excluded for leakage: {len(res.excluded)}"
        ),
    ]
    for status, ids in sorted(res.unresolved.items()):
        lines.append(f"- Not usable ({status}): {len(ids)}")
    if res.note:
        lines += ["", f"**{res.note}**"]
    if not res.table.empty:
        head = ["detector", "layer", "size", "variant", "feat"]
        head += [f"R@{k} before/after (d)" for k in ks] + ["AP before/after (d)"]
        lines += ["", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for _, r in res.table.iterrows():
            cells = [r["detector"], r["layer"], str(r["size_m"]), r["variant"],
                     str(int(r["n_features"]))]  # fmt: skip
            for k in ks:
                cells.append(
                    f"{_fmt(r[f'recall_at_{k}_before'])} / {_fmt(r[f'recall_at_{k}_after'])} "
                    f"({_fmt(r[f'd_recall_at_{k}'])})"
                )
            cells.append(f"{_fmt(r['ap_before'])} / {_fmt(r['ap_after'])} ({_fmt(r['d_ap'])})")
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Excluded claims (leakage control)", ""]
    if not res.excluded:
        lines.append("None.")
    for why, ids in reasons.items():
        lines.append(f"- {why}: {len(ids)}")
    for cid, why in sorted(res.excluded.items()):
        lines.append(f"  - `{cid}`: {why}")
    for status, ids in sorted(res.unresolved.items()):
        lines += ["", f"## Unresolved ({status})", "", "IDs: " + ", ".join(sorted(ids))]
    lines += ["", "Quotes and coordinates are internal and not included.", ""]
    return "\n".join(lines)


def write_report(aoi: str, res: EvalResult, ks, settings: Settings):
    out = settings.outputs_public / "prior_eval.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(aoi, res, ks))
    return out
