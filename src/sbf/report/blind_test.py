"""Phase 4 blind-test report and pass/fail gate (SPEC section 0 rules 1 and 5, Phase 4).

``evaluate_gate`` is a pure function of the harness metrics table. Labels are only used inside
``sbf.detect.harness``; this module consumes metrics and (for figures) per-group positive ranks.
Outputs are public: counts and metrics only, never coordinates (D8).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sbf.core.config import Settings, load_settings
from sbf.fuse.families import family_of

log = logging.getLogger(__name__)

# --- Operationalisation of SPEC's "recall@50 >= 0.5 on >= 1 historical layer and clearly beats
# random": a row passes iff ALL of the following hold (any AOI, any tile size).
GATE_K = 50
MIN_RECALL = 0.5
MIN_RECALL_MARGIN = 0.2  # recall@50 >= random_recall@50 + 0.2
MIN_AP_RATIO = 2.0  # AP >= 2 x random_AP
SMALL_N = 6  # <= this many scored reference features per AOI: recall moves in big steps
EXCLUDED_DETECTORS = frozenset({"random", "prior"})
HISTORICAL_FAMILIES = frozenset({"hist_anomaly", "hist_change"})
MODERN_FAMILIES = frozenset({"modern_optical", "SAR_L", "SAR_C", "DEM"})
STOP_NOTICE = (
    "STOP: blind test FAILED. SPEC requires a human review before any further phase "
    "(fusion, candidate sheets). Do not proceed automatically."
)
CRITERIA = {
    "k": GATE_K,
    "min_recall_at_50": MIN_RECALL,
    "min_margin_over_random_recall": MIN_RECALL_MARGIN,
    "min_ap_ratio_to_random": MIN_AP_RATIO,
    "layers": "historical (families hist_anomaly, hist_change)",
    "excluded_detectors": sorted(EXCLUDED_DETECTORS),
}
_ROW_KEYS = ["detector", "layer", "aoi_id", "size_m"]
_METRICS = ["recall_at_50", "random_recall_at_50", "ap", "random_ap", "n_features"]


@dataclass
class GateResult:
    status: str  # PASS | FAIL | NOT_RUN
    best_rows: list[dict] = field(default_factory=list)
    margin: float | None = None  # best recall@50 minus its random expectation
    modern_best: dict | None = None  # informative, never gating
    n_reference_features_scored: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    exclusions_status: str = "none"

    def to_json_dict(self, generated_at: str | None = None) -> dict:
        return {
            "status": self.status,
            "criteria": CRITERIA,
            "best_rows": self.best_rows,
            "margin": self.margin,
            "modern_best": self.modern_best,
            "n_reference_features_scored": self.n_reference_features_scored,
            "warnings": self.warnings,
            "generated_at": generated_at or datetime.now(UTC).isoformat(timespec="seconds"),
            "exclusions_status": self.exclusions_status,
        }


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _row_dict(r: pd.Series) -> dict:
    d = {
        "detector": str(r["detector"]),
        "layer": str(r["layer"]),
        "aoi": str(r["aoi_id"]),
        "size_m": int(r["size_m"]),
    }
    d["metrics"] = {k: _num(r.get(k)) for k in _METRICS}
    return d


def row_passes(r: pd.Series) -> bool:
    """The three-part pass rule on one metrics row (NaN never passes)."""
    rec, rrec = _num(r.get("recall_at_50")), _num(r.get("random_recall_at_50"))
    ap, rap = _num(r.get("ap")), _num(r.get("random_ap"))
    if None in (rec, rrec, ap, rap) or rap <= 0:
        return False
    return rec >= MIN_RECALL and rec >= rrec + MIN_RECALL_MARGIN and ap >= MIN_AP_RATIO * rap


def _with_family(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    m["family"] = [family_of(d, lay) for d, lay in zip(m["detector"], m["layer"], strict=True)]
    return m


def _gap(r: pd.Series) -> float:
    a, b = _num(r.get("recall_at_50")), _num(r.get("random_recall_at_50"))
    return -np.inf if a is None or b is None else a - b


def evaluate_gate(metrics: pd.DataFrame) -> GateResult:
    """PASS iff some non-random/prior historical row passes ``row_passes`` (see constants)."""
    if metrics is None or len(metrics) == 0:
        return GateResult("NOT_RUN")
    m = _with_family(metrics)
    m = m[~m["detector"].isin(EXCLUDED_DETECTORS)]
    hist = m[m["family"].isin(HISTORICAL_FAMILIES)]
    status_ex = "none"
    if "exclusions_status" in metrics:
        vals = set(metrics["exclusions_status"].dropna())
        status_ex = "draft" if "draft" in vals else ("reviewed" if "reviewed" in vals else "none")
    res = GateResult("NOT_RUN", exclusions_status=status_ex)
    if "n_features" in metrics:
        n = metrics.groupby("aoi_id")["n_features"].max().fillna(0)
        res.n_reference_features_scored = {str(k): int(v) for k, v in n.items()}
        for aoi, v in res.n_reference_features_scored.items():
            if v <= SMALL_N:
                res.warnings.append(
                    f"{aoi}: only {v} reference feature(s) scored; recall moves in steps of "
                    f"{1 / v:.2f}"
                    if v
                    else f"{aoi}: no reference feature scored"
                )
    mod = m[m["family"].isin(MODERN_FAMILIES)]
    if len(mod):
        res.modern_best = _row_dict(mod.loc[mod.apply(_gap, axis=1).idxmax()])
    if hist.empty:
        return res
    ok = hist[hist.apply(row_passes, axis=1)]
    if len(ok):
        res.status = "PASS"
        ok = ok.assign(_g=ok.apply(_gap, axis=1)).sort_values("_g", ascending=False)
        res.best_rows = [_row_dict(r) for _, r in ok.iterrows()]
    else:
        res.status = "FAIL"
        top = hist.assign(_g=hist.apply(_gap, axis=1)).sort_values("_g", ascending=False).head(3)
        res.best_rows = [_row_dict(r) for _, r in top.iterrows()]
    best = res.best_rows[0]["metrics"] if res.best_rows else {}
    if best.get("recall_at_50") is not None and best.get("random_recall_at_50") is not None:
        res.margin = best["recall_at_50"] - best["random_recall_at_50"]
    return res


def gate_path(settings: Settings | None = None) -> Path:
    return (settings or load_settings()).outputs_public / "blind_test_gate.json"


def write_gate(res: GateResult, settings: Settings | None = None) -> Path:
    p = gate_path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res.to_json_dict(), indent=2) + "\n")
    return p


# ----------------------------------------------------------------------------- report text
def _interpretation(metrics: pd.DataFrame, gate: GateResult) -> str:
    lines = ["## Honest interpretation", ""]
    n = gate.n_reference_features_scored
    n_txt = ", ".join(f"{a}: {v}" for a, v in sorted(n.items())) or "none"
    lines += [
        (
            f"- **Small n.** Reference features scored per AOI ({n_txt}). With so few positives, "
            "recall@K moves in large steps and a single feature decides a result; treat all numbers "
            "as indicative, not as precise estimates."
        )
    ]
    lines += [f"  - Warning: {w}" for w in gate.warnings]
    approx = bool(metrics["approximate_labels"].any()) if "approximate_labels" in metrics else True
    lines += [
        "- **Label noise.** Reference footprints are approximate"
        + (" (some features are of approximate precision, e.g. TA1 and PO1)" if approx else "")
        + ": point locations are buffered to the expected area, not surveyed outlines. "
        "A reference feature is a point-buffered footprint, so the true signature may be offset "
        "or differently sized (for scale, the Chornobyl benchmark dropped from about 98% to about "
        "65% once footprints were made realistic). Metrics are therefore likely pessimistic or "
        "optimistic by an unknown amount.",
        (
            "- **Invisible targets.** Deep burial and tunnel material leave little or no surface "
            "signature. A detector cannot recover what imagery does not show; misses on such "
            "features say nothing about the method's ceiling on visible ones."
        ),
        (
            "- **Resolution floor.** Tiles below roughly 16 px across (for example 64 m on 10 m "
            "Sentinel-2) carry mostly upsampling noise; results at those sizes are not informative."
        ),
    ]
    ex = gate.exclusions_status
    if ex == "reviewed":
        lines.append("- **Exclusions.** The infrastructure mask is human-reviewed.")
    else:
        lines.append(
            f"- **Exclusions (status: {ex}).** The exclusions mask is not human-reviewed, so "
            "all metrics are PROVISIONAL until it is."
        )
    lines.append(
        "- Terminology: this report speaks of *detected reference features* (known, public "
        "locations recovered by the blind test), never of confirmed contamination or burial."
    )
    return "\n".join(lines) + "\n"


def _negative_result(gate: GateResult) -> str:
    lines = [
        "## Negative result (SPEC Q4)",
        "",
        (
            'No historical detector/layer met the pass rule. Per SPEC, "a clear negative answer is '
            'a valid, publishable result": unsupervised anomaly and change detection, as '
            "configured here, did not recover the reference features better than random chance "
            "clearly enough to justify candidate generation."
        ),
        "",
    ]
    if gate.best_rows:
        b = gate.best_rows[0]
        mt = b["metrics"]
        lines.append(
            f"Closest result: `{b['detector']}` on `{b['layer']}` ({b['aoi']}, {b['size_m']} m): "
            f"recall@50 = {_fmt(mt['recall_at_50'])} vs random {_fmt(mt['random_recall_at_50'])}; "
            f"AP = {_fmt(mt['ap'])} vs random {_fmt(mt['random_ap'])}."
        )
        lines.append("")
    lines += [f"**{STOP_NOTICE}**", ""]
    return "\n".join(lines)


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def build_report(metrics: pd.DataFrame, gate: GateResult) -> str:
    """Markdown of ``outputs/public/blind_test.md`` (counts and metrics only)."""
    from sbf.detect.harness import to_markdown

    c = CRITERIA
    parts = [
        "# Phase 4 blind test",
        "",
        "## Method",
        "",
        (
            "Detectors are unsupervised and never see reference labels (SPEC rule: blind protocol). "
            "Labels are loaded only inside the scoring harness, which ranks tiles per detector, "
            "layer, AOI and tile size (excluded tiles removed before ranking) and reports "
            "feature-level recall@K, tile-level AP and the exact random expectation. Figures show "
            "known reference features only."
        ),
        "",
        "## Metrics",
        "",
        to_markdown(metrics) if len(metrics) else "_No score tables found._\n",
        "## Gate",
        "",
        f"**Decision: {gate.status}**",
        "",
        (
            f'Rule (operationalising "clearly beats random"): on at least one historical layer '
            f"(any AOI, any tile size; detectors {', '.join(c['excluded_detectors'])} excluded), "
            f"recall@{c['k']} >= {c['min_recall_at_50']}, recall@{c['k']} >= random recall + "
            f"{c['min_margin_over_random_recall']}, and AP >= {c['min_ap_ratio_to_random']} x random AP."
        ),
        "",
    ]
    if gate.status == "NOT_RUN":
        parts += ["No historical score tables were found; the gate was not evaluated.", ""]
    for b in gate.best_rows:
        mt = b["metrics"]
        parts.append(
            f"- {'Passing' if gate.status == 'PASS' else 'Best'}: `{b['detector']}` / "
            f"`{b['layer']}` / {b['aoi']} / {b['size_m']} m: recall@50 {_fmt(mt['recall_at_50'])} "
            f"(random {_fmt(mt['random_recall_at_50'])}), AP {_fmt(mt['ap'])} "
            f"(random {_fmt(mt['random_ap'])}), {mt['n_features']} features"
        )
    if gate.margin is not None:
        parts.append(f"- Margin over random recall@50: {gate.margin:.3f}")
    if gate.modern_best:
        b = gate.modern_best
        mt = b["metrics"]
        parts.append(
            f"- Best modern-layer result (informative, not gating): `{b['detector']}` / "
            f"`{b['layer']}` / {b['aoi']} / {b['size_m']} m: recall@50 "
            f"{_fmt(mt['recall_at_50'])} (random {_fmt(mt['random_recall_at_50'])})"
        )
    parts.append("")
    parts.append(_interpretation(metrics, gate))
    if gate.status == "FAIL":
        parts.append(_negative_result(gate))
    return "\n".join(parts)


# ----------------------------------------------------------------------------- orchestration
def positive_ranks(
    scores: pd.DataFrame, reference, exclusions, settings: Settings
) -> dict[tuple, dict]:
    """Per (aoi, detector, layer, size): ``{n_tiles, tiles_per_feature, best_ranks, pos_ranks}``.

    Ranks are 1-based over non-excluded tiles (same ordering as the harness). Only positive-tile
    ranks leave this function: no tile ids, no geometry.
    """
    import geopandas as gpd

    from sbf.detect.harness import reference_footprints
    from sbf.features.tiling import apply_exclusions, tile_geometries

    out: dict[tuple, dict] = {}
    for (aoi, size), sub in scores.groupby(["aoi_id", "size_m"]):
        tiles = apply_exclusions(
            tile_geometries(list(sub["tile_id"].unique()), settings.overlap, settings), exclusions
        )
        keep = tiles[~tiles["excluded"]]
        foot = reference_footprints(aoi, reference, settings.crs(aoi))
        hits = gpd.sjoin(keep[["tile_id", "geometry"]], foot, predicate="intersects", how="inner")[
            ["tile_id", "feature_id"]
        ]
        for (det, layer), g in sub.groupby(["detector", "layer"]):
            g = g[g["tile_id"].isin(set(keep["tile_id"]))]
            g = g.sort_values(["score", "tile_id"], ascending=[False, True], kind="stable")
            rank = pd.Series(np.arange(1, len(g) + 1), index=g["tile_id"].values)
            h = hits[hits["tile_id"].isin(rank.index)]
            if h.empty:
                continue
            h = h.assign(rank=rank.loc[h["tile_id"]].to_numpy())
            pf = h.groupby("feature_id").agg(n=("tile_id", "nunique"), best=("rank", "min"))
            out[(aoi, det, layer, int(size))] = {
                "n_tiles": len(g),
                "tiles_per_feature": pf["n"].tolist(),
                "best_ranks": pf["best"].tolist(),
                "pos_ranks": sorted(h["tile_id"].map(rank).astype(int).unique().tolist()),
            }
    return out


def run_blind_test(
    aois: list[str] | None = None, settings: Settings | None = None, figures: bool = True
) -> GateResult:
    """Read all score tables, evaluate, write report + gate json (+ figures)."""
    from sbf.detect.exclusions import load_exclusions
    from sbf.detect.harness import evaluate
    from sbf.detect.scores import read_scores

    s = settings or load_settings()
    folder = s.data_processed / "scores"
    found = sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
    use = [a for a in found if aois is None or a in aois]
    frames = [read_scores(a, settings=s) for a in use]
    frames = [f for f in frames if len(f)]
    s.outputs_public.mkdir(parents=True, exist_ok=True)
    if not frames:
        res = GateResult("NOT_RUN")
        metrics = pd.DataFrame()
    else:
        scores = pd.concat(frames, ignore_index=True)
        metrics = evaluate(scores, settings=s)
        res = evaluate_gate(metrics)
    (s.outputs_public / "blind_test.md").write_text(build_report(metrics, res))
    write_gate(res, s)
    log.info("blind test: %s (%d metric rows)", res.status, len(metrics))
    if figures and frames:
        from sbf.core.aoi import load_reference_features
        from sbf.report.blind_test_figures import write_figures

        write_figures(scores, load_reference_features(s), load_exclusions(s), s)
    return res
