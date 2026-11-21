"""Coverage map, summary and go/no-go outputs (ticket 03)."""

from __future__ import annotations

import json
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from shapely import wkt

from sbf.catalog.counts import SensorCount
from sbf.catalog.go import NO_GO, GoDecision, best_frames, era, evaluate_go, per_feature_frames
from sbf.core.config import Settings

COST_NOTE = {
    "scanned": "free (already scanned)",
    "needs_order": "$30/frame, needs human approval",
    "unknown": "unknown",
}


def read_coverage(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, dtype={"entity_id": str, "footprint_wkt": str, "intersecting_features": str}
    )
    df["intersecting_features"] = df["intersecting_features"].fillna("")
    df["footprint_wkt"] = df["footprint_wkt"].fillna("")
    return df


def coverage_map(
    coverage: pd.DataFrame, reference: gpd.GeoDataFrame, aois: gpd.GeoDataFrame, out: Path
) -> Path:
    m = folium.Map(location=[25.0, 2.5], zoom_start=6, tiles="OpenStreetMap")
    for r in aois.itertuples():
        folium.GeoJson(
            r.geometry.__geo_interface__,
            name=f"AOI {r.aoi_id}",
            style_function=lambda _: {"color": "black", "fill": False, "dashArray": "5"},
        ).add_to(m)
    fp = folium.FeatureGroup(name="Declass footprints")
    for r in coverage.itertuples():
        if not r.footprint_wkt:
            continue
        tip = (
            f"{r.entity_id} | {r.dataset} | {r.acquisition_date} | {r.resolution_class} | "
            f"{r.availability}"
        )
        folium.GeoJson(
            wkt.loads(r.footprint_wkt).__geo_interface__,
            tooltip=tip,
            style_function=lambda _: {"color": "#3366cc", "weight": 1, "fillOpacity": 0.05},
        ).add_to(fp)
    fp.add_to(m)
    rf = folium.FeatureGroup(name="Reference features")
    for r in reference.itertuples():
        if r.geometry is None:
            continue
        c = r.geometry.centroid  # refined features are polygons, not points
        folium.CircleMarker(
            [c.y, c.x],
            radius=5,
            color="red",
            fill=True,
            tooltip=f"{r.feature_id} ({r.precision})",
        ).add_to(rf)
    rf.add_to(m)
    folium.LayerControl().add_to(m)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    return out


def summary_markdown(
    coverage: pd.DataFrame,
    reference: gpd.GeoDataFrame,
    decision: GoDecision,
    counts: list[SensorCount],
    failures: list[dict],
) -> str:
    L = [
        "# Phase 1 coverage summary",
        "",
        f"**Decision: {decision.status}** (pending human review; SPEC section 6)",
        "",
    ]
    L += [f"- {r}" for r in decision.reasons]
    if decision.needs_review:
        L += [
            "",
            "## Browse-image review needed",
            "",
            (
                "The rule is satisfied only if frames with UNKNOWN cloud cover are usable. Check "
                "the browse images (`sbf ingest declass browse`) for these entity IDs, fill the "
                f"`usable` column (yes/no) in `{REVIEW_FILE}`, then re-run `sbf catalog report`:"
            ),
            "",
        ]
        L += [f"- {e}" for e in decision.review_entity_ids]
    L += [
        "",
        f"Frames in catalogue: {len(coverage)}. API failures recorded: {len(failures)}.",
        "",
        "## Per reference feature",
        "",
        "| Feature | Frames | Best frame | Date | Resolution | Cost | Era vs 1966-67 dismantling | Eras available |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in best_frames(coverage, reference).itertuples():
        if r.n_frames == 0:
            L.append(f"| {r.feature_id} | 0 | none | | | | | |")
        else:
            L.append(
                f"| {r.feature_id} | {r.n_frames} | {r.entity_id} | {r.acquisition_date} | "
                f"{r.resolution_class} | {COST_NOTE.get(r.availability, r.availability)} | "
                f"{r.era} | {r.eras_available} |"
            )
    L += [
        "",
        (
            "Era key: pre-1967 = before 1 Jan 1967, post-1967 = after 31 Dec 1967 (SPEC go rule); "
            "BASE has no geometry yet and cannot be intersected."
        ),
        "",
        "## Other sensors",
        "",
        "| Sensor | AOI | Status | Count | Detail |",
        "|---|---|---|---|---|",
    ]
    for c in counts:
        L.append(
            f"| {c.sensor} | {c.aoi_id} | {c.status} | {'' if c.count is None else c.count} | {c.detail} |"
        )
    if failures:
        L += ["", "## API failures (not filled in)", ""]
        L += [
            f"- {f['endpoint']} {f['context']}: {f['message']} (raw: {f.get('raw_file')})"
            for f in failures
        ]
    L += [
        "",
        (
            "Resolution classes derive from the camera/mission table in `sbf.catalog.resolution`, not "
            "from measured scan resolution. Stop here for human review of the go/no-go decision."
        ),
        "",
    ]
    return "\n".join(L)


def no_go_markdown(decision: GoDecision) -> str:
    return "\n".join(
        [
            "# NO_GO: Phase 1 coverage gate not met",
            "",
            "The pipeline stops here for human review (SPEC section 6, Phase 1).",
            "",
            "## What is missing",
            "",
        ]
        + [f"- {r}" for r in decision.reasons]
        + [""]
    )


REVIEW_FILE = "browse_review.csv"
REVIEW_COLUMNS = [
    "entity_id",
    "dataset",
    "date",
    "resolution_class",
    "features",
    "browse_url",
    "usable",
]


def read_review(path: Path) -> dict[str, bool]:
    """Human answers from browse_review.csv: entity_id -> usable. Blank/unclear rows are ignored."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    out: dict[str, bool] = {}
    for r in df.itertuples():
        v = str(getattr(r, "usable", "")).strip().lower()
        if v in {"yes", "y", "true", "1"}:
            out[r.entity_id] = True
        elif v in {"no", "n", "false", "0"}:
            out[r.entity_id] = False
    return out


def write_review(
    path: Path, coverage: pd.DataFrame, entity_ids: list[str], answers: dict[str, bool]
) -> Path:
    """Write the browse-review sheet for `entity_ids`, keeping earlier human answers.

    Extra columns already in the file (e.g. agent `suggested_usable` notes) are kept too.
    """
    extra = pd.DataFrame()
    if path.exists():
        old = pd.read_csv(path, dtype=str).fillna("")
        cols = [c for c in old.columns if c not in REVIEW_COLUMNS]
        if cols:
            extra = old[["entity_id", *cols]].drop_duplicates("entity_id")
    pf = per_feature_frames(coverage)
    rows = []
    for eid in entity_ids:
        g = pf[pf["entity_id"] == eid]
        feats = ";".join(sorted(set(g["feature_id"]))) if not g.empty else ""
        c = coverage[coverage["entity_id"] == eid].iloc[0]
        usable = "" if eid not in answers else ("yes" if answers[eid] else "no")
        rows.append(
            {
                "entity_id": eid,
                "dataset": c["dataset"],
                "date": c["acquisition_date"],
                "resolution_class": c["resolution_class"],
                "features": feats,
                "browse_url": c["browse_url"],
                "usable": usable,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
    if not extra.empty:
        out = out.merge(extra, on="entity_id", how="left")
        out[extra.columns] = out[extra.columns].fillna("")
    out.to_csv(path, index=False)
    return path


def build_report(
    settings: Settings,
    aois: gpd.GeoDataFrame,
    reference: gpd.GeoDataFrame,
    counts: list[SensorCount],
) -> GoDecision:
    """Write coverage_map.html, coverage_summary.md, sensor_counts.json, NO_GO.md /
    browse_review.csv as the three-state decision requires.

    Human answers in outputs/public/browse_review.csv (`usable` yes/no) are applied.
    """
    pub = settings.outputs_public
    pub.mkdir(parents=True, exist_ok=True)
    coverage = read_coverage(pub / "coverage.csv")
    fpath = settings.data_interim / "catalog" / "failures.json"
    failures = json.loads(fpath.read_text()) if fpath.exists() else []
    answers = read_review(pub / REVIEW_FILE)
    decision = evaluate_go(coverage, reference, review=answers)
    coverage_map(coverage, reference, aois, pub / "coverage_map.html")
    (pub / "coverage_summary.md").write_text(
        summary_markdown(coverage, reference, decision, counts, failures)
    )
    (pub / "sensor_counts.json").write_text(json.dumps([c.as_dict() for c in counts], indent=2))
    no_go = pub / "NO_GO.md"
    if decision.status == NO_GO:
        no_go.write_text(no_go_markdown(decision))
    else:
        no_go.unlink(missing_ok=True)
    if decision.needs_review:
        write_review(
            pub / REVIEW_FILE,
            coverage,
            decision.review_entity_ids + [e for e in answers if e in set(coverage["entity_id"])],
            answers,
        )
    return decision


__all__ = [
    "build_report",
    "coverage_map",
    "era",
    "no_go_markdown",
    "read_coverage",
    "read_review",
    "summary_markdown",
    "write_review",
]
