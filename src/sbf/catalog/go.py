"""Phase 1 go/no-go rule (SPEC section 6, Phase 1). Pure functions, no I/O."""

from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd

from sbf.catalog.resolution import RANK

CLOUD_MAX = 30.0  # usable frame: cloud < 30 %
GO_FEATURES = ("GZ1", "GZ2", "GZ3", "GZ4", "TA1")
DISMANTLING_YEAR = 1967  # pre = year < 1967, post = year > 1967 (1967 itself is neither)


GO = "GO"
NEEDS_REVIEW = "NEEDS_REVIEW"
NO_GO = "NO_GO"


@dataclass
class GoDecision:
    """Three-state Phase 1 decision. `go` is True only for status GO."""

    status: str = NO_GO
    reasons: list[str] = field(default_factory=list)  # what is missing (NO_GO) or satisfied (GO)
    usable_hires_frames: list[str] = field(default_factory=list)  # entity IDs, criterion A (known)
    temporal_features: list[str] = field(default_factory=list)  # feature IDs satisfying criterion B
    # <=3m frames over GO features with unknown cloud cover and no human answer yet
    review_entity_ids: list[str] = field(default_factory=list)

    @property
    def go(self) -> bool:
        return self.status == GO

    @property
    def needs_review(self) -> bool:
        return self.status == NEEDS_REVIEW


def per_feature_frames(coverage: pd.DataFrame) -> pd.DataFrame:
    """One row per (feature_id, frame) from the `;`-joined `intersecting_features` column."""
    if coverage.empty:
        return coverage.assign(feature_id=pd.Series(dtype=str))
    df = coverage.copy()
    df["feature_id"] = df["intersecting_features"].fillna("").astype(str).str.split(";")
    df = df.explode("feature_id")
    return df[df["feature_id"].fillna("") != ""].reset_index(drop=True)


def _year(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.year


def era(acq: str | None) -> str:
    y = pd.to_datetime(acq, errors="coerce")
    if pd.isna(y):
        return "undated"
    return (
        "pre-1967"
        if y.year < DISMANTLING_YEAR
        else "post-1967"
        if y.year > DISMANTLING_YEAR
        else "1967"
    )


def evaluate_go(
    coverage_df: pd.DataFrame,
    reference_gdf: gpd.GeoDataFrame,
    *,
    review: dict[str, bool] | None = None,
) -> GoDecision:
    """Three-state rule. Criteria: (A) >=1 usable (cloud<30) frame in the <=3m class over
    GZ1-GZ4 or TA1, AND (B) for some reference feature a frame before 1967 and one after.

    - GO: the rule holds using only frames with known cloud < 30 (or a human `review` answer
      True for an unknown-cloud frame).
    - NEEDS_REVIEW: the rule holds only if still-unreviewed unknown-cloud frames are counted;
      `review_entity_ids` lists exactly those frames for a human browse-image check.
    - NO_GO: the rule fails even counting unknown-cloud frames.

    `review` maps entity_id -> usable (human answers); it only overrides *unknown* cloud cover.
    Features absent from `reference_gdf` are ignored.
    """
    review = review or {}
    known = set(reference_gdf["feature_id"]) if len(reference_gdf) else set()
    pf = per_feature_frames(coverage_df)
    pf = pf[pf["feature_id"].isin(known)] if not pf.empty else pf
    reasons: list[str] = []

    hi = (
        pf[(pf["feature_id"].isin(GO_FEATURES)) & (pf["resolution_class"] == "<=3m")]
        if not pf.empty
        else pf
    )
    usable_ids: list[str] = []
    pending_ids: list[str] = []
    rejected = 0
    usable_feats: set[str] = set()
    if not hi.empty:
        cloud = pd.to_numeric(hi["cloud_cover"], errors="coerce")
        for (_, row), c in zip(hi.iterrows(), cloud, strict=True):
            eid = row["entity_id"]
            if pd.notna(c):
                ok = c < CLOUD_MAX
            elif eid in review:
                ok = review[eid]
                rejected += 0 if ok else 1
            else:
                pending_ids.append(eid)
                continue
            if ok:
                usable_ids.append(eid)
                usable_feats.add(row["feature_id"])
    usable_ids = sorted(set(usable_ids))
    pending_ids = sorted(set(pending_ids))

    if usable_ids:
        reasons.append(
            f"A met: {len(usable_ids)} usable frame(s) <=3m over {', '.join(sorted(usable_feats))} "
            f"(cloud<{CLOUD_MAX:.0f}% or confirmed by human browse review)"
        )
    elif pending_ids:
        reasons.append(
            f"A met only if unknown-cloud frames count: {len(pending_ids)} <=3m frame(s) over "
            f"{'/'.join(GO_FEATURES)} have no cloud value and need a browse-image check"
        )
    else:
        reasons.append(
            f"A NOT met: no usable (cloud<{CLOUD_MAX:.0f}%) <=3m frame over {'/'.join(GO_FEATURES)}"
        )
    if rejected:
        reasons.append(f"  note: {rejected} unknown-cloud frame(s) rejected by human review")

    temporal: list[str] = []
    if not pf.empty:
        yrs = _year(pf["acquisition_date"])
        for fid, grp in pf.assign(_y=yrs).groupby("feature_id"):
            if (grp["_y"] < DISMANTLING_YEAR).any() and (grp["_y"] > DISMANTLING_YEAR).any():
                temporal.append(str(fid))
    b_ok = bool(temporal)
    reasons.append(
        f"B met: pre- and post-1967 frames exist over {', '.join(sorted(temporal))}"
        if b_ok
        else "B NOT met: no reference feature has both a pre-1967 and a post-1967 frame"
    )
    if b_ok and usable_ids:
        status = GO
    elif b_ok and pending_ids:
        status = NEEDS_REVIEW
    else:
        status = NO_GO
    return GoDecision(status, reasons, usable_ids, sorted(temporal), pending_ids)


def best_frames(coverage: pd.DataFrame, reference: gpd.GeoDataFrame) -> pd.DataFrame:
    """Per reference feature: best frame (finest class, then scanned/free, then earliest)."""
    pf = per_feature_frames(coverage)
    rows = []
    for fid in reference["feature_id"]:
        g = pf[pf["feature_id"] == fid] if not pf.empty else pf
        if g.empty:
            rows.append({"feature_id": fid, "n_frames": 0})
            continue
        g = g.assign(
            _r=g["resolution_class"].map(RANK).fillna(3),
            _a=(g["availability"] != "scanned").astype(int),
            _d=g["acquisition_date"].fillna("9999"),
        ).sort_values(["_r", "_a", "_d"])
        b = g.iloc[0]
        rows.append(
            {
                "feature_id": fid,
                "n_frames": len(g),
                "entity_id": b["entity_id"],
                "acquisition_date": b["acquisition_date"],
                "resolution_class": b["resolution_class"],
                "availability": b["availability"],
                "era": era(b["acquisition_date"]),
                "eras_available": ", ".join(sorted({era(d) for d in g["acquisition_date"]})),
            }
        )
    return pd.DataFrame(rows)
