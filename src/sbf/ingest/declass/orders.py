"""Ranked order list for UNSCANNED Declass frames (ticket 04).

THIS MODULE NEVER PLACES AN ORDER. Purchasing scans is a human gate (SPEC section 0.1):
the output is a CSV for a person to review and approve.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sbf.catalog.go import DISMANTLING_YEAR, per_feature_frames

# SPEC section 0.1 / data-source table: USGS scan orders cost $30 per frame.
COST_PER_FRAME_USD = 30.0

# Value weights (documented heuristics, not from SPEC): resolution dominates, then date, then
# how many reference features the frame covers (+1 each).
RESOLUTION_WEIGHT = {"<=3m": 3, "3-10m": 2, "unknown": 1, ">10m": 0}


def date_weight(acq: str | None) -> int:
    """Pre-1967 frames are highest (baseline before dismantling, 3); 1967-68 brackets the
    1966-67 dismantling from after (2); later frames 1; undated 0."""
    d = pd.to_datetime(acq, errors="coerce")
    if pd.isna(d):
        return 0
    if d.year < DISMANTLING_YEAR:
        return 3
    return 2 if d.year <= DISMANTLING_YEAR + 1 else 1


ORDER_COLUMNS = [
    "rank",
    "entity_id",
    "dataset",
    "acquisition_date",
    "resolution_class",
    "features",
    "n_features",
    "score",
    "estimated_cost_usd",
    "cumulative_cost_usd",
    "browse_url",
]


def build_order_list(coverage: pd.DataFrame, feature_ids: set[str] | None = None) -> pd.DataFrame:
    """Unscanned (`needs_order`) frames intersecting reference features, ranked by value.

    score = resolution weight + date weight + number of covered reference features.
    `feature_ids` restricts which features count (default: all tagged in the coverage).
    """
    pf = per_feature_frames(coverage)
    if pf.empty:
        return pd.DataFrame(columns=ORDER_COLUMNS)
    pf = pf[pf["availability"] == "needs_order"]
    if feature_ids is not None:
        pf = pf[pf["feature_id"].isin(feature_ids)]
    rows = []
    for eid, g in pf.groupby("entity_id"):
        first = g.iloc[0]
        feats = sorted(set(g["feature_id"]))
        score = (
            RESOLUTION_WEIGHT.get(first["resolution_class"], 1)
            + date_weight(first["acquisition_date"])
            + len(feats)
        )
        rows.append(
            {
                "entity_id": eid,
                "dataset": first["dataset"],
                "acquisition_date": first["acquisition_date"],
                "resolution_class": first["resolution_class"],
                "features": ";".join(feats),
                "n_features": len(feats),
                "score": score,
                "estimated_cost_usd": COST_PER_FRAME_USD,
                "browse_url": first["browse_url"],
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=ORDER_COLUMNS)
    df["_d"] = df["acquisition_date"].fillna("9999")
    df = df.sort_values(["score", "_d", "entity_id"], ascending=[False, True, True]).drop(
        columns="_d"
    )
    df = df.reset_index(drop=True)
    df["rank"] = df.index + 1
    df["cumulative_cost_usd"] = df["estimated_cost_usd"].cumsum()
    return df[ORDER_COLUMNS]


def write_order_list(df: pd.DataFrame, path: Path) -> float:
    """Write the CSV; return the total estimated cost in USD."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return float(df["estimated_cost_usd"].sum()) if len(df) else 0.0
