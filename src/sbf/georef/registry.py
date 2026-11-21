"""Batch georeferencing and the frame registry (ticket 08; consumed by ticket 11).

`run_batch` georeferences every downloaded frame (panoramic -> strips automatically), continues
past failures (recorded, not raised) and writes
`data/processed/historical/registry.parquet` + `outputs/public/georef_summary.md`.

Inclusion rule (SPEC Phase 3, hardened after ticket 08): the honest number is `cv_rmse_m`
(spatial k-fold over GCP blocks), NOT the checkpoint RMSE (checkpoints sit next to training GCPs).
A frame is included iff `cv_rmse_m <= 20` AND `gcp_block_occupancy >= MIN_BLOCK_OCCUPANCY` (0.4:
share of a 5x5 grid of frame blocks with >= 3 GCPs). 10-20 m is included but flagged
(`rmse_gt_10m`); `low_gcp_coverage` and `cv_vs_checkpoint_gap` (cv > 2x checkpoint) are further
warnings. Failed/unsolved frames and frames without a CV number are excluded. Pixels outside GCP
support are marked in `{ID}_support.tif` (`support_file`; read it with `load_support`). `era` follows the catalog convention (`sbf.catalog.go.era`, boundary
`DISMANTLING_YEAR`): `pre_1967` (< 1967), `dismantling_1967` (1967), `post_1967` (> 1967),
`undated` when the acquisition date is missing.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

from sbf.catalog.go import DISMANTLING_YEAR
from sbf.core.config import Settings, load_settings
from sbf.georef.models import (
    MIN_BLOCK_OCCUPANCY,
    RMSE_EXCLUDE_M,
    RMSE_WARN_M,
    GeorefReport,
)

REGISTRY_COLUMNS = [
    "entity_id",
    "dataset",
    "camera",
    "acquisition_date",
    "era",
    "aoi",
    "method",
    "n_gcps",
    "checkpoint_rmse_m",
    "resolution_m",
    "included_for_change_detection",
    "warnings",
    "georef_file",
    "status",
    "error",
    # added with the spatial-CV gate (existing columns unchanged)
    "cv_rmse_m",
    "random_checkpoint_rmse_m",
    "gcp_hull_fraction",
    "gcp_block_occupancy",
    "support_file",
]


def registry_path(settings: Settings | None = None) -> Path:
    s = settings or load_settings()
    return s.data_processed / "historical" / "registry.parquet"


def summary_path(settings: Settings | None = None) -> Path:
    s = settings or load_settings()
    return s.outputs_public / "georef_summary.md"


def era_of(acquisition_date) -> str:
    """`pre_1967` | `dismantling_1967` | `post_1967` | `undated` (catalog boundary convention)."""
    d = pd.to_datetime(acquisition_date, errors="coerce")
    if pd.isna(d):
        return "undated"
    if d.year < DISMANTLING_YEAR:
        return "pre_1967"
    return "dismantling_1967" if d.year == DISMANTLING_YEAR else "post_1967"


def is_included(
    cv_rmse: float | None, georef_file: str | None, occupancy: float | None = None
) -> bool:
    """Gate: cv_rmse_m <= RMSE_EXCLUDE_M and gcp_block_occupancy >= MIN_BLOCK_OCCUPANCY."""

    def ok(v):
        return v is not None and not pd.isna(v)

    return (
        ok(cv_rmse)
        and cv_rmse <= RMSE_EXCLUDE_M
        and ok(occupancy)
        and occupancy >= MIN_BLOCK_OCCUPANCY
        and bool(georef_file)
    )


def registry_row(
    entity_id: str,
    cov: pd.Series | None,
    rep: GeorefReport | None,
    error: str | None = None,
    aoi: str | None = None,
) -> dict:
    cov = cov if cov is not None else pd.Series(dtype=object)
    row = {
        "entity_id": entity_id,
        "dataset": str(cov.get("dataset") or ""),
        "camera": str(cov.get("camera") or ""),
        "acquisition_date": pd.to_datetime(cov.get("acquisition_date"), errors="coerce"),
        "aoi": aoi or str(cov.get("aoi_id") or ""),
        "method": "none",
        "n_gcps": 0,
        "checkpoint_rmse_m": None,
        "resolution_m": None,
        "included_for_change_detection": False,
        "warnings": "",
        "georef_file": "",
        "status": "failed" if error else "ok",
        "error": error or "",
        "cv_rmse_m": None,
        "random_checkpoint_rmse_m": None,
        "gcp_hull_fraction": None,
        "gcp_block_occupancy": None,
        "support_file": "",
    }
    row["era"] = era_of(row["acquisition_date"])
    if rep is not None:
        row.update(
            method="manual_gcps" if rep.gcp_source == "manual_gcps" else rep.method,
            n_gcps=rep.n_gcps,
            checkpoint_rmse_m=rep.checkpoint_rmse_m,
            resolution_m=rep.resolution_m,
            warnings=";".join(rep.warnings),
            georef_file=rep.georef_file or "",
            cv_rmse_m=rep.cv_rmse_m,
            random_checkpoint_rmse_m=rep.random_checkpoint_rmse_m,
            gcp_hull_fraction=rep.gcp_hull_fraction,
            gcp_block_occupancy=rep.gcp_block_occupancy,
            support_file=rep.support_file or "",
        )
        row["included_for_change_detection"] = (
            is_included(rep.cv_rmse_m, rep.georef_file, rep.gcp_block_occupancy) and not error
        )
        if rep.method == "none" and not error:
            row["status"] = "unsolved"
    return row


def downloaded_entities(settings: Settings) -> list[str]:
    root = settings.data_raw / "declass"
    if not root.exists():
        return []
    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir()
        and any(
            p.suffix.lower() in {".tif", ".tiff"} and "browse" not in p.name.lower()
            for p in d.glob("*")
        )
    )


def _frame_dir(entity_id: str, settings: Settings) -> Path:
    return settings.data_processed / "historical" / entity_id


def run_batch(
    settings: Settings | None = None,
    *,
    aoi: str | None = None,
    only_missing: bool = False,
    runner: Callable[..., GeorefReport] | None = None,
    entities: list[str] | None = None,
    **run_kw,
) -> pd.DataFrame:
    """Georeference every downloaded frame; write registry.parquet and georef_summary.md.

    `aoi` restricts to frames whose coverage row is in that AOI. `only_missing` skips frames that
    already have a `georef_report.json` (their registry row is rebuilt from it). One failing frame
    never stops the batch (status `failed`, message in `error`)."""
    from sbf.catalog.report import read_coverage
    from sbf.georef.pipeline import run_frame

    s = settings or load_settings()
    runner = runner or run_frame
    cov = read_coverage(s.outputs_public / "coverage.csv")
    names = entities if entities is not None else downloaded_entities(s)
    rows: list[dict] = []
    for eid in names:
        crow = cov[cov["entity_id"] == eid]
        if aoi:
            crow = crow[crow["aoi_id"] == aoi]
        if crow.empty:
            if aoi is None:
                rows.append(registry_row(eid, None, None, "not in coverage.csv"))
            continue
        cr = crow.iloc[0]
        rep_file = _frame_dir(eid, s) / "georef_report.json"
        try:
            if only_missing and rep_file.exists():
                rep = GeorefReport.model_validate_json(rep_file.read_text())
            else:
                rep = runner(
                    eid, settings=s, aoi=aoi or str(cr.get("aoi_id") or "") or None, **run_kw
                )
            rows.append(registry_row(eid, cr, rep, aoi=aoi))
        except Exception as e:  # noqa: BLE001 - batch must continue past any per-frame failure
            msg = f"{type(e).__name__}: {e}"
            (_frame_dir(eid, s)).mkdir(parents=True, exist_ok=True)
            (_frame_dir(eid, s) / "georef_error.txt").write_text(traceback.format_exc())
            rows.append(registry_row(eid, cr, None, msg, aoi=aoi))
    df = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
    write_registry(df, s)
    return df


def write_registry(df: pd.DataFrame, settings: Settings | None = None) -> Path:
    s = settings or load_settings()
    out = registry_path(s)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["acquisition_date"] = pd.to_datetime(df["acquisition_date"], errors="coerce").dt.date
    df["acquisition_date"] = df["acquisition_date"].where(df["acquisition_date"].notna(), None)
    df.to_parquet(out, index=False)
    sp = summary_path(s)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(summary_markdown(df))
    return out


def load_registry(
    included_only: bool = True, settings: Settings | None = None, path: Path | None = None
) -> pd.DataFrame:
    """The frame registry. `acquisition_date` is datetime64[ns] (date only), `era` one of
    pre_1967 / dismantling_1967 / post_1967 / undated. `included_only` keeps frames usable for
    change detection (cv_rmse_m <= 20 m, gcp_block_occupancy >= 0.4, georeferenced file present)."""
    p = path or registry_path(settings)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `sbf georef batch` first")
    df = pd.read_parquet(p)
    df["acquisition_date"] = pd.to_datetime(df["acquisition_date"], errors="coerce")
    if included_only:
        df = df[df["included_for_change_detection"]]
    return df.reset_index(drop=True)


def load_support(
    entity_id: str, settings: Settings | None = None
) -> tuple[np.ndarray, Affine] | None:
    """Support mask of a frame: (uint8 array, 1 = within GCP support / 0 = extrapolated, affine
    transform) on the exact grid of `{ID}_georef.tif`, or None when no mask was written.
    Change detection should ignore pixels where the array is 0."""
    from sbf.georef.pipeline import processed_dir

    p = processed_dir(entity_id, settings) / f"{entity_id}_support.tif"
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        return src.read(1), src.transform


def summary_markdown(df: pd.DataFrame) -> str:
    n = len(df)
    inc = int(df["included_for_change_detection"].sum())
    ok = df["status"].eq("ok") & df["cv_rmse_m"].notna()
    flagged = df[df["included_for_change_detection"] & (df["cv_rmse_m"] > RMSE_WARN_M)]
    excl = df[df["status"].eq("ok") & ~df["included_for_change_detection"]]
    n_cov = int((ok & (df["gcp_block_occupancy"] < MIN_BLOCK_OCCUPANCY)).sum())
    failed = df[~df["status"].eq("ok")]
    lines = [
        "# Georeferencing summary (Phase 3)",
        "",
        f"- Frames processed: {n}",
        (
            f"- Included for change detection (CV RMSE <= {RMSE_EXCLUDE_M:.0f} m and block "
            f"occupancy >= {MIN_BLOCK_OCCUPANCY:.0%}): {inc}"
        ),
        f"- Included but flagged ({RMSE_WARN_M:.0f}-{RMSE_EXCLUDE_M:.0f} m): {len(flagged)}",
        f"- Excluded (CV RMSE > {RMSE_EXCLUDE_M:.0f} m, low GCP coverage or no CV): {len(excl)}",
        f"- of which low GCP coverage: {n_cov}",
        f"- Failed or unsolved (no usable model): {len(failed)}",
        "",
        (
            "Headline accuracy is the spatial-CV RMSE (leave-blocks-out over GCPs), which tests "
            "the fit where it has no nearby GCPs. Random checkpoints sit next to training GCPs and "
            "are optimistic (shown for comparison only). Use `{ID}_support.tif` to ignore pixels "
            "far from the GCP hull."
        ),
        "",
        (
            "| entity_id | dataset | camera | date | era | aoi | method | GCPs | CV RMSE (m) | "
            "blocked chk (m) | random chk (m) | hull | occupancy | res (m) | included | warnings |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def f(v, fmt):
        return "" if v is None or pd.isna(v) else format(v, fmt)

    for r in df.itertuples():
        d = "" if pd.isna(r.acquisition_date) else str(pd.to_datetime(r.acquisition_date).date())
        w = r.warnings if r.status == "ok" else f"{r.status}: {r.error}".strip()
        lines.append(
            f"| {r.entity_id} | {r.dataset} | {r.camera} | {d} | {r.era} | {r.aoi} | {r.method} | "
            f"{r.n_gcps} | {f(r.cv_rmse_m, '.1f')} | {f(r.checkpoint_rmse_m, '.1f')} | "
            f"{f(r.random_checkpoint_rmse_m, '.1f')} | {f(r.gcp_hull_fraction, '.2f')} | "
            f"{f(r.gcp_block_occupancy, '.2f')} | {f(r.resolution_m, '.2f')} | "
            f"{'yes' if r.included_for_change_detection else 'no'} | {w} |"
        )
    return "\n".join(lines) + "\n"
