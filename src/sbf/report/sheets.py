"""Review sheets (restricted), restricted index and public aggregate stats (SPEC Phase 6, D8).

Sheets are self-contained HTML (base64 PNG chips) written only via ``sbf.core.restricted``.
Coordinates appear only in one labelled block of a restricted sheet and are never logged.
No fieldwork content (routes, access, visit planning) is produced (SPEC 0.4).
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
from collections.abc import Mapping
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio
import shapely
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry

from sbf.core.config import Settings, load_settings
from sbf.core.restricted import encrypt_if_key, restricted_path
from sbf.fuse.candidates import LABEL_MULTIPLIER, SCORE_BANDS
from sbf.fuse.families import FAMILIES
from sbf.ingest.modern.chips import extract_chip, stretch
from sbf.report.confidence import (
    INVISIBLE_CAVEAT,
    LABEL_NOISE_CAVEAT,
    STATEMENTS,
    Confidence,
    assess_confidence,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
TEMPLATES = Path(__file__).parent / "templates"
SCALE_BAR_M = 200.0
MODERN_CHIP_LAYERS = ("s2_median", "palsar_hv", "dem")
SHEET_DIR = "candidate_sheets"
CONF_CSV = "candidate_confidence_{aoi}.csv"
UNENCRYPTED_WARNING = (
    "WARNING: RESTRICTED_KEY is not set; restricted outputs (candidate sheets, confidence "
    "tables) are unencrypted at rest."
)

_env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"]))


# ---- chips ----------------------------------------------------------------------------------
def _reproject(geom: BaseGeometry, src_crs, dst_crs) -> BaseGeometry:
    t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely.transform(geom, lambda c: np.column_stack(t.transform(c[:, 0], c[:, 1])))


def _pixel_size_m(src: rasterio.DatasetReader, lat: float) -> float:
    if src.crs is not None and src.crs.is_geographic:
        return abs(src.transform.a) * 111_320.0 * math.cos(math.radians(lat))
    return abs(src.transform.a)


def render_chip(
    path: str | Path,
    geom: BaseGeometry,
    geom_crs: str,
    *,
    rgb: bool = False,
    half_width_m: float | None = None,
) -> str | None:
    """Base64 PNG of a raster chip around `geom` with its outline and a 200 m scale bar.

    None if the candidate is not covered by the raster.
    """
    with rasterio.open(path) as src:
        g = _reproject(geom, geom_crs, src.crs)
        c = g.centroid
        res = _pixel_size_m(src, c.y)
        minx, miny, maxx, maxy = g.bounds
        ext_m = max(maxx - minx, maxy - miny) * (111_320.0 if src.crs.is_geographic else 1.0)
        hw_m = half_width_m or max(2 * SCALE_BAR_M * 0.75, 2.0 * ext_m)
        hw_pix = round(hw_m / res)
        bands = (3, 2, 1) if rgb and src.count >= 3 else (1,)
        chip = extract_chip(path, c.x, c.y, hw_m * (abs(src.transform.a) / res), bands)
        if chip is None:
            return None
        row, col = src.index(c.x, c.y)
        inv = ~src.transform
        pts = [inv @ (x, y) for x, y in _exterior(g)]
    c0, r0 = col - hw_pix, row - hw_pix
    xs = [p[0] - c0 - 0.5 for p in pts]
    ys = [p[1] - r0 - 0.5 for p in pts]
    img = stretch(chip.transpose(1, 2, 0) if chip.shape[0] == 3 else chip[0])
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(np.nan_to_num(img), cmap=None if img.ndim == 3 else "gray", interpolation="nearest")
    ax.plot(xs, ys, color="#ff2d2d", linewidth=1.2)
    n = img.shape[1]
    bar = SCALE_BAR_M / res
    ax.plot([n * 0.05, n * 0.05 + bar], [n * 0.94] * 2, color="white", lw=3, solid_capstyle="butt")
    ax.text(n * 0.05, n * 0.89, "200 m", color="white", fontsize=7)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(img.shape[0] - 0.5, -0.5)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _exterior(g: BaseGeometry):
    poly = g if g.geom_type == "Polygon" else max(g.geoms, key=lambda p: p.area)
    return list(poly.exterior.coords)


def _covers(path: str | Path, geom: BaseGeometry, geom_crs: str) -> bool:
    try:
        with rasterio.open(path) as src:
            c = _reproject(geom, geom_crs, src.crs).centroid
            b = src.bounds
            return b.left <= c.x <= b.right and b.bottom <= c.y <= b.top
    except Exception:  # noqa: BLE001 - unreadable raster simply does not cover
        return False


def historical_frames(registry: pd.DataFrame, geom: BaseGeometry, geom_crs: str) -> list[dict]:
    """Included registry frames whose raster covers the candidate, chronological."""
    rows = []
    for r in registry.itertuples(index=False):
        f = getattr(r, "georef_file", None)
        if f is None or (isinstance(f, float) and math.isnan(f)) or not Path(str(f)).exists():
            continue
        if _covers(f, geom, geom_crs):
            rows.append(
                {
                    "entity_id": str(r.entity_id),
                    "date": pd.to_datetime(r.acquisition_date).date(),
                    "rmse": float(getattr(r, "checkpoint_rmse_m", float("nan"))),
                    "file": str(f),
                }
            )
    return sorted(rows, key=lambda d: (d["date"], d["entity_id"]))


def _modern_layers(aoi: str, settings: Settings) -> list[tuple[str, Path, bool]]:
    from sbf.ingest.modern.process import read_manifest, stack_dir

    m = read_manifest(aoi, settings)
    if not m:
        return []
    out = []
    for name in MODERN_CHIP_LAYERS:
        ent = m["layers"].get(name)
        if ent:
            out.append((name, stack_dir(aoi, settings) / ent["file"], name.startswith("s2_")))
    return out


# ---- per-candidate assembly -----------------------------------------------------------------
def _fmt(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"


def candidate_confidence(
    row, frames: list[dict], prior: float | None, blind_test_passed: bool
) -> Confidence:
    rmses = [f["rmse"] for f in frames if not math.isnan(f["rmse"])]
    n_dates = (
        int(row["n_present"])
        if "n_present" in row and pd.notna(row["n_present"])
        else len({f["date"] for f in frames})
    )
    return assess_confidence(
        label=row["label"],
        mean_S=float(row["mean_S"]),
        families=row.get("families", ""),
        best_rmse_m=min(rmses) if rmses else None,
        n_hist_dates=n_dates,
        prior_contribution=prior,
        blind_test_passed=blind_test_passed,
    )


def has_fusion_columns(ranked) -> bool:
    """True when the candidates table carries the Phase 6 per-family and n_present columns."""
    cols = set(ranked.columns)
    return "n_present" in cols and "contrib_prior" in cols and "fam_prior" in cols


def _warn_fallback(ranked) -> None:
    if not has_fusion_columns(ranked):
        log.warning(
            "candidates table lacks fam_*/contrib_*/n_present columns (older fusion output): "
            "falling back to covering frame dates for the single-date cap and omitting "
            "per-family numbers; re-run `sbf fuse run`"
        )


def _num(v) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def _prior_of(cid: str, family_table: pd.DataFrame | None, row=None) -> float | None:
    if row is not None and "contrib_prior" in row and _num(row["contrib_prior"]) is not None:
        return _num(row["contrib_prior"])
    if family_table is None or cid not in family_table.index or "prior" not in family_table:
        return None
    v = family_table.loc[cid, "prior"]
    return None if pd.isna(v) else float(v)


def render_sheet(
    row,
    geom: BaseGeometry,
    crs: str,
    aoi: str,
    frames: list[dict],
    modern: list[tuple[str, Path, bool]],
    conf: Confidence,
    family_table: pd.DataFrame | None = None,
) -> str:
    chips = []
    for f in frames:
        png = render_chip(f["file"], geom, crs)
        if png:
            chips.append(
                {
                    "png": png,
                    "title": f"{f['date']} {f['entity_id']}",
                    "sub": f"checkpoint RMSE {_fmt(f['rmse'])} m",
                }
            )
    for name, path, rgb in modern:
        png = render_chip(path, geom, crs, rgb=rgb)
        if png:
            chips.append({"png": png, "title": name, "sub": "modern layer"})
    cid = row["cand_id"]
    if "fam_prior" in row and "contrib_prior" in row:
        # template has one value column: "mean <m>, contribution <c>"
        fam_rows = [
            (
                f,
                (
                    f"mean {_fmt(_num(row.get(f'fam_{f}')))}, "
                    f"contribution {_fmt(_num(row.get(f'contrib_{f}')))}"
                ),
            )
            for f in FAMILIES
        ]
    elif family_table is not None and cid in family_table.index:
        fam_rows = [(f, _fmt(v)) for f, v in family_table.loc[cid].items()]
    else:
        fam_rows = [(f.strip(), "contributing") for f in str(row["families"]).split(",") if f]
    prior = _prior_of(cid, family_table, row)
    c = geom.centroid
    ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(c.x, c.y)
    return _env.get_template("sheet.html.j2").render(
        cand_id=cid, rank=int(row["rank"]), aoi=aoi, label=row["label"], bracket=row["bracket"],
        mean_S=float(row["mean_S"]), area_m2=float(row["area_m2"]), confidence=conf,
        chips=chips, family_rows=fam_rows, prior="n/a" if prior is None else f"{prior:.2f}",
        statements=STATEMENTS, invisible_caveat=INVISIBLE_CAVEAT,
        label_noise_caveat=LABEL_NOISE_CAVEAT, lonlat=f"{ll[0]:.5f}, {ll[1]:.5f}",
        crs=crs, xy=f"{c.x:.1f}, {c.y:.1f}",
    )  # fmt: skip


def assess_all(
    ranked,
    registry: pd.DataFrame,
    crs: str,
    *,
    blind_test_passed: bool,
    family_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    """Confidence for every candidate -> (table cand_id/rank/label/mean_S/confidence, frames)."""
    rows, frames_by = [], {}
    _warn_fallback(ranked)
    for _, r in ranked.iterrows():
        fr = historical_frames(registry, r.geometry, crs)
        frames_by[r["cand_id"]] = fr
        conf = candidate_confidence(
            r, fr, _prior_of(r["cand_id"], family_table, r), blind_test_passed
        )
        rows.append(
            {
                "cand_id": r["cand_id"], "rank": int(r["rank"]), "label": r["label"],
                "mean_S": float(r["mean_S"]), "confidence": conf.level,
            }
        )  # fmt: skip
    cols = ["cand_id", "rank", "label", "mean_S", "confidence"]
    return pd.DataFrame(rows, columns=cols), frames_by


def write_sheets(
    ranked,
    registry: pd.DataFrame,
    aoi: str,
    crs: str,
    *,
    top_n: int = 20,
    blind_test_passed: bool = False,
    family_table: pd.DataFrame | None = None,
    settings: Settings | None = None,
) -> list[Path]:
    """Write top-N sheets, index and confidence table (restricted only); returns paths.

    D8: only counts are logged.
    """
    s = settings or load_settings()
    table, frames_by = assess_all(
        ranked, registry, crs, blind_test_passed=blind_test_passed, family_table=family_table
    )
    conf_by = dict(zip(table["cand_id"], table["confidence"], strict=True))
    modern = _modern_layers(aoi, s)
    paths: list[Path] = []
    top = ranked.sort_values("rank").head(top_n)
    for _, r in top.iterrows():
        cid = r["cand_id"]
        fr = frames_by[cid]
        conf = candidate_confidence(r, fr, _prior_of(cid, family_table, r), blind_test_passed)
        html = render_sheet(r, r.geometry, crs, aoi, fr, modern, conf, family_table)
        p = restricted_path(SHEET_DIR, f"{cid}.html", settings=s)
        p.write_text(html, encoding="utf-8")
        paths.append(p)
    idx_rows = [
        {
            **{k: r[k] for k in ("rank", "cand_id", "label", "mean_S")},
            "confidence": conf_by[r["cand_id"]],
        }
        for _, r in top.iterrows()
    ]
    ip = restricted_path(SHEET_DIR, "index.html", settings=s)
    ip.write_text(
        _env.get_template("index.html.j2").render(aoi=aoi, rows=idx_rows, statements=STATEMENTS),
        encoding="utf-8",
    )
    paths.append(ip)
    cp = restricted_path(CONF_CSV.format(aoi=aoi), settings=s)
    table.to_csv(cp, index=False)
    paths.append(cp)
    out = [encrypt_if_key(p) for p in paths]
    log.info("sheets %s: %d sheets written", aoi, len(top))
    return out


def key_warning() -> str | None:
    return None if os.environ.get("RESTRICTED_KEY") else UNENCRYPTED_WARNING


# ---- public aggregate stats -----------------------------------------------------------------
def _band(v: float) -> str:
    for lo, hi in SCORE_BANDS:
        if lo <= v < hi:
            return f"{lo:.1f}-{min(hi, 1.0):.1f}"
    return "out-of-range"


def public_stats_md(tables: Mapping[str, pd.DataFrame]) -> str:
    """Counts per AOI x confidence x label x score band; nothing else."""
    lines = [
        "# Candidate statistics (aggregate counts only)",
        "",
        STATEMENTS[0],
        STATEMENTS[1],
        "Locations and identifiers are restricted.",
        "",
        "| AOI | confidence | label | score band | count |",
        "|---|---|---|---|---|",
    ]
    for aoi in sorted(tables):
        t = tables[aoi]
        if t.empty:
            continue
        g = (
            t.assign(band=t["mean_S"].map(_band))
            .groupby(["confidence", "label", "band"], sort=True)
            .size()
        )
        for (conf, label, band), n in g.items():
            lines.append(f"| {aoi} | {conf} | {label} | {band} | {int(n)} |")
    unknown = {lab for t in tables.values() for lab in t.get("label", [])} - set(LABEL_MULTIPLIER)
    if unknown:
        lines.append("")
        lines.append(f"Note: {len(unknown)} unrecognised label(s).")
    return "\n".join(lines) + "\n"


def write_public_stats(settings: Settings | None = None) -> Path | None:
    """Build outputs/public/candidates_public_stats.md from restricted confidence tables."""
    s = settings or load_settings()
    base = restricted_path(settings=s)
    tables = {
        f.stem.removeprefix("candidate_confidence_"): pd.read_csv(f)
        for f in sorted(base.glob("candidate_confidence_*.csv"))
    }
    if not tables:
        return None
    s.outputs_public.mkdir(parents=True, exist_ok=True)
    out = s.outputs_public / "candidates_public_stats.md"
    out.write_text(public_stats_md(tables))
    return out


__all__ = [
    "key_warning", "public_stats_md", "render_chip", "render_sheet", "write_public_stats",
    "write_sheets",
]  # fmt: skip
