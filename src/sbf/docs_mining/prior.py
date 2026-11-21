"""Spatial prior from accepted document claims (SPEC Phase 5, ticket 15).

Model (all constants below, in one place)
-----------------------------------------
Per accepted claim with a spatially resolved anchor (metric CRS, pixel centres of the modern
10 m grid):

* distance + bearing: annular sector = radial Gaussian around ``distance_km`` (sigma =
  ``distance_uncertainty_km``, default ``DEFAULT_DISTANCE_SIGMA_FRAC`` x distance) times a
  von-Mises angular falloff around the bearing (kappa = 1/sigma_rad^2, i.e. Gaussian-like;
  sigma = ``bearing_uncertainty_deg``, default ``DEFAULT_BEARING_SIGMA_DEG``).
* distance only: full annulus (radial Gaussian only).
* bearing only or neither: isotropic Gaussian around the anchor with sigma
  ``DEFAULT_ISOTROPIC_SIGMA_KM`` (a bearing without a distance has no usable range, so it is
  not turned into a ray).
* Each claim surface is normalised to unit mass over its support window (anchor +/- 5 sigma;
  mass beyond the AOI grid is renormalised away), weighted by ``SOURCE_WEIGHTS``, summed,
  divided by the maximum, then ``prior = EPS + (1 - EPS) * sum/max`` so the prior lies in
  [EPS, 1] and never zeroes out detector scores. Anchor position uncertainty is ignored.

``describes_reference_feature`` claims are still part of the PRODUCTION prior; the evaluation
(``prior_eval``) excludes them. Output rasters/tables live in git-ignored ``data/`` and are never
printed with coordinates.
"""

from __future__ import annotations

import csv
import logging
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.transform import Affine

from sbf.core.config import Settings, load_settings
from sbf.core.crs import lonlat_to_xy
from sbf.core.raster import write_cog
from sbf.detect.scores import write_scores
from sbf.docs_mining.gazetteer import Anchor, Gazetteer, load_gazetteer
from sbf.docs_mining.pdf import docs_interim_dir
from sbf.docs_mining.review import parse_decision
from sbf.features.tiling import make_grid
from sbf.ingest.modern.grid import grid_bounds, modern_grid

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- constants
SOURCE_WEIGHTS = {"official": 1.0, "testimony": 0.7, "secondary": 0.4}
DEFAULT_DISTANCE_SIGMA_FRAC = 0.5  # sigma = 50 % of distance when uncertainty is null
DEFAULT_BEARING_SIGMA_DEG = 45.0
DEFAULT_ISOTROPIC_SIGMA_KM = 2.0
EPS = 0.01  # uniform floor of the final prior
SUPPORT_SIGMAS = 5.0  # support window half-width in sigmas
PRIOR_DETECTOR = "prior"
PRIOR_LAYER = "docs_prior"

# ----------------------------------------------------------------------------- compass
_POINTS = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5, "E": 90.0, "ESE": 112.5, "SE": 135.0,
    "SSE": 157.5, "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5, "W": 270.0,
    "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}  # fmt: skip
_TOKEN = r"ouest|west|north|south|east|nord|sud|est|n|s|e|w|o"
_TOKEN_LETTER = {"ouest": "W", "west": "W", "north": "N", "nord": "N", "south": "S", "sud": "S",
                 "east": "E", "est": "E", "n": "N", "s": "S", "e": "E", "w": "W", "o": "W"}  # fmt: skip
_FILLER = re.compile(
    r"\b(?:to|of|the|toward|towards|direction|dir|vers|au|a|l|du|de|la|plein|due)\b"
)


def parse_bearing(text: str | None) -> float | None:
    """Azimuth in degrees clockwise from north (0-360) or None if not parseable.

    Accepts the 16-point compass in English and French (N, NNE, NE, ..., "north-east",
    "nord-est", "sud-ouest", "O"/"ONO" for west) and plain numbers ("45", "45 deg").
    """
    if not text or not text.strip():
        return None
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c)).casefold()
    m = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*(?:°|deg(?:rees?|res)?)?\s*", t)
    if m:
        return float(m.group(1).replace(",", ".")) % 360.0
    t = _FILLER.sub(" ", re.sub(r"[-_/'.\s]+", " ", t))
    t = re.sub(r"\s+", "", t)
    if not t or not re.fullmatch(rf"(?:{_TOKEN})+", t):
        return None
    letters = "".join(_TOKEN_LETTER[x] for x in re.findall(_TOKEN, t))
    return _POINTS.get(letters)


# ----------------------------------------------------------------------------- claims
@dataclass(frozen=True)
class Claim:
    """Reviewed claim without its quote (quotes never leave data/interim)."""

    extraction_id: str
    anchor: str
    distance_km: float | None
    distance_uncertainty_km: float | None
    bearing: str | None
    bearing_uncertainty_deg: float | None
    confidence_source: str
    describes_reference_feature: bool


def _f(v: str | None) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def review_csv_path(settings: Settings | None = None) -> Path:
    return docs_interim_dir(settings or load_settings()) / "review.csv"


def load_accepted_claims(settings: Settings | None = None, path: Path | None = None) -> list[Claim]:
    """Rows of the reviewed CSV with ``accept`` = y (human gate: nothing accepted, nothing used)."""
    p = path or review_csv_path(settings)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `sbf docs export-review` and review it")
    out = []
    with p.open(newline="") as f:
        for r in csv.DictReader(f):
            if parse_decision(r.get("accept", "")) is not True:
                continue
            out.append(
                Claim(
                    extraction_id=r["extraction_id"],
                    anchor=r.get("anchor", ""),
                    distance_km=_f(r.get("distance_km")),
                    distance_uncertainty_km=_f(r.get("distance_uncertainty_km")),
                    bearing=(r.get("bearing") or None),
                    bearing_uncertainty_deg=_f(r.get("bearing_uncertainty_deg")),
                    confidence_source=(r.get("confidence_source") or "secondary"),
                    describes_reference_feature=str(r.get("describes_reference_feature", ""))
                    .strip()
                    .lower()
                    == "true",
                )
            )
    return out


@dataclass(frozen=True)
class ResolvedClaim:
    claim: Claim
    anchor: Anchor | None
    status: str  # ok | unknown_anchor | anchor_needs_location | other_aoi
    azimuth_deg: float | None  # parsed bearing (None if absent or unparseable)

    @property
    def usable(self) -> bool:
        return self.status == "ok"


def resolve_claims(
    claims: list[Claim], aoi_id: str, gazetteer: Gazetteer | None = None,
    settings: Settings | None = None,
) -> list[ResolvedClaim]:  # fmt: skip
    gz = gazetteer or load_gazetteer(settings)
    out = []
    for c in claims:
        a = gz.resolve(c.anchor)
        if a is None:
            status = "unknown_anchor"
        elif not a.resolved:
            status = "anchor_needs_location"
        elif a.aoi_id != aoi_id:
            status = "other_aoi"
        else:
            status = "ok"
        out.append(ResolvedClaim(c, a, status, parse_bearing(c.bearing)))
    return out


# ----------------------------------------------------------------------------- surfaces
def claim_surface(
    xs: np.ndarray,
    ys: np.ndarray,
    anchor_xy: tuple[float, float],
    *,
    distance_km: float | None,
    distance_sigma_km: float | None = None,
    azimuth_deg: float | None = None,
    bearing_sigma_deg: float | None = None,
) -> tuple[slice, slice, np.ndarray] | None:
    """Unit-mass surface of one claim on the grid with pixel-centre coordinates ``xs`` (west to
    east) and ``ys`` (north to south), metres. Returns ``(row_slice, col_slice, window)`` with
    ``window.sum() == 1`` or None if the support does not touch the grid.

    Bearing is used only together with a distance (see module docstring).
    """
    sector = distance_km is not None and distance_km > 0
    res = float(abs(xs[1] - xs[0])) if len(xs) > 1 else 1.0
    if sector:
        d_m = distance_km * 1000.0
        s_m = (
            (distance_sigma_km * 1000.0) if distance_sigma_km else DEFAULT_DISTANCE_SIGMA_FRAC * d_m
        )
        s_m = max(s_m, res)
        reach = d_m + SUPPORT_SIGMAS * s_m
    else:
        s_m = max(DEFAULT_ISOTROPIC_SIGMA_KM * 1000.0, res)
        d_m = 0.0
        reach = SUPPORT_SIGMAS * s_m
    ax, ay = anchor_xy
    cols = np.flatnonzero((xs >= ax - reach) & (xs <= ax + reach))
    rows = np.flatnonzero((ys >= ay - reach) & (ys <= ay + reach))
    if cols.size == 0 or rows.size == 0:
        return None
    rs, cs = slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
    dx = (xs[cs] - ax)[None, :]
    dy = (ys[rs] - ay)[:, None]
    r = np.hypot(dx, dy)
    if sector:
        w = np.exp(-0.5 * ((r - d_m) / s_m) ** 2)
        if azimuth_deg is not None:
            sig = math.radians(bearing_sigma_deg or DEFAULT_BEARING_SIGMA_DEG)
            theta = np.arctan2(dx, dy)  # clockwise from north
            w = w * np.exp((np.cos(theta - math.radians(azimuth_deg)) - 1.0) / sig**2)
    else:
        w = np.exp(-0.5 * (r / s_m) ** 2)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        return None
    return rs, cs, (w / total).astype("float64")


def resolved_claim_surface(
    rc: ResolvedClaim, xs: np.ndarray, ys: np.ndarray, crs: str
) -> tuple[slice, slice, np.ndarray] | None:
    """`claim_surface` for a usable resolved claim (anchor lon/lat projected to ``crs``)."""
    a, c = rc.anchor, rc.claim
    axy = lonlat_to_xy(a.geometry.x, a.geometry.y, crs)
    return claim_surface(
        xs, ys, axy,
        distance_km=c.distance_km,
        distance_sigma_km=c.distance_uncertainty_km,
        azimuth_deg=rc.azimuth_deg,
        bearing_sigma_deg=c.bearing_uncertainty_deg,
    )  # fmt: skip


def grid_axes(grid) -> tuple[np.ndarray, np.ndarray]:
    t, w, h, _ = grid
    return t.c + (np.arange(w) + 0.5) * t.a, t.f + (np.arange(h) + 0.5) * t.e


def build_prior(
    claims: list[ResolvedClaim], grid, eps: float = EPS
) -> tuple[np.ndarray, list[str], list[str]]:
    """Prior array (float32, in [eps, 1]) on ``grid``; returns ``(prior, used_ids, off_grid_ids)``.

    Raises ValueError when no claim is usable (no information: refuse to emit a flat prior).
    """
    xs, ys = grid_axes(grid)
    crs = grid[3]
    acc = np.zeros((len(ys), len(xs)), dtype="float64")
    used, off = [], []
    for rc in claims:
        if not rc.usable:
            continue
        surf = resolved_claim_surface(rc, xs, ys, crs)
        if surf is None:
            off.append(rc.claim.extraction_id)
            continue
        rs, cs, w = surf
        acc[rs, cs] += (
            SOURCE_WEIGHTS.get(rc.claim.confidence_source, SOURCE_WEIGHTS["secondary"]) * w
        )
        used.append(rc.claim.extraction_id)
    if not used:
        raise ValueError("no usable claims for the prior (unresolved anchors or off-grid)")
    prior = eps + (1.0 - eps) * acc / acc.max()
    return prior.astype("float32"), used, off


# ----------------------------------------------------------------------------- tile scores
def tile_mean_prior(prior: np.ndarray, transform: Affine, bounds: np.ndarray, eps: float = EPS):
    """Mean prior per tile; ``bounds`` is an (n, 4) array (minx, miny, maxx, maxy) in grid CRS.

    Pixel windows are rounded to the nearest pixel like ``tile_windows``; windows are clipped to
    the raster (mean over the covered part) and fully outside tiles get ``eps``.
    """
    h, w = prior.shape
    integ = np.zeros((h + 1, w + 1), dtype="float64")
    integ[1:, 1:] = prior.astype("float64").cumsum(0).cumsum(1)
    px, py = transform.a, -transform.e
    n_c = np.rint((bounds[:, 2] - bounds[:, 0]) / px).astype(int)
    n_r = np.rint((bounds[:, 3] - bounds[:, 1]) / py).astype(int)
    c0 = np.rint((bounds[:, 0] - transform.c) / px).astype(int)
    r0 = np.rint((transform.f - bounds[:, 3]) / py).astype(int)
    c1, r1 = np.clip(c0 + n_c, 0, w), np.clip(r0 + n_r, 0, h)
    c0, r0 = np.clip(c0, 0, w), np.clip(r0, 0, h)
    area = (c1 - c0) * (r1 - r0)
    tot = integ[r1, c1] - integ[r0, c1] - integ[r1, c0] + integ[r0, c0]
    return np.where(area > 0, tot / np.maximum(area, 1), eps)


def tile_prior_table(
    prior: np.ndarray, grid, tile_ids: list[str], aoi_id: str, settings: Settings
) -> pd.DataFrame:
    """Score-contract frame (detector ``prior``, layer ``docs_prior``) for the given tile IDs."""
    from sbf.features.tiling import parse_tile_id, tile_geometries

    frames = []
    ids_arr = np.array(tile_ids)
    sizes = np.array([parse_tile_id(t)[1] for t in tile_ids])
    for size in sorted(set(sizes.tolist())):
        ids = ids_arr[sizes == size].tolist()
        g = tile_geometries(ids, settings.overlap, settings)
        sc = tile_mean_prior(prior, grid[0], g.geometry.bounds.to_numpy(), EPS)
        frames.append(
            pd.DataFrame(
                {
                    "tile_id": ids,
                    "aoi_id": aoi_id,
                    "size_m": int(size),
                    "detector": PRIOR_DETECTOR,
                    "layer": PRIOR_LAYER,
                    "score": sc,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ----------------------------------------------------------------------------- orchestration
@dataclass
class PriorResult:
    raster: Path
    score_tables: list[Path]
    n_accepted: int
    used_ids: list[str]
    unresolved: dict[str, list[str]]  # status -> extraction ids
    off_grid_ids: list[str]


def prior_raster_path(aoi_id: str, settings: Settings | None = None) -> Path:
    s = settings or load_settings()
    return s.data_processed / "prior" / f"prior_{aoi_id}.tif"


def build_prior_products(
    aoi_id: str, settings: Settings | None = None, min_accepted: int = 1,
    claims: list[Claim] | None = None, grid=None,
) -> PriorResult:  # fmt: skip
    """Build ``prior_{AOI}.tif`` (COG + provenance) and the per-tile ``prior`` score tables.

    Human gate: refuses when fewer than ``min_accepted`` accepted rows exist in review.csv.
    """
    s = settings or load_settings()
    claims = load_accepted_claims(s) if claims is None else claims
    if len(claims) < max(1, min_accepted):
        raise RuntimeError(
            f"{len(claims)} accepted claims (need >= {max(1, min_accepted)}): review "
            "data/interim/docs/review.csv first (human gate)"
        )
    grid = grid or modern_grid(aoi_id, s)
    resolved = resolve_claims(claims, aoi_id, settings=s)
    prior, used, off = build_prior(resolved, grid)
    t, _, _, crs = grid
    path = write_cog(
        prior_raster_path(aoi_id, s), prior, {"crs": crs, "transform": t},
        prov={
            "source": "docs_mining_prior",
            "source_id": f"prior_{aoi_id}",
            "acquisition_date": None,
            "steps": [
                ("annular-sector gaussian prior from accepted document claims (sbf.docs_mining.prior)"),
                (
                    f"weights={SOURCE_WEIGHTS} eps={EPS} default_distance_sigma_frac="
                    f"{DEFAULT_DISTANCE_SIGMA_FRAC} default_bearing_sigma_deg="
                    f"{DEFAULT_BEARING_SIGMA_DEG} default_isotropic_sigma_km="
                    f"{DEFAULT_ISOTROPIC_SIGMA_KM}"
                ),
                "extraction_ids=" + ",".join(sorted(used)),
            ],
            "inputs": [str(review_csv_path(s))],
        },
    )  # fmt: skip
    frames = []
    for size in s.tile_sizes_m:
        grid_gdf = make_grid(aoi_id, size, s.overlap, s, bounds=grid_bounds(grid))
        if len(grid_gdf) == 0:
            continue
        frames.append(tile_prior_table(prior, grid, grid_gdf["tile_id"].tolist(), aoi_id, s))
    # one table per (aoi, detector, layer) holding every tile size
    tables = write_scores(pd.concat(frames, ignore_index=True), s) if frames else []
    unresolved: dict[str, list[str]] = {}
    for rc in resolved:
        if not rc.usable:
            unresolved.setdefault(rc.status, []).append(rc.claim.extraction_id)
    return PriorResult(path, tables, len(claims), used, unresolved, off)
