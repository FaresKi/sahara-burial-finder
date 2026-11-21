"""Validated, append-only refinement records (SPEC 0.2: never invent coordinates)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import Point, box, mapping
from shapely.geometry.base import BaseGeometry

from sbf.core.aoi import REFINEMENTS_FILE, _read_refinements, load_aois, load_reference_features
from sbf.core.config import Settings, load_settings
from sbf.core.crs import transform_geometry

METHODS = {"manual_digitised", "imagery_centroid"}
REFINERS = {"human", "agent"}
DEFAULT_MAX_MOVE_M = 500.0


class RefinementError(ValueError):
    """A refinement was refused (missing evidence, out of bounds, unexplained move...)."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _original(feature_id: str, s: Settings):
    ref = load_reference_features(s, refined=False).set_index("feature_id")
    if feature_id not in ref.index:
        raise RefinementError(f"unknown feature_id {feature_id!r}; known: {sorted(ref.index)}")
    return ref.loc[feature_id]


def add_refinement(
    feature_id: str,
    geometry: BaseGeometry,
    *,
    evidence_raster: str | Path,
    screenshot: str | Path,
    method: str = "manual_digitised",
    refined_by: str = "human",
    layer: str | None = None,
    entity_id: str | None = None,
    notes: str = "",
    force_note: str | None = None,
    max_move_m: float = DEFAULT_MAX_MOVE_M,
    settings: Settings | None = None,
) -> dict:
    """Validate and append a refinement record; returns the written GeoJSON feature.

    `geometry` is EPSG:4326 (Point or Polygon). Refuses unless the evidence raster and screenshot
    exist and the geometry lies within the feature's AOI and the raster's bounds, and refuses to
    move an `iaea`-precision feature more than `max_move_m` without `force_note`.
    """
    s = settings or load_settings()
    if method not in METHODS:
        raise RefinementError(f"method must be one of {sorted(METHODS)}")
    if refined_by not in REFINERS:
        raise RefinementError(f"refined_by must be one of {sorted(REFINERS)}")
    if geometry is None or geometry.is_empty or geometry.geom_type not in ("Point", "Polygon"):
        raise RefinementError("geometry must be a non-empty Point or Polygon (EPSG:4326)")
    if not geometry.is_valid:
        raise RefinementError("geometry is not valid")
    raster, shot = Path(evidence_raster), Path(screenshot)
    if not raster.is_file():
        raise RefinementError(f"evidence raster not found: {raster}")
    if not shot.is_file():
        raise RefinementError(f"screenshot not found: {shot}")
    orig = _original(feature_id, s)

    aoi = load_aois(s).set_index("aoi_id")
    if orig["aoi_id"] not in aoi.index:
        raise RefinementError(f"AOI {orig['aoi_id']!r} not found")
    if not aoi.loc[orig["aoi_id"], "geometry"].covers(geometry):
        raise RefinementError(f"geometry is outside AOI {orig['aoi_id']}")
    with rasterio.open(raster) as src:
        if src.crs is None:
            raise RefinementError("evidence raster has no CRS")
        rb = box(*src.bounds)
        in_raster = transform_geometry(geometry, "EPSG:4326", src.crs.to_string())
    if not rb.covers(in_raster):
        raise RefinementError("geometry is outside the evidence raster bounds")

    og = orig["geometry"]
    if orig["precision"] == "iaea" and og is not None and not og.is_empty:
        crs = s.crs(orig["aoi_id"])
        d = transform_geometry(og, "EPSG:4326", crs).distance(
            transform_geometry(geometry, "EPSG:4326", crs).centroid
        )
        if d > max_move_m and not (force_note and force_note.strip()):
            raise RefinementError(
                f"move of {d:.0f} m exceeds {max_move_m:.0f} m for IAEA-precision {feature_id}; "
                "supply --force-note explaining why"
            )
        if d > max_move_m:
            notes = f"{notes} [force-note: {force_note.strip()}; moved {d:.0f} m]".strip()

    ev = {"raster_path": str(raster), "screenshot_path": str(shot), "sha256": sha256_file(raster)}
    if layer:
        ev["layer"] = layer
    if entity_id:
        ev["entity_id"] = entity_id
    feat = {
        "type": "Feature",
        "properties": {
            "feature_id": feature_id,
            "geometry_type": "point" if geometry.geom_type == "Point" else "polygon",
            "method": method,
            "evidence": ev,
            "refined_by": refined_by,
            "refined_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "notes": notes,
            "supersedes_precision": orig["precision"],
        },
        "geometry": mapping(geometry),
    }
    feats = _read_refinements(s) + [feat]
    path = s.config_dir / REFINEMENTS_FILE
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, indent=2) + "\n")
    return feat


def import_polygon(
    feature_id: str,
    file: str | Path,
    *,
    evidence_raster: str | Path,
    screenshot: str | Path,
    **kw,
) -> dict:
    """Import a human-digitised polygon (GeoJSON/GPKG, e.g. from QGIS); needs exactly one polygon."""
    gdf = gpd.read_file(file)
    if gdf.crs is None:
        raise RefinementError("imported file has no CRS")
    gdf = gdf.to_crs("EPSG:4326")
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if len(geoms) != 1 or geoms[0].geom_type != "Polygon":
        raise RefinementError("file must contain exactly one Polygon feature")
    kw.setdefault("method", "manual_digitised")
    return add_refinement(
        feature_id, geoms[0], evidence_raster=evidence_raster, screenshot=screenshot, **kw
    )


def list_refinements(settings: Settings | None = None) -> list[dict]:
    """One row per reference feature: id, precision, refined, evidence, geometry type, lon/lat."""
    s = settings or load_settings()
    rows = []
    for _, r in load_reference_features(s).iterrows():
        g = r["geometry"]
        has = g is not None and not g.is_empty
        c = g.centroid if has else Point()
        rows.append(
            {
                "feature_id": r["feature_id"],
                "precision": r["precision"],
                "refined": bool(r["refined"]),
                "geometry_type": g.geom_type if has else None,
                "lon": c.x if has else None,
                "lat": c.y if has else None,
                "evidence": r["refinement_evidence"],
            }
        )
    return rows
