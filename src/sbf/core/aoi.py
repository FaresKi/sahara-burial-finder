"""AOI and reference-feature loaders."""

from __future__ import annotations

import json

import geopandas as gpd
from shapely.geometry import shape

from sbf.core.config import Settings, load_settings

PRECISIONS = {"iaea", "approximate", "to_locate"}
_REF_COLS = ["feature_id", "name", "aoi_id", "precision", "expected_area_km2", "notes", "geometry"]


def load_aois(settings: Settings | None = None) -> gpd.GeoDataFrame:
    """AOI polygons, columns `aoi_id`, `name`, `geometry`, EPSG:4326."""
    s = settings or load_settings()
    gdf = gpd.read_file(s.config_dir / "aois.geojson")
    gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
    return gdf[["aoi_id", "name", "geometry"]]


REFINEMENTS_FILE = "reference_refinements.geojson"


def _read_refinements(s: Settings) -> list[dict]:
    """Refinement features (raw GeoJSON dicts), file order = chronological append order."""
    path = s.config_dir / REFINEMENTS_FILE
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("features", [])


def load_reference_features(
    settings: Settings | None = None, refined: bool = True
) -> gpd.GeoDataFrame:
    """Reference features (EPSG:4326). BASE has null geometry and precision `to_locate`.

    Points are NOT buffered here; the scoring harness buffers by `expected_area_km2` (polygons
    are used as is). `config/reference_features.geojson` is never modified; with `refined=True`
    the latest record per feature_id in `config/reference_refinements.geojson` replaces the
    geometry. Extra columns: `refined` (bool), `original_geometry` (WKT or None),
    `refinement_evidence` (raster path or None). `refined=False` returns the originals only
    (same columns, `refined` all False).
    """
    s = settings or load_settings()
    gdf = gpd.read_file(s.config_dir / "reference_features.geojson")
    gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
    bad = set(gdf["precision"]) - PRECISIONS
    if bad:
        raise ValueError(f"invalid precision values: {bad}")
    gdf = gdf[_REF_COLS].copy()
    gdf["refined"] = False
    gdf["original_geometry"] = None
    gdf["refinement_evidence"] = None
    if not refined:
        return gdf
    latest: dict[str, dict] = {}
    for feat in _read_refinements(s):
        latest[feat["properties"]["feature_id"]] = feat  # later records win
    for fid, feat in latest.items():
        idx = gdf.index[gdf["feature_id"] == fid]
        if len(idx) == 0:
            raise ValueError(f"refinement for unknown feature_id {fid!r}")
        i = idx[0]
        orig = gdf.at[i, "geometry"]
        gdf.at[i, "original_geometry"] = None if orig is None or orig.is_empty else orig.wkt
        gdf.at[i, "geometry"] = shape(feat["geometry"])
        gdf.at[i, "refined"] = True
        ev = feat["properties"].get("evidence") or {}
        gdf.at[i, "refinement_evidence"] = ev.get("raster_path")
    return gdf
