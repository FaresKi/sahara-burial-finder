"""Draft builder for ``config/exclusions.geojson`` (infrastructure/settlement mask).

HUMAN GATE (SPEC section 8): every item is written with ``status = draft_needs_human_review``
and ``approximate = true``. The mask is a reviewable stub: a human must check each polygon
against imagery and set ``status`` to ``reviewed`` before results that depend on it are
reported as final. ``write_draft`` refuses to overwrite a file containing reviewed items.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from sbf.core.config import Settings, load_settings

log = logging.getLogger(__name__)
DRAFT = "draft_needs_human_review"
REVIEWED = "reviewed"


@dataclass(frozen=True)
class ExclusionItem:
    """A manually specified infrastructure bbox (lon_min, lat_min, lon_max, lat_max, WGS84)."""

    name: str
    aoi_id: str
    kind: str
    bbox: tuple[float, float, float, float]
    source: str
    approximate: bool = True


# Public, approximate anchors only; to be replaced by polygons digitised from imagery.
DEFAULT_ITEMS: tuple[ExclusionItem, ...] = (
    ExclusionItem(
        name="Reggane oasis / town (rough bbox)",
        aoi_id="REG",
        kind="settlement",
        bbox=(0.12, 26.68, 0.25, 26.75),
        source="approximate box around Reggane town (about 26.72 N, 0.17 E) clipped to the AOI; "
        "NOT digitised from imagery",
    ),
)


def build_draft_exclusions(
    items: tuple[ExclusionItem, ...] | list[ExclusionItem] | None = None,
) -> gpd.GeoDataFrame:
    """GeoDataFrame (EPSG:4326) of draft exclusions; all rows carry the draft status."""
    items = DEFAULT_ITEMS if items is None else items
    return gpd.GeoDataFrame(
        {
            "name": [i.name for i in items],
            "aoi_id": [i.aoi_id for i in items],
            "kind": [i.kind for i in items],
            "source": [i.source for i in items],
            "approximate": [i.approximate for i in items],
            "status": DRAFT,
        },
        geometry=[box(*i.bbox) for i in items],
        crs="EPSG:4326",
    )


def exclusions_path(settings: Settings | None = None) -> Path:
    return (settings or load_settings()).config_dir / "exclusions.geojson"


def write_draft(
    gdf: gpd.GeoDataFrame | None = None, settings: Settings | None = None, overwrite: bool = False
) -> Path:
    """Write the draft to ``config/exclusions.geojson``; never clobbers human-reviewed content."""
    path = exclusions_path(settings)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path.name} exists; pass overwrite=True to replace the draft")
    if path.exists() and REVIEWED in set(load_exclusions(settings)["status"]):
        raise FileExistsError(f"{path.name} contains reviewed items; refusing to overwrite")
    gdf = build_draft_exclusions() if gdf is None else gdf
    path.write_text(json.dumps(json.loads(gdf.to_json()), indent=2) + "\n")
    log.info("wrote %d draft exclusions", len(gdf))
    return path


def load_exclusions(settings: Settings | None = None) -> gpd.GeoDataFrame | None:
    """Exclusions (EPSG:4326) or None if the file does not exist."""
    path = exclusions_path(settings)
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    return gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")


def review_status(gdf: gpd.GeoDataFrame | None) -> str:
    """``none`` | ``draft`` (any item unreviewed) | ``reviewed``."""
    if gdf is None or len(gdf) == 0:
        return "none"
    if "status" not in gdf.columns:
        return "draft"
    return "reviewed" if set(gdf["status"]) == {REVIEWED} else "draft"
