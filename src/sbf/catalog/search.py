"""Phase 1 catalogue search: dataset discovery, scene search, coverage table (ticket 02)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from sbf.catalog.m2m import CatalogClient, M2MError
from sbf.catalog.resolution import match_system, resolution_class

START_DEFAULT = "1959-01-01"
END_DEFAULT = "1984-12-31"
PAGE_SIZE = 100
SOURCE = "USGS EarthExplorer M2M"

COVERAGE_COLUMNS = [
    "entity_id",
    "dataset",
    "mission",
    "camera",
    "camera_resolution",
    "acquisition_date",
    "cloud_cover",
    "availability",
    "browse_url",
    "footprint_wkt",
    "intersecting_features",
    "resolution_class",
    "aoi_id",
    "raw_file",
]

# ---------------------------------------------------------------------------------------
# M2M FIELD MAPPING. Single table. Scene fields checked against the first live run (2026-11-21:
# cloudCover is null for all Declass scenes; resolution comes from "Camera Resolution"). Originally
# available when written). Correct here after the first live run; nothing else hard-codes
# M2M field names. "path" entries are dotted paths into the scene-search result dict.
# ---------------------------------------------------------------------------------------
SCENE_FIELDS = {
    "entity_id": ["entityId"],
    "cloud_cover": ["cloudCover"],
    "browse_path": ["browse.0.browsePath", "browse.0.overlayPath"],
    "footprint": ["spatialCoverage", "spatialBounds"],  # GeoJSON geometry
    "acq_date": ["temporalCoverage.startDate", "acquisitionDate"],
    # Entries of result["metadata"] = [{"fieldName": ..., "value": ...}]; matched case-insensitively
    "mission_meta": ["Mission", "Mission Name", "Platform"],
    "camera_meta": ["Camera Type", "Camera", "Camera Mode"],
    # Verified live 2026-11-21: Declass 1 "Stereo High|Stereo Medium|Vertical Low", Declass 3
    # "2 to 4 feet". This is the field resolution_class keys on first.
    "resolution_meta": ["Camera Resolution"],
    "acq_date_meta": ["Acquisition Date", "Acquisition date"],
    # result["options"] flags: bulk/download/order/secondary
    "available_flag": "options.download",
    "order_flag": "options.order",
}
DATASET_FIELDS = {
    "name": ["datasetAlias", "datasetName"],  # value passed back as `datasetName` in scene-search
    "title": ["collectionLongName", "collectionName"],
    "abstract": ["abstractText"],
}
# Declass 1/2/3 recogniser, applied to name/title/abstract of dataset-search results.
# Matches titles ("Declass 2 = KH-7 ...") and aliases ("declassii"); verified live 2026-11-21:
# corona2 = "Declass 1", declassii = "Declass 2", declassiii = "Declass 3".
DECLASS_PATTERN = re.compile(
    r"declass(?:ified)?[\s:_-]*(?:imagery[\s:_-]*)?(iii|ii|i|1|2|3)\b", re.IGNORECASE
)
_SERIES = {"i": "1", "ii": "2", "iii": "3"}


def _dig(obj: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(obj, list):
            try:
                obj = obj[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
        if obj is None:
            return None
    return obj


def _first(obj: dict, paths: list[str]) -> Any:
    for p in paths:
        v = _dig(obj, p)
        if v not in (None, ""):
            return v
    return None


def _meta(result: dict, names: list[str]) -> str | None:
    wanted = [n.lower() for n in names]
    items = result.get("metadata") or []
    for want in wanted:
        for it in items:
            if str(it.get("fieldName", "")).strip().lower().rstrip(":") == want and it.get("value"):
                return str(it["value"]).strip()
    return None


# ------------------------------------------------------------------ failures


@dataclass
class Failure:
    endpoint: str
    context: str
    message: str
    raw_file: str | None = None


@dataclass
class SearchResult:
    datasets: list[dict] = field(default_factory=list)
    coverage: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=COVERAGE_COLUMNS))
    failures: list[Failure] = field(default_factory=list)


# ------------------------------------------------------------------ discovery


def discover_datasets(
    client: CatalogClient,
    keywords: tuple[str, ...] = ("declass",),
    failures: list[Failure] | None = None,
) -> list[dict]:
    """dataset-search per keyword; keep datasets whose name/title/abstract match Declass 1/2/3."""
    kept: dict[str, dict] = {}
    for kw in keywords:
        try:
            env = client.call("dataset-search", {"datasetName": kw})
        except M2MError as exc:
            if failures is not None:
                failures.append(Failure("dataset-search", kw, str(exc), client.last_raw_file))
                continue
            raise
        raw = client.last_raw_file
        for d in env.get("data") or []:
            name = _first(d, DATASET_FIELDS["name"])
            if not name:
                continue
            text = " ".join(
                str(_first(d, [k]) or "")
                for k in DATASET_FIELDS["name"]
                + DATASET_FIELDS["title"]
                + DATASET_FIELDS["abstract"]
            )
            m = DECLASS_PATTERN.search(text)
            if m and name not in kept:
                kept[name] = {
                    "dataset": name,
                    "title": _first(d, DATASET_FIELDS["title"]),
                    "declass_series": _SERIES.get(m.group(1).lower(), m.group(1)),
                    "raw_file": raw,
                }
    return sorted(kept.values(), key=lambda x: x["dataset"])


# ------------------------------------------------------------------ scenes


def _parse_date(value: Any) -> str | None:
    """ISO date (YYYY-MM-DD) from the leading 10 chars, or None if it doesn't parse."""
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10]).isoformat()
    except ValueError:
        return None


def _cloud(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def scene_to_row(result: dict, *, dataset: str, aoi_id: str, raw_file: str | None) -> dict:
    """Map one scene-search result to a coverage row (field names via SCENE_FIELDS)."""
    fp = _first(result, SCENE_FIELDS["footprint"])
    footprint_wkt = ""
    if isinstance(fp, dict):
        try:
            footprint_wkt = shape(fp).wkt
        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ):  # malformed geometry: leave empty, row stays traceable
            footprint_wkt = ""
    mission = _meta(result, SCENE_FIELDS["mission_meta"])
    camera = _meta(result, SCENE_FIELDS["camera_meta"])
    camera_resolution = _meta(result, SCENE_FIELDS["resolution_meta"])
    if mission is None:
        mission = match_system(camera, dataset)
    acq = _parse_date(
        _first(result, SCENE_FIELDS["acq_date"]) or _meta(result, SCENE_FIELDS["acq_date_meta"])
    )
    if _dig(result, SCENE_FIELDS["available_flag"]):
        availability = "scanned"
    elif _dig(result, SCENE_FIELDS["order_flag"]):
        availability = "needs_order"
    else:
        availability = "unknown"
    return {
        "entity_id": _first(result, SCENE_FIELDS["entity_id"]),
        "dataset": dataset,
        "mission": mission,
        "camera": camera,
        "camera_resolution": camera_resolution,
        "acquisition_date": acq,
        "cloud_cover": _cloud(_first(result, SCENE_FIELDS["cloud_cover"])),
        "availability": availability,
        "browse_url": _first(result, SCENE_FIELDS["browse_path"]),
        "footprint_wkt": footprint_wkt,
        "intersecting_features": "",
        "resolution_class": resolution_class(camera_resolution, camera, mission, dataset),
        "aoi_id": aoi_id,
        "raw_file": raw_file,
    }


def search_scenes(
    client: CatalogClient,
    dataset: str,
    aoi_id: str,
    aoi_geom: BaseGeometry,
    start: str,
    end: str,
    failures: list[Failure],
) -> list[dict]:
    """Paginated scene-search for one dataset x AOI. Failures are recorded, not raised."""
    rows: list[dict] = []
    start_no = 1
    while True:
        payload = {
            "datasetName": dataset,
            "maxResults": PAGE_SIZE,
            "startingNumber": start_no,
            "metadataType": "full",
            "sceneFilter": {
                "spatialFilter": {"filterType": "geojson", "geoJson": mapping(aoi_geom)},
                "acquisitionFilter": {"start": start, "end": end},
            },
        }
        try:
            env = client.call("scene-search", payload)
        except M2MError as exc:
            failures.append(
                Failure(
                    "scene-search", f"{dataset}/{aoi_id}@{start_no}", str(exc), client.last_raw_file
                )
            )
            break
        raw = client.last_raw_file
        data = env.get("data") or {}
        results = data.get("results") or []
        rows += [scene_to_row(r, dataset=dataset, aoi_id=aoi_id, raw_file=raw) for r in results]
        nxt = data.get("nextRecord")
        if not results or not nxt or int(nxt) <= start_no:
            break
        start_no = int(nxt)
    return rows


def tag_intersections(coverage: pd.DataFrame, reference: gpd.GeoDataFrame) -> pd.DataFrame:
    """Fill `intersecting_features` (';'-joined IDs of reference features inside the footprint).

    Reference features with null geometry (BASE) are skipped. Points are tested as points.
    """
    out = coverage.copy()
    refs = [(r.feature_id, r.geometry) for r in reference.itertuples() if r.geometry is not None]
    tags = []
    for w in out["footprint_wkt"]:
        if not isinstance(w, str) or not w:
            tags.append("")
            continue
        fp = wkt.loads(w)
        tags.append(";".join(fid for fid, g in refs if fp.intersects(g)))
    out["intersecting_features"] = tags
    return out


def run_search(
    client: CatalogClient,
    aois: gpd.GeoDataFrame,
    reference: gpd.GeoDataFrame,
    aoi_ids: list[str],
    start: str = START_DEFAULT,
    end: str = END_DEFAULT,
) -> SearchResult:
    res = SearchResult()
    res.datasets = discover_datasets(client, failures=res.failures)
    rows: list[dict] = []
    for aoi_id in aoi_ids:
        geom = aois.loc[aois.aoi_id == aoi_id, "geometry"].iloc[0]
        for d in res.datasets:
            rows += search_scenes(client, d["dataset"], aoi_id, geom, start, end, res.failures)
    df = pd.DataFrame(rows, columns=COVERAGE_COLUMNS)
    if not df.empty:
        df = df.drop_duplicates(subset=["entity_id", "dataset"], keep="first").reset_index(
            drop=True
        )
    res.coverage = tag_intersections(df, reference)
    return res


def write_search_outputs(res: SearchResult, *, coverage_csv: Path, interim_dir: Path) -> None:
    coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    res.coverage.to_csv(coverage_csv, index=False)
    interim_dir.mkdir(parents=True, exist_ok=True)
    (interim_dir / "datasets.json").write_text(json.dumps(res.datasets, indent=2))
    (interim_dir / "failures.json").write_text(
        json.dumps([f.__dict__ for f in res.failures], indent=2)
    )
