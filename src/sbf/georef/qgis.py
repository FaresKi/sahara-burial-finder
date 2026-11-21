"""QGIS Georeferencer fallback: export seed GCPs as a `.points` file, read the human's back.

File format (QGIS 3 Georeferencer "Save GCP points"): optional first line `#CRS: <crs>`, header
`mapX,mapY,sourceX,sourceY,enable,dX,dY,residual`, then one row per point. QGIS stores the source
pixel row NEGATED (sourceY = -row, y axis up, origin at the image top-left corner); this module
exports `sourceY = -py` and on import negates back (when every sourceY is <= 0; a file with all
positive sourceY is accepted as positive-down rows and noted). Pixel convention is the same
"corner" convention used everywhere in `sbf.georef` (QGIS' pixel edge = integer). Unverified
against a real QGIS install: confirm the row-sign and half-pixel convention on the first manual
session (the `qgis_import` round trip is covered by a test of this module's own convention).

A minimal `.qgs` project was deliberately skipped: the Georeferencer keeps its state outside the
project file, so the README (open frame + reference layer, load the points, save) is enough.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError

from sbf.georef.models import UTM_CRS

HEADER = "mapX,mapY,sourceX,sourceY,enable,dX,dY,residual"


def write_points(gcps: pd.DataFrame, path: Path, crs: str = UTM_CRS) -> Path:
    """GCP table (`px, py, x, y`, optional `residual_m`) -> QGIS `.points` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"#CRS: {CRS.from_user_input(crs).to_wkt()}".replace("\n", " "), HEADER]
    for r in gcps.itertuples():
        res = getattr(r, "residual_m", 0.0)
        lines.append(
            f"{r.x:.4f},{r.y:.4f},{r.px:.4f},{-r.py:.4f},1,0,0,{0.0 if pd.isna(res) else res:.4f}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


@dataclass
class PointsFile:
    gcps: pd.DataFrame  # px, py (rows positive down), x, y (UTM)
    crs: str | None
    notes: list[str]


def read_points(path: Path, target_crs: str = UTM_CRS) -> PointsFile:
    text = Path(path).read_text().splitlines()
    crs = None
    body: list[str] = []
    for ln in text:
        if ln.startswith("#"):
            if ln.upper().startswith("#CRS:"):
                crs = ln.split(":", 1)[1].strip() or None
            continue
        if ln.strip():
            body.append(ln)
    if not body or not body[0].lower().startswith("mapx"):
        raise ValueError(f"{path}: not a QGIS .points file (missing '{HEADER}' header)")
    df = pd.read_csv(io.StringIO("\n".join(body)))
    df = df.rename(columns=str.strip)
    need = {"mapX", "mapY", "sourceX", "sourceY"}
    if not need <= set(df.columns):
        raise ValueError(f"{path}: missing columns {sorted(need - set(df.columns))}")
    notes: list[str] = []
    if "enable" in df.columns:
        off = int((df["enable"] == 0).sum())
        if off:
            notes.append(f"{off} disabled points skipped")
        df = df[df["enable"] != 0]
    sy = df["sourceY"].to_numpy(float)
    if (sy <= 0).all():
        py = -sy
    elif (sy >= 0).all():
        py = sy
        notes.append("sourceY positive: taken as positive-down rows")
    else:
        raise ValueError(f"{path}: mixed-sign sourceY; cannot tell the row convention")
    x, y = df["mapX"].to_numpy(float), df["mapY"].to_numpy(float)
    if crs:
        try:
            src = CRS.from_user_input(crs)
            if src != CRS.from_user_input(target_crs):
                x, y = Transformer.from_crs(src, target_crs, always_xy=True).transform(x, y)
                notes.append(f"map coordinates reprojected to {target_crs}")
        except (CRSError, ValueError):  # unparseable CRS line: assume the target CRS
            notes.append("unreadable #CRS line; assumed " + target_crs)
    out = pd.DataFrame({"px": df["sourceX"].to_numpy(float), "py": py, "x": x, "y": y})
    return PointsFile(out.reset_index(drop=True), crs, notes)


README = """# Manual georeferencing in QGIS: {entity_id}

1. Open QGIS (3.x). Layer > Add Layer > Add Raster Layer: load the reference layer
   `{reference}` (modern Sentinel-2 composite, EPSG:32631). Use it, never web tiles with unclear licences.
2. Project > New. Set the project CRS to EPSG:32631.
3. Layer > Georeferencer. File > Open Raster: `{frame}` (the un-georeferenced frame).
4. Settings > Transformation Settings: Polynomial 2 (or Thin Plate Spline with many points),
   target CRS EPSG:32631. The fit is redone by `sbf`; QGIS is only used to collect points.
5. File > Load GCP Points: `{points}` (automatic seed points, {n_seed} of them; delete the bad ones
   and add yours). Add at least 12-20 well spread points on stable, unambiguous features (bedrock
   corners, wadi junctions, road intersections). Avoid dunes. Pick them with "From map canvas".
6. File > Save GCP Points As: `{points_out}` (keep the `.points` extension).
7. Back in the terminal: `sbf georef qgis-import --entity-id {entity_id} --points {points_out}`
   (20% of your points are held out as checkpoints; the report method is `manual_gcps`).

Notes: QGIS stores source rows negated (sourceY = -row); `sbf` handles this. Do not edit the
`sourceX/sourceY` columns by hand.
"""


def write_readme(
    path: Path, *, entity_id: str, reference: str, frame: str, points: Path, n_seed: int
):
    out = points.with_name(f"{entity_id}_manual.points")
    path.write_text(
        README.format(
            entity_id=entity_id,
            reference=reference,
            frame=frame,
            points=points,
            points_out=out,
            n_seed=n_seed,
        )
    )
    return path


# --------------------------------------------------------------------------- CLI-level helpers


def _reference_hint(entity_id: str, settings, aoi: str | None) -> str:
    try:
        import json

        from sbf.georef.pipeline import coverage_row
        from sbf.ingest.modern.process import stack_dir

        row = coverage_row(entity_id, settings, aoi)
        aoi_id = aoi or str(row.get("aoi_id") or "")
        sd = stack_dir(aoi_id, settings)
        layers = json.loads((sd / "stack.json").read_text())["layers"]
        return str(sd / layers["s2_median"]["file"])
    except (OSError, KeyError, ValueError):
        return (
            "data/processed/modern/<AOI>/s2_median.tif (build with `sbf ingest modern build-stack`)"
        )


def export_qgis(entity_id: str, *, settings=None, aoi: str | None = None, out_dir=None) -> dict:
    """Write `<processed>/<id>/qgis/<id>_seed.points` (seeded with gcps.csv when present) and a
    README with the manual steps. Returns the paths."""
    from sbf.core.config import load_settings
    from sbf.georef.pipeline import find_raw_frame, processed_dir

    s = settings or load_settings()
    base = processed_dir(entity_id, s)
    out = Path(out_dir) if out_dir else base / "qgis"
    frame = find_raw_frame(entity_id, s, base)
    csv = base / "gcps.csv"
    seed = pd.read_csv(csv) if csv.exists() else pd.DataFrame(columns=["px", "py", "x", "y"])
    points = write_points(seed, out / f"{entity_id}_seed.points")
    readme = write_readme(
        out / "README.md",
        entity_id=entity_id,
        reference=_reference_hint(entity_id, s, aoi),
        frame=str(frame),
        points=points,
        n_seed=len(seed),
    )
    return {"points": points, "readme": readme, "frame": frame, "n_seed": len(seed)}


def import_qgis(entity_id: str, points: Path, *, settings=None, **run_kw):
    """Read a QGIS `.points` file and rerun fit + warp + report (`gcp_source = manual_gcps`)."""
    from sbf.georef.pipeline import run_frame

    pf = read_points(points)
    if len(pf.gcps) == 0:
        raise ValueError(f"{points}: no enabled points")
    return run_frame(
        entity_id,
        settings=settings,
        manual_gcps=pf.gcps,
        extra_warnings=[f"qgis_import: {n}" for n in pf.notes],
        **run_kw,
    )
