"""Single-frame georeferencing: pure solver (`solve_frame`) and IO wrapper (`run_frame`).

Ticket 08 reuses `solve_frame` per strip (pass the strip array and its footprint quad) and
`warp.warp_frame` for mosaicking; `GcpTable`/`write_gcps_csv` are the shared GCP interchange
(also what a QGIS fallback reads back).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from shapely import wkt

from sbf.catalog.resolution import match_system
from sbf.core.config import Settings, load_settings
from sbf.features.preprocess import clahe
from sbf.georef import fit as fitmod
from sbf.georef.coarse import (
    coarse_candidates,
    footprint_corners_xy,
    ground_resolution_m,
    quad_area,
)
from sbf.georef.frame_io import as_source, downsample_source
from sbf.georef.gcps import Matcher, find_gcps
from sbf.georef.mask import MIN_STABLE_FRACTION, stable_terrain_mask
from sbf.georef.models import (
    FEW_GCPS,
    SUPPORT_DIST_FRAC,
    UTM_CRS,
    GeorefReport,
    GZOffset,
    apply_quality_gate,
)
from sbf.georef.reference import ReferenceLayer
from sbf.georef.warp import estimate_resolution_m

PANORAMIC_SYSTEMS = {"KH-4", "KH-4A", "KH-4B", "KH-9 pan", "KH-7"}
LOW_OVERLAP = 0.5
TRIM_MIN_M = 10.0  # outlier trim never removes residuals below this


def is_panoramic(*texts: str | None) -> bool:
    """Camera/dataset text indicates a panoramic or strip camera (needs per-strip warping)."""
    blob = " ".join(t for t in texts if t)
    return match_system(blob) in PANORAMIC_SYSTEMS or "panoram" in blob.lower()


@dataclass
class FrameSolution:
    report: GeorefReport
    model: fitmod.TransformModel | None = None
    final_model: fitmod.TransformModel | None = None  # refit on all GCPs; used for warping
    gcps: pd.DataFrame | None = None
    extra: dict = field(default_factory=dict)


def _trim(px, xy, order: int):
    """One pass: drop GCPs whose poly-`order` residual exceeds max(3 robust sigma, TRIM_MIN_M)."""
    if len(px) < fitmod.min_points(f"poly{order}") + 4:
        return np.ones(len(px), bool)
    m = fitmod.fit_transform(px, xy, f"poly{order}")
    r = fitmod.residuals_m(m, px, xy)
    sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
    return r <= max(3 * sigma, TRIM_MIN_M)


def prepare_reference(reference: ReferenceLayer, rep: GeorefReport, use_mask: bool = True):
    """(CLAHE'd reference image, stable-terrain mask or None); records mask warnings on `rep`."""
    ref_img = clahe(reference.array)
    ref_mask = None
    if use_mask and reference.terrain:
        ref_mask = stable_terrain_mask(reference.mask_stack())
        frac = float(ref_mask.mean())
        rep.stable_fraction = round(frac, 4)
        if frac < MIN_STABLE_FRACTION:
            rep.warnings.append("stable_mask_too_restrictive_ignored")
            ref_mask = None
    elif use_mask:
        ref_mask = np.isfinite(reference.array)
    return ref_img, ref_mask


def solve_frame(
    frame,
    footprint_lonlat,
    reference: ReferenceLayer,
    *,
    entity_id: str,
    camera: str = "",
    dataset: str = "",
    aoi_id: str | None = None,
    method: str = "auto",
    matchers: tuple[Matcher, ...] | None = None,
    seed: int = 0,
    smoothing: float = 0.0,
    frame_res_m: float | None = None,
    use_mask: bool = True,
) -> FrameSolution:
    """Coarse placement -> CLAHE/downsample -> masked matching -> RANSAC -> fit + checkpoints.

    `frame` is a 2-D array or a `FrameSource` (matching only needs a decimated read)."""
    src = as_source(frame)
    shape = src.shape
    rep = GeorefReport(
        entity_id=entity_id,
        aoi_id=aoi_id,
        method="none",
        reference=reference.name,
        seed=seed,
        frame_shape=shape,
        panoramic=is_panoramic(camera, dataset),
    )
    if rep.panoramic:
        rep.warnings.append("panoramic_needs_strips")

    corners = footprint_corners_xy(footprint_lonlat)
    cands = coarse_candidates(shape, corners)
    # overlap of footprint quad bounding box with the reference extent
    l, b, r, t = reference.bounds
    cx0, cy0, cx1, cy1 = (
        corners[:, 0].min(),
        corners[:, 1].min(),
        corners[:, 0].max(),
        corners[:, 1].max(),
    )
    ix = max(0.0, min(cx1, r) - max(cx0, l))
    iy = max(0.0, min(cy1, t) - max(cy0, b))
    bbox_area = max((cx1 - cx0) * (cy1 - cy0), 1e-9)
    if ix * iy / bbox_area < LOW_OVERLAP or quad_area(corners) <= 0:
        rep.warnings.append("low_overlap")

    ref_img, ref_mask = prepare_reference(reference, rep, use_mask)

    fres = frame_res_m or ground_resolution_m(shape, corners)
    small, scale = downsample_source(src, reference.resolution_m / fres)
    out = find_gcps(
        small,
        scale,
        cands,
        ref_img,
        reference.transform,
        ref_mask,
        matchers=matchers,
        seed=seed,
    )
    if out is None or len(out.px) == 0:
        rep.warnings += ["no_matches", "few_gcps"]
        return FrameSolution(rep)
    rep.matcher, rep.coarse_variant = out.matcher, out.variant
    rep.corner_variant_inliers = out.inliers_per_variant
    rep.n_candidates = len(out.px)
    px, xy = out.px, out.xy

    return fit_gcps(rep, px, xy, shape, method=method, seed=seed, smoothing=smoothing)


def fit_gcps(
    rep: GeorefReport,
    px: np.ndarray,
    xy: np.ndarray,
    shape: tuple[int, int],
    *,
    method: str = "auto",
    seed: int = 0,
    smoothing: float = 0.0,
    trim: bool = True,
) -> FrameSolution:
    """GCPs -> (trim) -> seeded spatially blocked ~20% checkpoint split + spatial CV -> model selection -> report fields.

    Shared by the automatic frame solve and the manual (QGIS) path."""
    px, xy = np.asarray(px, float), np.asarray(xy, float)
    if len(px) < fitmod.min_points("poly1") + 2:
        rep.n_gcps = len(px)
        rep.warnings += ["few_gcps", "too_few_gcps_for_fit"]
        return FrameSolution(rep)
    # trim gross outliers relative to a low-order global fit, then split + select
    if trim:
        keep = _trim(px, xy, 2 if len(px) >= 30 else 1)
        px, xy = px[keep], xy[keep]
    rep.n_gcps = len(px)
    try:
        sel = fitmod.select_model(
            px, xy, method, seed=seed, smoothing=smoothing, extent=(0.0, 0.0, shape[1], shape[0])
        )
    except ValueError:
        rep.warnings += ["few_gcps", "too_few_gcps_for_fit"]
        return FrameSolution(rep)

    rep.method = sel.method
    rep.poly_order = int(sel.method[-1]) if sel.method.startswith("poly") else None
    rep.tps_smoothing = smoothing if sel.method == "tps" else None
    rep.n_checkpoints = len(sel.check_idx)
    rep.checkpoint_rmse_m = round(sel.check_rmse_m, 3)
    rep.train_rmse_m = round(sel.train_rmse_m, 3)
    rep.cv_rmse_m = None if sel.cv_rmse_m is None else round(sel.cv_rmse_m, 3)
    rep.random_checkpoint_rmse_m = (
        None if sel.random_check_rmse_m is None else round(sel.random_check_rmse_m, 3)
    )
    hull, occ = fitmod.coverage_metrics(px, (0.0, 0.0, float(shape[1]), float(shape[0])))
    rep.gcp_hull_fraction, rep.gcp_block_occupancy = round(hull, 4), round(occ, 4)
    rep.candidate_rmse_m = {
        k: (None if v is None else round(v, 3)) for k, v in sel.candidate_rmse_m.items()
    }
    final = fitmod.fit_transform(px, xy, sel.method, smoothing=smoothing)
    rep.resolution_m = estimate_resolution_m(final, shape)

    if rep.n_gcps < FEW_GCPS:
        rep.warnings.append("few_gcps")
    apply_quality_gate(rep)

    res = fitmod.residuals_m(sel.model, px, xy)
    role = np.full(len(px), "train", dtype=object)
    role[sel.check_idx] = "check"
    gcps = pd.DataFrame(
        {
            "id": np.arange(len(px)),
            "px": px[:, 0],
            "py": px[:, 1],
            "x": xy[:, 0],
            "y": xy[:, 1],
            "role": role,
            "residual_m": res,
        }
    )
    return FrameSolution(rep, sel.model, final, gcps)


# --------------------------------------------------------------------------- IO


def processed_dir(entity_id: str, settings: Settings | None = None) -> Path:
    from sbf.ingest.declass.acquire import entity_dir

    s = settings or load_settings()
    return entity_dir(s.data_processed / "historical", entity_id)


def raw_dir(entity_id: str, settings: Settings | None = None) -> Path:
    from sbf.ingest.declass.acquire import entity_dir

    s = settings or load_settings()
    return entity_dir(s.data_raw / "declass", entity_id)


def find_raw_frame(entity_id: str, settings: Settings | None = None, out_dir: Path | None = None):
    """The frame to georeference: the assembled mosaic when the raw directory holds several parts
    (`assemble.assemble_frame`), else the largest non-browse GeoTIFF."""
    from sbf.georef.assemble import assemble_frame

    s = settings or load_settings()
    return assemble_frame(raw_dir(entity_id, s), out_dir or processed_dir(entity_id, s)).path


def write_gcps_csv(gcps: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    gcps.to_csv(path, index=False, float_format="%.4f")
    return path


def read_frame(path: Path) -> np.ndarray:
    """Whole-frame read; only for small frames and tests (the pipeline streams instead)."""
    with rasterio.open(path) as src:
        return src.read(1)


def coverage_row(entity_id: str, settings: Settings, aoi: str | None = None) -> pd.Series:
    from sbf.catalog.report import read_coverage

    cov = read_coverage(settings.outputs_public / "coverage.csv")
    rows = cov[cov["entity_id"] == entity_id]
    if aoi:
        rows = rows[rows["aoi_id"] == aoi]
    if rows.empty:
        raise KeyError(
            f"entity {entity_id!r} not in coverage.csv" + (f" for AOI {aoi}" if aoi else "")
        )
    return rows.iloc[0]


def footprint_lonlat(row: pd.Series) -> np.ndarray:
    geom = wkt.loads(row["footprint_wkt"])
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    return np.asarray(geom.exterior.coords)[:, :2]


def _cog_from_tiled(tmp: Path, out: Path, prov: dict) -> Path:
    import rasterio.shutil

    from sbf.core.provenance import write_prov

    out.parent.mkdir(parents=True, exist_ok=True)
    rasterio.shutil.copy(tmp, out, driver="COG", compress="DEFLATE")
    tmp.unlink(missing_ok=True)
    write_prov(out, **prov)
    return out


def _write_support(sol, warped, entity_id, odir, rep, raster_path, ref_sources) -> str | None:
    """`{ID}_support.tif` on the georef grid (see `sbf.georef.support`); None if no GCP support."""
    from sbf.georef.strips import strips_support_fn
    from sbf.georef.support import frame_support_fn, grid_of, support_polygon, write_support

    shape = rep.frame_shape
    grid = grid_of(warped.transform, warped.shape)
    if getattr(sol, "strip_models", None):
        if not sol.support_px:
            return None
        fn = strips_support_fn(sol.strip_models, sol.plan, shape, sol.support_px)
    elif sol.final_model is not None and sol.gcps is not None and len(sol.gcps) >= 3:
        poly = support_polygon(
            sol.gcps[["px", "py"]].to_numpy(float), (0.0, 0.0, float(shape[1]), float(shape[0]))
        )
        fn = frame_support_fn(sol.final_model, shape, poly)
    else:
        return None
    out = odir / f"{entity_id}_support.tif"
    write_support(
        out,
        grid,
        fn,
        prov={
            "source": "sbf.georef.support",
            "source_id": entity_id,
            "acquisition_date": None,
            "steps": [
                (
                    f"1 = within {SUPPORT_DIST_FRAC:.0%} of the frame diagonal of the GCP convex "
                    "hull (frame px), mapped through the georef model; 0 elsewhere; same grid"
                )
            ],
            "inputs": [str(raster_path), *ref_sources],
        },
    )
    return str(out)


def run_frame(
    entity_id: str,
    *,
    settings: Settings | None = None,
    raster: Path | None = None,
    aoi: str | None = None,
    method: str = "auto",
    reference: ReferenceLayer | None = None,
    reference_path: Path | None = None,
    reference_mode: str = "pan",
    matchers: tuple[Matcher, ...] | None = None,
    seed: int = 0,
    gz=None,
    make_overlay: bool = True,
    strips: bool | None = None,
    n_strips: int = 8,
    strip_overlap: float = 0.1,
    strip_axis: str = "auto",
    manual_gcps: pd.DataFrame | None = None,
    part_pattern: str | None = None,
    extra_warnings: list[str] | None = None,
) -> GeorefReport:
    """Full single-frame run with IO. Outputs under `data/processed/historical/{entity_id}/`.

    Multi-part scans are assembled first; panoramic frames (or `strips=True`) are solved per
    strip; `manual_gcps` (columns px, py, x, y; e.g. from QGIS) skips matching entirely and
    reruns fit + warp + report with `gcp_source = "manual_gcps"`."""
    from sbf.georef import reference as refmod
    from sbf.georef.assemble import DEFAULT_PART_PATTERN, assemble_frame
    from sbf.georef.frame_io import RasterSource
    from sbf.georef.strips import solve_strips, warp_strips_to_file
    from sbf.georef.warp import warp_to_file

    s = settings or load_settings()
    row = coverage_row(entity_id, s, aoi)
    aoi_id = aoi or str(row.get("aoi_id") or "") or None
    odir = processed_dir(entity_id, s)
    odir.mkdir(parents=True, exist_ok=True)
    assembly = None
    if raster:
        path = Path(raster)
    else:
        assembly = assemble_frame(
            raw_dir(entity_id, s), odir, part_pattern=part_pattern or DEFAULT_PART_PATTERN
        )
        path = assembly.path
    source = RasterSource(path)
    fp = footprint_lonlat(row)
    camera, dataset = str(row.get("camera") or ""), str(row.get("dataset") or "")
    panoramic = is_panoramic(camera, dataset)
    use_strips = panoramic if strips is None else strips

    if manual_gcps is not None:
        rep = GeorefReport(
            entity_id=entity_id,
            aoi_id=aoi_id,
            method="none",
            reference="manual_gcps",
            seed=seed,
            frame_shape=source.shape,
            panoramic=panoramic,
            gcp_source="manual_gcps",
        )
        if panoramic:
            rep.warnings.append("manual_gcps_frame_wide")
        sol = fit_gcps(
            rep,
            manual_gcps[["px", "py"]].to_numpy(float),
            manual_gcps[["x", "y"]].to_numpy(float),
            source.shape,
            method=method,
            seed=seed,
            trim=False,
        )
        ref_sources: list[str] = []
    else:
        corners = footprint_corners_xy(fp)
        pad = max(1000.0, 0.1 * float(np.ptp(corners, axis=0).max()))
        bounds = (
            corners[:, 0].min() - pad,
            corners[:, 1].min() - pad,
            corners[:, 0].max() + pad,
            corners[:, 1].max() + pad,
        )
        if reference is None:
            if reference_path:
                reference = refmod.load_file_reference(reference_path, bounds_xy=bounds)
            else:
                if not aoi_id:
                    raise ValueError("AOI unknown: pass --aoi or fill aoi_id in coverage.csv")
                reference = refmod.load_stack_reference(
                    aoi_id, s, mode=reference_mode, bounds_xy=bounds
                )
        kw = {
            "entity_id": entity_id,
            "camera": camera,
            "dataset": dataset,
            "aoi_id": aoi_id,
            "method": method,
            "matchers": matchers,
            "seed": seed,
        }
        if use_strips:
            sol = solve_strips(
                source,
                fp,
                reference,
                n_strips=n_strips,
                overlap=strip_overlap,
                axis=strip_axis,
                **kw,
            )
        else:
            sol = solve_frame(source, fp, reference, **kw)
        ref_sources = list(reference.sources)
    rep = sol.report
    rep.warnings += extra_warnings or []
    if assembly is not None and not assembly.passthrough:
        rep.assembly_file = str(assembly.json_path)
        rep.warnings += assembly.warnings
    if sol.gcps is not None:
        rep.gcps_file = str(write_gcps_csv(sol.gcps, odir / "gcps.csv"))

    warped = None
    out = odir / f"{entity_id}_georef.tif"
    tmp = odir / f".{entity_id}_tiled.tif"
    if getattr(sol, "strip_models", None) and any(m is not None for m in sol.strip_models):
        warped = warp_strips_to_file(
            source, sol.strip_models, sol.plan, tmp, resolution_m=rep.resolution_m
        )
        warp_step = (
            f"{len(sol.strip_models)} strips warped to a common grid, linear seam feathering"
        )
    elif sol.final_model is not None:
        warped = warp_to_file(source, sol.final_model, tmp, resolution_m=rep.resolution_m)
        warp_step = "inverse-map windowed warp"
    if warped is not None:
        how = (
            "manual GCPs (QGIS)"
            if manual_gcps is not None
            else (
                f"coarse placement from footprint corners (variant {rep.coarse_variant}, "
                f"matcher {rep.matcher}); CLAHE + decimated read for matching; SIFT/ORB + RANSAC"
            )
        )
        steps = [
            how,
            (
                f"{rep.method} fit, {rep.n_gcps} GCPs ({rep.n_checkpoints} checkpoints), "
                f"spatial-CV RMSE {rep.cv_rmse_m} m, blocked checkpoint RMSE {rep.checkpoint_rmse_m} m, "
                f"train RMSE {rep.train_rmse_m} m"
            ),
            f"{warp_step} to {UTM_CRS} at {warped.resolution_m} m",
        ]
        if assembly is not None and not assembly.passthrough:
            steps.insert(0, f"assembled {len(assembly.parts)} scan parts (assembly.json)")
        _cog_from_tiled(
            tmp,
            out,
            {
                "source": str(row.get("dataset") or "USGS Declass"),
                "source_id": entity_id,
                "acquisition_date": str(row.get("acquisition_date") or "") or None,
                "steps": steps,
                "inputs": [str(path), *ref_sources],
            },
        )
        warped.path = out
        rep.georef_file = str(out)
        rep.support_file = _write_support(sol, warped, entity_id, odir, rep, path, ref_sources)
    rep.gz_offsets = {}
    if gz is None:
        from sbf.core.aoi import load_reference_features

        gz = load_reference_features(s)
    if warped is not None and make_overlay:
        from sbf.georef.overlay import gz_overlay

        png, inside = gz_overlay(warped, gz, entity_id, s.outputs_public / "georef")
        rep.overlay_file = str(png) if png else None
        rep.gz_offsets = {fid: GZOffset(in_frame=True) for fid in inside}
    (odir / "georef_report.json").write_text(rep.model_dump_json(indent=2))
    return rep
