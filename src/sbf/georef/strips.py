"""Panoramic frames: per-strip solve, common-grid warp and seam feathering (ticket 08).

A panoramic (scan-arm) camera has strong across-track scale change (about 1/cos^2 of the scan
angle) plus bowing, so no frame-wide polynomial fits (SPEC section 7). The frame is split along
`axis` (default: the longer film dimension, i.e. the scan direction along which the scale varies)
into `n` strips with `overlap` (fraction of a strip length) shared with each neighbour. Each
strip is matched against the reference, seeded from the frame-wide affine solution, and gets its
own model in FULL-FRAME pixel coordinates (so strips share one pixel space and one source).
Strips with too few GCPs inherit the nearest strip's model (else the frame-wide model) and are
flagged `strip_fallback`. Overall checkpoint RMSE is the RMSE over ALL checkpoints, each scored by
its own strip's train-only model; overall `cv_rmse_m` pools every strip's out-of-fold residuals
(spatial CV inside each strip's own extent). Coverage metrics are computed per strip and for the
pooled GCPs over the whole frame; fallback strips contribute no GCPs and no support. Warping goes to one shared grid; in the overlap the strips are
blended with weights ramping linearly 0 -> 1 across the overlap (distance to the strip edge).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.windows import Window

from sbf.georef import fit as fitmod
from sbf.georef.coarse import (
    CoarseCandidate,
    coarse_candidates,
    footprint_corners_xy,
    ground_resolution_m,
)
from sbf.georef.fit import invert_on_lattice
from sbf.georef.frame_io import as_source, downsample_source
from sbf.georef.gcps import Matcher, find_gcps
from sbf.georef.models import (
    FEW_GCPS,
    UTM_CRS,
    StripReport,
    apply_quality_gate,
)
from sbf.georef.pipeline import FrameSolution, prepare_reference, solve_frame
from sbf.georef.reference import ReferenceLayer
from sbf.georef.warp import (
    Grid,
    WarpFile,
    WarpResult,
    _nodata_for,
    block_source_coords,
    frame_bounds_xy,
    open_tiled_output,
    sample_source,
)

DEFAULT_STRIPS = 8
DEFAULT_OVERLAP = 0.10
MIN_STRIP_PX = 64  # small-image pixels; narrower strips are not matched


@dataclass(frozen=True)
class StripPlan:
    axis: str  # "x" (strips are column ranges) or "y" (row ranges)
    ranges: tuple[tuple[float, float], ...]  # extended (with overlap) range of each strip, frame px
    overlap_px: float  # length of the shared zone between neighbours

    @property
    def n(self) -> int:
        return len(self.ranges)


def plan_strips(
    shape: tuple[int, int], n: int = DEFAULT_STRIPS, overlap: float = DEFAULT_OVERLAP, axis="auto"
) -> StripPlan:
    h, w = shape[:2]
    if axis == "auto":
        axis = "x" if w >= h else "y"
    if axis not in ("x", "y"):
        raise ValueError("axis must be auto, x or y")
    if n < 1:
        raise ValueError("n strips must be >= 1")
    length = float(w if axis == "x" else h)
    core = length / n
    ov = overlap * core if n > 1 else 0.0
    ranges = []
    for k in range(n):
        e0 = 0.0 if k == 0 else k * core - ov / 2
        e1 = length if k == n - 1 else (k + 1) * core + ov / 2
        ranges.append((e0, e1))
    return StripPlan(axis, tuple(ranges), ov)


def _extent(plan: StripPlan, k: int, shape) -> tuple[float, float, float, float]:
    h, w = shape[:2]
    a, b = plan.ranges[k]
    return (a, 0.0, b, float(h)) if plan.axis == "x" else (0.0, a, float(w), b)


def _axis_coord(plan: StripPlan, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    return sx if plan.axis == "x" else sy


def strip_weights(plan: StripPlan, k: int, t: np.ndarray) -> np.ndarray:
    """Feather weight of strip `k` at along-axis position `t` (0 outside the strip)."""
    a, b = plan.ranges[k]
    w = np.where((t >= a) & (t <= b), 1.0, 0.0)
    if plan.overlap_px > 0:
        if k > 0:
            w = np.minimum(w, (t - a) / plan.overlap_px)
        if k < plan.n - 1:
            w = np.minimum(w, (b - t) / plan.overlap_px)
    inside = (t >= a) & (t <= b)
    return np.where(inside, np.clip(w, 1e-4, 1.0), 0.0).astype("float32")


@dataclass
class StripSolution(FrameSolution):
    plan: StripPlan | None = None
    strip_models: list = field(default_factory=list)  # final (all-GCP) model per strip or None
    shape: tuple[int, int] | None = None
    support_px: dict = field(default_factory=dict)  # strip index -> own GCP px (no fallbacks)


def _local_candidate(variant: int, model, extent) -> CoarseCandidate:
    """Homography frame px -> xy: least-squares affine of `model` over the pixel `extent`."""
    x0, y0, x1, y1 = extent
    u, v = np.meshgrid(np.linspace(x0, x1, 5), np.linspace(y0, y1, 5))
    p = np.column_stack([u.ravel(), v.ravel()])
    q = model(p)
    coef, *_ = np.linalg.lstsq(np.column_stack([p, np.ones(len(p))]), q, rcond=None)
    H = np.eye(3)
    H[:2, :] = coef.T
    corners = model(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float))
    return CoarseCandidate(variant, H, corners)


def _shift(H: np.ndarray, dx: float, dy: float) -> np.ndarray:
    T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])
    return H @ T


def solve_strips(
    frame,
    footprint_lonlat,
    reference: ReferenceLayer,
    *,
    entity_id: str,
    camera: str = "",
    dataset: str = "",
    aoi_id: str | None = None,
    method: str = "auto",
    n_strips: int = DEFAULT_STRIPS,
    overlap: float = DEFAULT_OVERLAP,
    axis: str = "auto",
    matchers: tuple[Matcher, ...] | None = None,
    seed: int = 0,
    smoothing: float = 0.0,
    frame_res_m: float | None = None,
    use_mask: bool = True,
) -> StripSolution:
    src = as_source(frame)
    shape = src.shape
    plan = plan_strips(shape, n_strips, overlap, axis)
    wide = solve_frame(
        src,
        footprint_lonlat,
        reference,
        entity_id=entity_id,
        camera=camera,
        dataset=dataset,
        aoi_id=aoi_id,
        method=method,
        matchers=matchers,
        seed=seed,
        smoothing=smoothing,
        frame_res_m=frame_res_m,
        use_mask=use_mask,
    )
    rep = wide.report.model_copy(deep=True)
    rep.strip_axis = plan.axis
    rep.warnings = [w for w in rep.warnings if w != "panoramic_needs_strips"]
    wide_rmse = wide.report.checkpoint_rmse_m
    rep.candidate_rmse_m = {"frame_wide": wide_rmse}

    corners = footprint_corners_xy(footprint_lonlat)
    base_seeds = coarse_candidates(shape, corners)
    # affine of the frame-wide GCPs as the global seed (poly1 even if the wide fit chose more)
    wide_seed = None
    if wide.gcps is not None and len(wide.gcps) >= 6:
        wide_seed = fitmod.fit_transform(
            wide.gcps[["px", "py"]].to_numpy(), wide.gcps[["x", "y"]].to_numpy(), "poly1"
        )
    ref_img, ref_mask = prepare_reference(reference, rep, use_mask)
    fres = frame_res_m or ground_resolution_m(shape, corners)
    small, scale = downsample_source(src, reference.resolution_m / fres)

    per: list[dict | None] = [None] * plan.n
    variant = rep.coarse_variant or 0
    # centre strip first, then outwards: each strip is seeded by the nearest solved neighbour's
    # model extrapolated over its own extent (a global affine cannot follow 1/cos^2 scale change)
    for k in sorted(range(plan.n), key=lambda j: (abs(j - (plan.n - 1) / 2), j)):
        a, b = plan.ranges[k]
        s0, s1 = round(a / scale), round(b / scale)
        off = s0 * scale
        if plan.axis == "x":
            sub, (dx, dy) = small[:, s0:s1], (off, 0.0)
        else:
            sub, (dx, dy) = small[s0:s1, :], (0.0, off)
        if min(sub.shape) < MIN_STRIP_PX:
            continue
        solved = [j for j in range(plan.n) if per[j] is not None]
        if solved:
            nb = min(solved, key=lambda j: (abs(j - k), j))
            seeds = [_local_candidate(variant, per[nb]["final"], _extent(plan, k, shape))]
        elif wide_seed is not None:
            seeds = [_local_candidate(variant, wide_seed, _extent(plan, k, shape))]
        else:
            seeds = base_seeds
        cands = [CoarseCandidate(c.variant, _shift(c.H, dx, dy), c.corners_xy) for c in seeds]
        out = find_gcps(
            np.ascontiguousarray(sub),
            scale,
            cands,
            ref_img,
            reference.transform,
            ref_mask,
            matchers=matchers,
            seed=seed + k,
        )
        if out is None or len(out.px) < fitmod.min_points("poly1") + 2:
            continue
        px = out.px + np.array([dx, dy])
        xy = out.xy
        try:
            sel = fitmod.select_model(
                px,
                xy,
                method,
                seed=seed + k,
                smoothing=smoothing,
                extent=_extent(plan, k, shape),
            )
        except ValueError:
            continue
        final = fitmod.fit_transform(px, xy, sel.method, smoothing=smoothing)
        per[k] = {"px": px, "xy": xy, "sel": sel, "final": final}

    # per-strip reports, fallback inheritance
    ok = [k for k, p in enumerate(per) if p is not None]
    strip_reps: list[StripReport] = []
    models: list = [None] * plan.n
    for k in range(plan.n):
        a, b = plan.ranges[k]
        sr = StripReport(index=k, axis_range_px=(round(a, 1), round(b, 1)))
        p = per[k]
        if p is not None:
            sel = p["sel"]
            sr.method, sr.n_gcps = sel.method, len(p["px"])
            sr.n_checkpoints = len(sel.check_idx)
            sr.checkpoint_rmse_m = round(sel.check_rmse_m, 3)
            sr.train_rmse_m = round(sel.train_rmse_m, 3)
            sr.cv_rmse_m = None if sel.cv_rmse_m is None else round(sel.cv_rmse_m, 3)
            if sel.random_check_rmse_m is not None:
                sr.random_checkpoint_rmse_m = round(sel.random_check_rmse_m, 3)
            hull, occ = fitmod.coverage_metrics(p["px"], _extent(plan, k, shape))
            sr.gcp_hull_fraction, sr.gcp_block_occupancy = round(hull, 4), round(occ, 4)
            models[k] = p["final"]
            if sr.n_gcps < 10:
                sr.warnings.append("few_gcps")
        else:
            sr.warnings.append("strip_fallback")
            if ok:
                nb = min(ok, key=lambda j: (abs(j - k), j))
                models[k], sr.fallback_from = per[nb]["final"], f"strip:{nb}"
                sr.method = per[nb]["sel"].method
            elif wide.final_model is not None:
                models[k], sr.fallback_from = wide.final_model, "frame"
                sr.method = wide.final_model.method
        strip_reps.append(sr)
    rep.strips = strip_reps

    if all(m is None for m in models):
        rep.method = "none"
        rep.warnings += ["no_matches", "few_gcps"]
        return StripSolution(rep, None, None, wide.gcps, plan=plan, shape=shape)

    # overall: pooled checkpoints / train residuals across strips (own train-only model)
    chk, trn, rows = [], [], []
    gid = 0
    for k in ok:
        p = per[k]
        sel = p["sel"]
        res = fitmod.residuals_m(sel.model, p["px"], p["xy"])
        role = np.full(len(res), "train", dtype=object)
        role[sel.check_idx] = "check"
        chk.append(res[sel.check_idx])
        trn.append(res[sel.train_idx])
        rows.append(
            pd.DataFrame(
                {
                    "id": np.arange(gid, gid + len(res)),
                    "px": p["px"][:, 0],
                    "py": p["px"][:, 1],
                    "x": p["xy"][:, 0],
                    "y": p["xy"][:, 1],
                    "role": role,
                    "residual_m": res,
                    "strip": k,
                }
            )
        )
        gid += len(res)
    chk_all = np.concatenate(chk) if chk else np.array([])
    trn_all = np.concatenate(trn) if trn else np.array([])
    rep.method = "strips"
    rep.n_gcps = int(sum(len(p["px"]) for p in per if p))
    rep.n_checkpoints = len(chk_all)
    rep.checkpoint_rmse_m = round(fitmod.rmse(chk_all), 3) if len(chk_all) else None
    rep.train_rmse_m = round(fitmod.rmse(trn_all), 3) if len(trn_all) else None
    rep.poly_order = None
    if rep.n_gcps < FEW_GCPS:
        rep.warnings.append("few_gcps")
    if len(ok) < plan.n:
        rep.warnings.append("strip_fallback")
    cv_all = [per[k]["sel"].cv_res for k in ok if len(per[k]["sel"].cv_res)]
    cv_pool = np.concatenate(cv_all) if cv_all else np.array([])
    rep.cv_rmse_m = round(fitmod.rmse(cv_pool), 3) if len(cv_pool) else None
    rnd = [per[k]["sel"].random_check_rmse_m for k in ok]
    rnd = [r for r in rnd if r is not None]
    rep.random_checkpoint_rmse_m = (
        round(float(np.sqrt(np.mean(np.square(rnd)))), 3) if rnd else None
    )
    all_px = np.concatenate([per[k]["px"] for k in ok]) if ok else np.empty((0, 2))
    hull, occ = fitmod.coverage_metrics(all_px, (0.0, 0.0, float(shape[1]), float(shape[0])))
    rep.gcp_hull_fraction, rep.gcp_block_occupancy = round(hull, 4), round(occ, 4)
    if rep.checkpoint_rmse_m is None:
        rep.warnings.append("no_checkpoints")
    apply_quality_gate(rep)
    rep.resolution_m = strip_resolution_m(models, plan, shape)
    gcps = pd.concat(rows, ignore_index=True) if rows else None
    support_px = {k: per[k]["px"] for k in ok}
    return StripSolution(
        rep,
        None,
        None,
        gcps,
        plan=plan,
        strip_models=models,
        shape=shape,
        support_px=support_px,
    )


def strip_resolution_m(models, plan: StripPlan, shape) -> float:
    """Finest ground pixel size over the strips (measured at each strip centre)."""
    best = []
    for k, m in enumerate(models):
        if m is None:
            continue
        x0, y0, x1, y1 = _extent(plan, k, shape)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        p = m(np.array([[cx, cy], [cx + 50, cy], [cx, cy + 50]]))
        best.append(0.5 * (np.hypot(*(p[1] - p[0])) + np.hypot(*(p[2] - p[0]))) / 50.0)
    return round(float(min(best)), 2)


# --------------------------------------------------------------------------- warp


def strips_grid(models, plan: StripPlan, shape, res: float) -> Grid:
    bs = [
        frame_bounds_xy(m, shape, extent=_extent(plan, k, shape))
        for k, m in enumerate(models)
        if m is not None
    ]
    b = np.array(bs)
    return Grid.from_bounds((b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()), res)


def _blend_blocks(src, models, plan: StripPlan, grid: Grid, block: int, step: int):
    invs = [
        None if m is None else invert_on_lattice(m, src.shape, extent=_extent(plan, k, src.shape))
        for k, m in enumerate(models)
    ]
    integer = np.issubdtype(src.dtype, np.integer)
    for r0, c0, nr, nc in grid.blocks(block):
        acc = np.zeros((nr, nc), "float32")
        wsum = np.zeros((nr, nc), "float32")
        for k, inv in enumerate(invs):
            if inv is None:
                continue
            sx, sy = block_source_coords(inv, grid, r0, c0, nr, nc, step)
            wgt = strip_weights(plan, k, _axis_coord(plan, sx, sy))
            if not wgt.any():
                continue
            tile, inside = sample_source(src, sx, sy, valid=wgt > 0)
            wgt = np.where(inside, wgt, 0.0)
            acc += wgt * tile.astype("float32")
            wsum += wgt
        val = acc / np.maximum(wsum, 1e-9)
        if integer:
            info = np.iinfo(src.dtype)
            val = np.clip(np.rint(val), info.min, info.max)
        yield r0, c0, val.astype(src.dtype), wsum > 0


def warp_strips(
    frame,
    models: list,
    plan: StripPlan,
    *,
    resolution_m: float,
    bounds=None,
    block: int = 1024,
    step: int = 16,
    nodata: float | None = None,
    crs: str = UTM_CRS,
) -> WarpResult:
    """In-memory feathered mosaic of the strips on one grid (tests / small frames)."""
    src = as_source(frame)
    grid = (
        Grid.from_bounds(bounds, resolution_m)
        if bounds
        else strips_grid(models, plan, src.shape, resolution_m)
    )
    nd = _nodata_for(src.dtype, nodata)
    out = np.full((grid.H, grid.W), nd, dtype=src.dtype)
    for r0, c0, tile, inside in _blend_blocks(src, models, plan, grid, block, step):
        sub = out[r0 : r0 + tile.shape[0], c0 : c0 + tile.shape[1]]
        sub[inside] = tile[inside]
    return WarpResult(out, grid.transform, crs, float(nd), resolution_m)


def warp_strips_to_file(
    frame,
    models: list,
    plan: StripPlan,
    path: str | Path,
    *,
    resolution_m: float,
    block: int = 1024,
    step: int = 16,
    nodata: float | None = None,
    crs: str = UTM_CRS,
) -> WarpFile:
    src = as_source(frame)
    grid = strips_grid(models, plan, src.shape, resolution_m)
    nd = _nodata_for(src.dtype, nodata)
    path = Path(path)
    with open_tiled_output(path, grid, src.dtype, nd, crs) as dst:
        for r0, c0, tile, inside in _blend_blocks(src, models, plan, grid, block, step):
            blk = np.where(inside, tile, np.asarray(nd, dtype=tile.dtype)).astype(src.dtype)
            dst.write(blk, 1, window=Window(c0, r0, tile.shape[1], tile.shape[0]))
    return WarpFile(path, grid.transform, crs, float(nd), resolution_m, (grid.H, grid.W))


def strips_support_fn(models, plan: StripPlan, shape, support_px: dict, *, step: int = 16):
    """Support-mask block function for a strip mosaic: a pixel is supported when, in some strip
    that has its OWN GCPs, it maps inside that strip's range and within D of that strip's GCP hull.
    Fallback strips (no entry in `support_px`) never support anything."""
    from shapely import contains_xy

    from sbf.georef.support import support_polygon

    h, w = shape[:2]
    layers = []
    for k, pts in support_px.items():
        if models[k] is None or len(pts) < 3:
            continue
        ext = _extent(plan, k, shape)
        layers.append(
            (k, invert_on_lattice(models[k], shape, extent=ext), support_polygon(pts, ext))
        )

    def fn(grid: Grid, r0: int, c0: int, nr: int, nc: int) -> np.ndarray:
        out = np.zeros((nr, nc), bool)
        for k, inv, poly in layers:
            sx, sy = block_source_coords(inv, grid, r0, c0, nr, nc, step)
            inside = (sx >= 0) & (sx <= w) & (sy >= 0) & (sy <= h)
            inside &= strip_weights(plan, k, _axis_coord(plan, sx, sy)) > 0
            out |= inside & contains_xy(poly, sx, sy)
        return out

    return fn
