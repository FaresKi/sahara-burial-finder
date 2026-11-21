"""Multi-part scan assembly.

USGS Declass scans are often delivered as several TIFF parts per entity (suffix `_a`, `_b`, ...),
overlapping along the film strip. `assemble_frame(entity_dir, out_dir)` returns ONE un-georeferenced
frame: a single part is passed through untouched; several parts are mosaicked by SIFT matching of
their overlap zones (similarity without scale: translation + small rotation, RANSAC) on
decimated reads, then composited blockwise at full resolution with distance-to-edge feathering
into a tiled GeoTIFF `assembled.tif`, with `assembly.json` (order, offsets, rotation, inliers).

UNVERIFIED against real downloads: `DEFAULT_PART_PATTERN` (letter suffix before the extension) and
the assumption that part letters follow the strip order. Check on the first real download and
override `part_pattern` if the naming differs. Seam accuracy is ~ the decimation factor (a few
full-resolution pixels); the later GCP fit absorbs it.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from rasterio.windows import Window

from sbf.georef.frame_io import RasterSource, _quiet
from sbf.georef.warp import Grid, open_tiled_output, sample_source

DEFAULT_PART_PATTERN = r"^(?P<base>.+?)_(?P<part>[a-z])\.tiff?$"
MATCH_MAX_DIM = 2500  # decimate parts so the longest side is at most this for matching
ZONE_FRAC = 0.35  # a part's overlap is searched in this fraction of its edge region
MIN_INLIERS = 12
MAX_ROT_DEG = 2.0
NEGLIGIBLE_ROT_DEG = 0.1  # below this the rotation is treated as noise (pure translation)
MAX_SCALE_DEV = 0.03
BLOCK = 1024


class AssemblyError(ValueError):
    pass


@dataclass
class PartInfo:
    file: str
    shape: tuple[int, int]
    order: int
    offset_xy: tuple[float, float]  # canvas position of the part's top-left (px)
    rotation_deg: float = 0.0
    inliers: int | None = None  # RANSAC inliers of the match to the previous part
    side: str | None = None  # where this part sits relative to the previous one


@dataclass
class Assembly:
    path: Path
    passthrough: bool
    shape: tuple[int, int]
    parts: list[PartInfo] = field(default_factory=list)
    json_path: Path | None = None
    warnings: list[str] = field(default_factory=list)


def list_tifs(entity_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in Path(entity_dir).glob("*")
        if p.suffix.lower() in {".tif", ".tiff"} and "browse" not in p.name.lower()
    )


def detect_parts(files: list[Path], part_pattern: str = DEFAULT_PART_PATTERN) -> list[Path]:
    """Files of the largest group sharing a base name under `part_pattern`, in part order.
    Empty when fewer than two files match."""
    rx = re.compile(part_pattern, re.IGNORECASE)
    groups: dict[str, list[tuple[str, Path]]] = {}
    for f in files:
        m = rx.match(f.name)
        if m:
            groups.setdefault(m.group("base"), []).append((m.group("part").lower(), f))
    if not groups:
        return []
    best = max(groups.values(), key=len)
    if len(best) < 2:
        return []
    return [f for _, f in sorted(best, key=lambda t: t[0])]


# --------------------------------------------------------------------------- matching

_SIDES = ("right", "left", "below", "above")  # where part B sits relative to part A


def _zone_masks(shape_a, shape_b, side: str, frac: float):
    ha, wa = shape_a
    hb, wb = shape_b
    ma = np.zeros((ha, wa), np.uint8)
    mb = np.zeros((hb, wb), np.uint8)
    if side == "right":
        ma[:, int(wa * (1 - frac)) :] = 255
        mb[:, : int(wb * frac)] = 255
    elif side == "left":
        ma[:, : int(wa * frac)] = 255
        mb[:, int(wb * (1 - frac)) :] = 255
    elif side == "below":
        ma[int(ha * (1 - frac)) :, :] = 255
        mb[: int(hb * frac), :] = 255
    else:
        ma[: int(ha * frac), :] = 255
        mb[int(hb * (1 - frac)) :, :] = 255
    return ma, mb


def _u8(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(a.astype("float32"))
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        hi = lo + 1
    return np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype("uint8")


def estimate_pair(a: np.ndarray, b: np.ndarray, *, zone_frac=ZONE_FRAC, min_inliers=MIN_INLIERS):
    """Similarity (no scale) mapping part-b px -> part-a px, in the given (decimated) pixels.

    Returns dict(side, R (2x2), t (2,), rot_deg, inliers) for the best of 4 placements, or None.
    """
    ua, ub = _u8(a), _u8(b)
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02)
    best = None
    for side in _SIDES:
        ma, mb = _zone_masks(a.shape, b.shape, side, zone_frac)
        ka, da = sift.detectAndCompute(ua, ma)
        kb, db = sift.detectAndCompute(ub, mb)
        if da is None or db is None or len(ka) < 4 or len(kb) < 4:
            continue
        knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        if len(good) < 4:
            continue
        pb = np.array([kb[g.queryIdx].pt for g in good], "float32")
        pa = np.array([ka[g.trainIdx].pt for g in good], "float32")
        cv2.setRNGSeed(0)
        M, inl = cv2.estimateAffinePartial2D(pb, pa, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None or inl is None:
            continue
        s = float(np.hypot(M[0, 0], M[1, 0]))
        rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        n = int(inl.sum())
        if abs(s - 1) > MAX_SCALE_DEV or abs(rot) > MAX_ROT_DEG:
            continue
        if best is None or n > best["inliers"]:
            if abs(rot) < NEGLIGIBLE_ROT_DEG:
                rot = 0.0
            th = math.radians(rot)
            R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            m = inl.ravel().astype(bool)
            t = (pa[m] - pb[m] @ R.T).mean(axis=0)
            best = {"side": side, "R": R, "t": t, "rot_deg": rot, "inliers": n}
    if best is None or best["inliers"] < min_inliers:
        return None
    return best


def _to_full(R: np.ndarray, t: np.ndarray, f: float):
    """Decimated centre-convention similarity -> full-resolution corner-convention (R, t)."""
    tc = t + 0.5 - R @ np.array([0.5, 0.5])  # centre -> corner convention (decimated px)
    return R, tc * f


def assemble_frame(
    entity_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    part_pattern: str = DEFAULT_PART_PATTERN,
    zone_frac: float = ZONE_FRAC,
    min_inliers: int = MIN_INLIERS,
    force: bool = False,
) -> Assembly:
    """One frame from `entity_dir`. Single (or unmatched) tif: passthrough of the largest
    non-browse tif. Multiple parts: mosaic into `out_dir/assembled.tif` + `assembly.json`
    (`out_dir` defaults to `entity_dir`; the pipeline passes the processed directory)."""
    entity_dir = Path(entity_dir)
    files = list_tifs(entity_dir)
    if not files:
        raise FileNotFoundError(
            f"no raw frame (.tif) under {entity_dir}; run `sbf ingest declass download`"
        )
    parts = detect_parts(files, part_pattern)
    if not parts:
        biggest = max(files, key=lambda p: p.stat().st_size)
        shape = RasterSource(biggest).shape
        return Assembly(biggest, True, shape, [PartInfo(biggest.name, shape, 0, (0.0, 0.0))])
    out_dir = Path(out_dir) if out_dir else entity_dir
    out = out_dir / "assembled.tif"
    meta = out_dir / "assembly.json"
    if out.exists() and meta.exists() and not force:
        d = json.loads(meta.read_text())
        if d.get("sources") == [[p.name, p.stat().st_size] for p in parts]:
            ps = [
                PartInfo(**{**q, "shape": tuple(q["shape"]), "offset_xy": tuple(q["offset_xy"])})
                for q in d["parts"]
            ]
            return Assembly(out, False, tuple(d["shape"]), ps, meta, d.get("warnings", []))

    srcs = [RasterSource(p) for p in parts]
    dtype = srcs[0].dtype
    f = max(1.0, max(max(s.shape) for s in srcs) / MATCH_MAX_DIM)
    small = [s.read_decimated(f) for s in srcs]
    fx = [s.shape[1] / sm.shape[1] for s, sm in zip(srcs, small, strict=True)]
    # chain: A_k maps part-k full-res corner px -> canvas px (before origin shift)
    mats = [(np.eye(2), np.zeros(2))]
    infos = [{"inliers": None, "side": None}]
    for k in range(1, len(srcs)):
        est = estimate_pair(small[k - 1], small[k], zone_frac=zone_frac, min_inliers=min_inliers)
        if est is None:
            raise AssemblyError(
                f"cannot match parts {parts[k - 1].name} and {parts[k].name} "
                f"(< {min_inliers} inliers); assemble manually"
            )
        R, t = _to_full(est["R"], est["t"], fx[k])
        Rp, tp = mats[-1]
        mats.append((Rp @ R, Rp @ t + tp))
        infos.append({"inliers": est["inliers"], "side": est["side"], "rot": est["rot_deg"]})
    mats = [
        (R, np.rint(t) if np.allclose(R, np.eye(2), atol=1e-9) else t) for R, t in mats
    ]  # integer translation when there is no rotation
    corners = []
    for s, (R, t) in zip(srcs, mats, strict=True):
        h, w = s.shape
        c = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
        corners.append(c @ R.T + t)
    allc = np.concatenate(corners)
    origin = np.floor(allc.min(axis=0))
    mats = [(R, t - origin) for R, t in mats]
    W, H = (math.ceil(v) for v in allc.max(axis=0) - origin)
    grid = Grid(0.0, float(H), 1.0, W, H)

    out_dir.mkdir(parents=True, exist_ok=True)
    nd = 0
    with _quiet(), open_tiled_output(out, grid, dtype, nd, crs=None) as dst:
        for r0, c0, nr, nc in grid.blocks(BLOCK):
            acc = np.zeros((nr, nc), "float32")
            wsum = np.zeros((nr, nc), "float32")
            cx, cy = np.meshgrid(c0 + np.arange(nc) + 0.5, r0 + np.arange(nr) + 0.5)
            cxy = np.column_stack([cx.ravel(), cy.ravel()])
            for s, (R, t) in zip(srcs, mats, strict=True):
                p = (cxy - t) @ R  # inverse of p_canvas = R p_part + t (R orthonormal)
                sx, sy = p[:, 0].reshape(nr, nc), p[:, 1].reshape(nr, nc)
                tile, inside = sample_source(s, sx, sy)
                if not inside.any():
                    continue
                h, w = s.shape
                d = np.minimum.reduce([sx, w - sx, sy, h - sy]).astype("float32")
                wgt = np.where(inside, np.maximum(d, 1e-3), 0.0).astype("float32")
                acc += wgt * tile.astype("float32")
                wsum += wgt
            val = np.where(wsum > 0, acc / np.maximum(wsum, 1e-9), 0)
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                val = np.clip(np.rint(val), info.min, info.max)
            dst.write(val.astype(dtype), 1, window=Window(c0, r0, nc, nr))

    pinfos = []
    for k, (p, s, (R, t)) in enumerate(zip(parts, srcs, mats, strict=True)):
        pinfos.append(
            PartInfo(
                p.name,
                s.shape,
                k,
                (round(float(t[0]), 2), round(float(t[1]), 2)),
                round(infos[k].get("rot", 0.0), 4),
                infos[k]["inliers"],
                infos[k]["side"],
            )
        )
    warns = [f"seam_accuracy_about_{f:.1f}px"] if f > 1 else []
    meta.write_text(
        json.dumps(
            {
                "sources": [[p.name, p.stat().st_size] for p in parts],
                "part_pattern": part_pattern,
                "match_decimation": round(f, 3),
                "shape": [H, W],
                "dtype": str(np.dtype(dtype)),
                "parts": [q.__dict__ for q in pinfos],
                "warnings": warns,
            },
            indent=2,
            default=list,
        )
    )
    return Assembly(out, False, (H, W), pinfos, meta, warns)
