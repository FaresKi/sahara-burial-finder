"""Change detection between co-registered dates (SPEC Phase 4, detector 3; ticket 11).

Pipeline per pair (A earlier, B later; historical-vs-modern always has the modern layer as B):

1. `select_pairs`: included frames only; pre_1967 x post_1967 (overlapping), pre_1967 x
   dismantling_1967, and every historical frame vs the modern ``s2_median``; same-era pairs on
   request. Overlap of the two footprints must reach ``PairPolicy.min_overlap_m2``.
2. `prepare_pair`: both sources are resampled onto ONE grid: the intersection of footprints (and of
   the GCP-support masks when `sbf.georef.load_support` exists), at the COARSER native resolution
   (the finer one is block-averaged, nothing is ever upsampled), aligned to the tile lattice
   origin. Then CLAHE on both (desert film saturation) and per-pair quantile (histogram) matching
   of B to A inside the stable-terrain area. Modern S2 "pan" = mean of B2, B3, B4.
3. `diff_scores`: per tile, ``change_diff`` = mean of the within-pair ranks of (1 - SSIM) and
   normalised absolute difference, each taken as the MINIMUM over small pixel shifts of B within
   +-r px (r = ceil(max(RMSE_A, RMSE_B) / pixel), capped at 3). Residual misregistration makes
   every edge look like change; searching the shift that best explains the tile is the main
   false-alarm suppressor. Genuinely new structure (trenches, tracks) cannot be explained by a
   shift and survives.
4. `embed_scores`: ``change_embed`` = cosine distance between the encoder embeddings of the two
   chips. The shift-min trick is too expensive for embeddings (one encoder pass per shift), so it
   is NOT applied there; use 128 m tiles (>= ~16 px at the coarse resolution) so a few pixels of
   misregistration are a small fraction of the tile.

Tiles with > 50 % unstable (dune-like, `stable_terrain_mask`) pixels, or < 50 % pixels inside the
support/footprint, are dropped (counted in the log). Labels are never used (D7); only counts and
tile IDs are logged (D8).

Layer names (shared with fusion, which maps any ``pair_*`` layer of a change detector to
``hist_change``): `historical_layer_name` = ``{camera_short}_{entity_id}`` lower-case; pair layer
= ``pair_{A}_{B}``; historical-vs-modern = ``pair_{A}_s2_median``.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds
from scipy.stats import rankdata

from sbf.catalog.resolution import match_system
from sbf.core.config import Settings, load_settings
from sbf.detect.embedding import _write_shard, cached_ids, embedding_dir, load_embeddings
from sbf.detect.scores import write_scores
from sbf.features.embeddings import TileEncoder
from sbf.features.preprocess import clahe
from sbf.features.tiling import grid_origin, make_grid

log = logging.getLogger(__name__)

CRS = "EPSG:32631"
MODERN_LAYER = "s2_median"
PAN_BANDS = ("B2", "B3", "B4")
MAX_SHIFT_PX = 3
MAX_UNSTABLE_FRAC = 0.5
MIN_VALID_FRAC = 0.5
_CAMERA_SHORT = {
    "KH-4B": "kh4b",
    "KH-4A": "kh4a",
    "KH-4": "kh4",
    "KH-7": "kh7",
    "KH-9 mapping": "kh9map",
    "KH-9 pan": "kh9pan",
    "ARGON": "argon",
    "LANYARD": "lanyard",
}
HISTORICAL_ERAS = ("pre_1967", "dismantling_1967", "post_1967")

#: bounds (minx, miny, maxx, maxy) in EPSG:32631 -> (unstable bool array, transform) or None
UnstableProvider = Callable[[tuple[float, float, float, float]], "tuple[np.ndarray, Affine] | None"]


# --------------------------------------------------------------------------- naming


def camera_short(camera: str | None, dataset: str | None = None) -> str:
    """kh4 | kh4a | kh4b | kh7 | kh9map | kh9pan | argon | lanyard, else ``hist``."""
    return _CAMERA_SHORT.get(match_system(camera, dataset) or "", "hist")


def historical_layer_name(row: Mapping | pd.Series) -> str:
    """Shared layer-naming convention: ``{camera_short}_{entity_id}`` lower-case."""
    return f"{camera_short(row.get('camera'), row.get('dataset'))}_{row['entity_id']}".lower()


def pair_layer_name(name_a: str, name_b: str) -> str:
    return f"pair_{name_a}_{name_b}"


# --------------------------------------------------------------------------- sources & pairs


@dataclass
class LayerSource:
    """One dated raster: a georeferenced frame (band 1) or a modern layer (mean of ``bands``)."""

    name: str
    path: Path
    kind: str  # "historical" | "modern"
    era: str
    entity_id: str = ""
    date: pd.Timestamp | None = None
    rmse_m: float = 0.0
    bands: tuple[int, ...] = (1,)
    _info: tuple | None = field(default=None, repr=False, compare=False)

    def info(self) -> tuple[tuple[float, float, float, float], float]:
        """(bounds, pixel size in metres)."""
        if self._info is None:
            with rasterio.open(self.path) as src:
                b = src.bounds
                self._info = ((b.left, b.bottom, b.right, b.top), abs(src.transform.a))
        return self._info

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.info()[0]

    @property
    def res(self) -> float:
        return self.info()[1]


def _rmse_of(row: Mapping) -> float:
    for k in ("cv_rmse_m", "checkpoint_rmse_m"):  # cv_rmse_m only exists after spatial validation
        v = row.get(k)
        if v is not None and not pd.isna(v):
            return float(v)
    return 0.0


def frame_sources(registry: pd.DataFrame, settings: Settings | None = None) -> list[LayerSource]:
    """Included registry frames as `LayerSource`s (files that do not exist are skipped)."""
    s = settings or load_settings()
    out: list[LayerSource] = []
    reg = registry
    if "included_for_change_detection" in reg.columns:
        reg = reg[reg["included_for_change_detection"].astype(bool)]
    for row in reg.to_dict("records"):
        gf = str(row.get("georef_file") or "")
        if not gf:
            continue
        p = Path(gf)
        p = p if p.is_absolute() else s.root / p
        if not p.exists():
            log.warning("georef file for %s missing; frame skipped", row["entity_id"])
            continue
        date = pd.to_datetime(row.get("acquisition_date"), errors="coerce")
        out.append(
            LayerSource(
                name=historical_layer_name(row),
                path=p,
                kind="historical",
                era=str(row.get("era") or "undated"),
                entity_id=str(row["entity_id"]),
                date=None if pd.isna(date) else date,
                rmse_m=_rmse_of(row),
            )
        )
    return out


def load_modern_layers(
    aoi_id: str, settings: Settings | None = None, names: Sequence[str] = (MODERN_LAYER,)
) -> dict[str, LayerSource]:
    """Modern layers from ``data/processed/modern/{aoi}/stack.json``; S2 uses pan = mean(B2-B4)."""
    import json

    s = settings or load_settings()
    d = s.data_processed / "modern" / aoi_id
    mf = d / "stack.json"
    if not mf.exists():
        raise FileNotFoundError(f"modern stack for {aoi_id} not built ({mf})")
    layers = json.loads(mf.read_text())["layers"]
    out: dict[str, LayerSource] = {}
    for n in names:
        if n not in layers:
            continue
        info = layers[n]
        bnames = [b["name"] for b in info.get("bands", [])]
        idx = tuple(bnames.index(b) + 1 for b in PAN_BANDS if b in bnames) or (1,)
        out[n] = LayerSource(n, d / info["file"], "modern", "modern", bands=idx)
    return out


@dataclass(frozen=True)
class PairPolicy:
    min_overlap_m2: float = 250_000.0  # 0.25 km2
    same_era: bool = False
    historical_vs_modern: bool = True


@dataclass(frozen=True)
class Pair:
    a: LayerSource  # earlier
    b: LayerSource  # later (modern layer for historical-vs-modern)
    overlap_m2: float

    @property
    def name(self) -> str:
        return pair_layer_name(self.a.name, self.b.name)


def _overlap(a: tuple, b: tuple) -> tuple[float, float, float, float] | None:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _area(b: tuple | None) -> float:
    return 0.0 if b is None else (b[2] - b[0]) * (b[3] - b[1])


def select_pairs(
    registry: pd.DataFrame | Sequence[LayerSource],
    modern_layers: Mapping[str, LayerSource] | None = None,
    policy: PairPolicy | None = None,
    settings: Settings | None = None,
    aoi: str | None = None,
) -> list[Pair]:
    """Pairs to score, deterministic order (by A name, then B name).

    ``registry`` is the frame registry DataFrame (only included frames are used) or ready
    `LayerSource`s. Priority pairs: (pre_1967, post_1967), (pre_1967, dismantling_1967) and each
    historical frame vs the modern ``s2_median``. ``policy.same_era`` adds pairs inside one era.
    Undated frames only pair with the modern layer.
    """
    policy = policy or PairPolicy()
    if isinstance(registry, pd.DataFrame):
        reg = registry
        if aoi and "aoi" in reg.columns:
            reg = reg[reg["aoi"].astype(str).isin([aoi, ""])]
        frames = frame_sources(reg, settings)
    else:
        frames = list(registry)
    frames.sort(key=lambda f: (f.date if f.date is not None else pd.Timestamp.max, f.name))
    pairs: list[Pair] = []
    for fa, fb in itertools.combinations(frames, 2):
        ea, eb = fa.era, fb.era
        ok = (ea == "pre_1967" and eb in ("post_1967", "dismantling_1967")) or (
            ea in ("post_1967", "dismantling_1967") and eb == "pre_1967"
        )
        if not ok and policy.same_era and ea == eb and ea in HISTORICAL_ERAS:
            ok = True
        if not ok:
            continue
        if ea != "pre_1967" and eb == "pre_1967":
            fa, fb = fb, fa  # earlier first
        area = _area(_overlap(fa.bounds, fb.bounds))
        if area >= policy.min_overlap_m2:
            pairs.append(Pair(fa, fb, area))
    if policy.historical_vs_modern and modern_layers and MODERN_LAYER in modern_layers:
        m = modern_layers[MODERN_LAYER]
        for f in frames:
            area = _area(_overlap(f.bounds, m.bounds))
            if area >= policy.min_overlap_m2:
                pairs.append(Pair(f, m, area))
    pairs.sort(key=lambda p: (p.a.name, p.b.name))
    return pairs


# --------------------------------------------------------------------------- co-registration


def _read_gray(src: LayerSource, bounds: tuple) -> tuple[np.ndarray, Affine]:
    """Windowed read of the pixels covering ``bounds``; multi-band = mean (NaN-aware)."""
    with rasterio.open(src.path) as ds:
        win = from_bounds(*bounds, transform=ds.transform)
        c0, r0 = math.floor(win.col_off), math.floor(win.row_off)
        c1, r1 = math.ceil(win.col_off + win.width), math.ceil(win.row_off + win.height)
        win = Window(c0, r0, c1 - c0, r1 - r0).intersection(Window(0, 0, ds.width, ds.height))
        arr = ds.read(list(src.bands), window=win, masked=True).astype("float32").filled(np.nan)
        tr = ds.window_transform(win)
    gray = arr[0] if arr.shape[0] == 1 else np.nanmean(arr, axis=0).astype("float32")
    return gray, tr


def _resample(
    arr: np.ndarray, src_tr: Affine, shape: tuple[int, int], dst_tr: Affine, method: Resampling
) -> np.ndarray:
    dst = np.full(shape, np.nan, dtype="float32")
    reproject(
        arr.astype("float32"),
        dst,
        src_transform=src_tr,
        src_crs=CRS,
        dst_transform=dst_tr,
        dst_crs=CRS,
        resampling=method,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def common_grid(
    bounds: tuple, res: float, origin: tuple[float, float]
) -> tuple[Affine, tuple[int, int]]:
    """Pixel grid of size ``res`` inside ``bounds``, lattice-aligned to ``origin``."""
    ox, oy = origin
    x0, y0, x1, y1 = bounds
    xs = ox + math.ceil((x0 - ox) / res - 1e-6) * res
    yt = oy + math.floor((y1 - oy) / res + 1e-6) * res
    ncol = math.floor((x1 - xs) / res + 1e-6)
    nrow = math.floor((yt - y0) / res + 1e-6)
    if ncol < 2 or nrow < 2:
        raise ValueError("pair footprints do not overlap on a common grid")
    return from_origin(xs, yt, res, res), (nrow, ncol)


def _support_loader() -> Callable | None:
    import sbf.georef as g

    return getattr(g, "load_support", None)  # added by the spatial-validation work


def _support_on_grid(
    src: LayerSource, shape: tuple[int, int], tr: Affine, loader: Callable | None
) -> np.ndarray | None:
    """Boolean support (True = inside GCP support) on the pair grid, or None when unavailable."""
    if loader is None or not src.entity_id:
        return None
    try:
        sup = loader(src.entity_id)
    except Exception as e:  # noqa: BLE001 - support is optional
        log.warning("support mask for %s unreadable (%s)", src.name, type(e).__name__)
        return None
    if sup is None:
        return None
    arr, stf = sup
    out = _resample(np.asarray(arr).astype("float32"), stf, shape, tr, Resampling.nearest)
    return np.nan_to_num(out, nan=0.0) > 0.5


def _quantile_match(b: np.ndarray, a: np.ndarray, stable: np.ndarray, n_q: int = 256) -> np.ndarray:
    """Map B's grey levels onto A's distribution using quantiles of the stable-area pixels."""
    va = a[stable & np.isfinite(a) & np.isfinite(b)]
    vb = b[stable & np.isfinite(a) & np.isfinite(b)]
    if va.size < 50:
        log.warning("histogram matching skipped: only %d stable pixels", va.size)
        return b
    q = np.linspace(0, 100, n_q)
    qa, qb = np.percentile(va, q), np.percentile(vb, q)
    qb, ix = np.unique(qb, return_index=True)  # strictly increasing for interp
    return np.interp(b, qb, qa[ix]).astype("float32")


@dataclass
class PreparedPair:
    name: str
    a: np.ndarray  # CLAHE'd, NaN outside footprint/support
    b: np.ndarray  # CLAHE'd and histogram-matched to ``a``
    transform: Affine
    res: float
    unstable: np.ndarray  # bool, same grid
    shift_px: int  # misregistration search radius r
    grid_tag: str

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        h, w = self.a.shape
        t = self.transform
        return (t.c, t.f + t.e * h, t.c + t.a * w, t.f)


def shift_radius(rmse_a: float, rmse_b: float, res: float, cap: int = MAX_SHIFT_PX) -> int:
    """r = ceil(max(RMSE) / pixel), capped (default 3)."""
    return int(min(cap, max(0, math.ceil(max(rmse_a, rmse_b) / res - 1e-9))))


def prepare_pair(
    pair: Pair,
    aoi_id: str,
    settings: Settings | None = None,
    unstable_provider: UnstableProvider | None = None,
    support_loader: Callable | None | str = "auto",
    harmonise: bool = True,
) -> PreparedPair:
    """Co-register to the common grid, apply CLAHE and match B to A over stable terrain."""
    s = settings or load_settings()
    a_src, b_src = pair.a, pair.b
    ov = _overlap(a_src.bounds, b_src.bounds)
    if ov is None:
        raise ValueError(f"{pair.name}: footprints do not overlap")
    res = max(a_src.res, b_src.res)  # coarser; the finer one is block-averaged, never upsample
    tr, shape = common_grid(ov, res, grid_origin(aoi_id, s))
    loader = _support_loader() if support_loader == "auto" else support_loader
    layers = []
    for src in (a_src, b_src):
        arr, stf = _read_gray(src, ov)
        method = Resampling.average if src.res < res * 0.95 else Resampling.bilinear
        out = _resample(arr, stf, shape, tr, method)
        sup = _support_on_grid(src, shape, tr, loader)
        if sup is not None:
            out = np.where(sup, out, np.nan).astype("float32")
        layers.append(out)
    a, b = layers
    both = np.isfinite(a) & np.isfinite(b)
    a = np.where(both, a, np.nan).astype("float32")
    b = np.where(both, b, np.nan).astype("float32")
    unstable = np.zeros(shape, dtype=bool)
    if unstable_provider is not None:
        pad = 300.0  # context for the 210 m texture window
        res_u = unstable_provider((ov[0] - pad, ov[1] - pad, ov[2] + pad, ov[3] + pad))
        if res_u is not None:
            um, utf = res_u
            unstable = _resample(um.astype("float32"), utf, shape, tr, Resampling.nearest) > 0.5
    if harmonise:
        a = clahe(a)
        b = clahe(b)
        b = _quantile_match(b, a, ~unstable)
        b = np.where(both, np.clip(b, 0.0, 1.0), np.nan).astype("float32")
    tag = f"{res:.2f}m{round(tr.c)}x{round(tr.f)}"
    r = shift_radius(a_src.rmse_m, b_src.rmse_m, res)
    return PreparedPair(pair.name, a, b, tr, res, unstable, r, tag)


def stack_unstable_provider(
    aoi_id: str, settings: Settings | None = None
) -> UnstableProvider | None:
    """Dune/unstable-terrain mask from the modern stack (`stable_terrain_mask`); None if no stack."""
    from sbf.georef.mask import stable_terrain_mask
    from sbf.georef.reference import load_stack_reference

    s = settings or load_settings()

    def provider(bounds):
        ref = load_stack_reference(aoi_id, s, mode="pan", bounds_xy=bounds)
        stable = stable_terrain_mask(ref.mask_stack())
        return (~stable & np.isfinite(ref.array)), ref.transform

    if not (s.data_processed / "modern" / aoi_id / "stack.json").exists():
        log.warning("no modern stack for %s: dune mask unavailable, nothing is excluded", aoi_id)
        return None
    return provider


# --------------------------------------------------------------------------- tile chips


def _tile_slices(grid, prep: PreparedPair):
    t = prep.transform
    h, w = prep.a.shape
    for tid, geom in zip(grid["tile_id"], grid.geometry, strict=True):
        x0, y0, x1, y1 = geom.bounds
        n_c, n_r = round((x1 - x0) / t.a), round((y1 - y0) / -t.e)
        c0, r0 = round((x0 - t.c) / t.a), round((t.f - y1) / -t.e)
        if c0 < 0 or r0 < 0 or c0 + n_c > w or r0 + n_r > h:
            continue
        yield tid, (slice(r0, r0 + n_r), slice(c0, c0 + n_c))


def _pair_tiles(prep: PreparedPair, aoi_id: str, size_m: int, s: Settings):
    """Tile IDs with their slices; drops unstable and unsupported tiles. Returns (kept, counts)."""
    grid = make_grid(aoi_id, size_m, s.overlap, s, bounds=prep.bounds)
    kept, n_unstable, n_nosupport, total = [], 0, 0, 0
    for tid, sl in _tile_slices(grid, prep):
        total += 1
        ca, cb = prep.a[sl], prep.b[sl]
        if float((np.isfinite(ca) & np.isfinite(cb)).mean()) < MIN_VALID_FRAC:
            n_nosupport += 1
        elif float(prep.unstable[sl].mean()) > MAX_UNSTABLE_FRAC:
            n_unstable += 1
        else:
            kept.append((tid, sl))
    log.info(
        "%s size=%d: %d tiles, kept %d, dropped %d unstable, %d outside support",
        prep.name,
        size_m,
        total,
        len(kept),
        n_unstable,
        n_nosupport,
    )
    return kept, {"tiles": total, "unstable": n_unstable, "no_support": n_nosupport}


# --------------------------------------------------------------------------- change_diff


def _ssim_win(side: int) -> int:
    w = min(7, side)
    return w if w % 2 else w - 1


def tile_diff_metrics(a: np.ndarray, b: np.ndarray, r: int = 0) -> tuple[float, float]:
    """(min over shifts of normalised-input absolute difference, min over shifts of 1 - SSIM).

    ``b`` is compared with the central part of ``a`` after being shifted by every (dy, dx) in
    [-r, r]^2; each metric keeps its minimum. NaN pixels are ignored (filled identically in both
    so they contribute nothing to SSIM). ``(nan, nan)`` when the tile is too small.
    """
    from skimage.metrics import structural_similarity

    n = a.shape[0]
    side = n - 2 * r
    if side < 3:
        return float("nan"), float("nan")
    ac = a[r : n - r, r : n - r]
    win = _ssim_win(side)
    best_abs, best_ssim = math.inf, math.inf
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            bs = b[r + dy : n - r + dy, r + dx : n - r + dx]
            ok = np.isfinite(ac) & np.isfinite(bs)
            if ok.mean() < MIN_VALID_FRAC:
                continue
            absd = float(np.abs(ac - bs)[ok].mean())
            fill = float(np.median(ac[ok]))
            fa, fb = np.where(ok, ac, fill), np.where(ok, bs, fill)
            d = 1.0 - float(structural_similarity(fa, fb, win_size=win, data_range=1.0))
            best_abs, best_ssim = min(best_abs, absd), min(best_ssim, d)
    if math.isinf(best_abs):
        return float("nan"), float("nan")
    return best_abs, best_ssim


def _rank01(x: np.ndarray) -> np.ndarray:
    return (rankdata(x, method="average") - 1) / max(len(x) - 1, 1)


def diff_scores(
    prep: PreparedPair,
    aoi_id: str,
    size_m: int = 64,
    settings: Settings | None = None,
    shift_px: int | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """``change_diff`` table for one prepared pair (layer = pair name).

    score = mean of within-pair ranks of the shift-min (1 - SSIM) and the shift-min normalised
    absolute difference (abs diff divided by the pair-wide std of A). ``shift_px`` overrides the
    search radius (0 disables the shift search).
    """
    s = settings or load_settings()
    r = prep.shift_px if shift_px is None else int(shift_px)
    kept, _ = _pair_tiles(prep, aoi_id, size_m, s)
    sd = float(np.nanstd(prep.a)) or 1.0
    ids, m_abs, m_ssim = [], [], []
    for tid, sl in kept:
        ab, ss = tile_diff_metrics(prep.a[sl], prep.b[sl], r)
        if np.isfinite(ab) and np.isfinite(ss):
            ids.append(tid)
            m_abs.append(ab / sd)
            m_ssim.append(ss)
    if not ids:
        return _empty()
    score = (_rank01(np.array(m_ssim)) + _rank01(np.array(m_abs))) / 2.0
    df = pd.DataFrame(
        {
            "tile_id": ids,
            "aoi_id": aoi_id,
            "size_m": int(size_m),
            "detector": "change_diff",
            "layer": prep.name,
            "score": score,
        }
    )
    if write:
        write_scores(df, s)
    return df


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["tile_id", "aoi_id", "size_m", "detector", "layer", "score", "score_norm"]
    )


# --------------------------------------------------------------------------- change_embed


def _embed_side(
    prep: PreparedPair,
    arr: np.ndarray,
    cache_layer: str,
    kept: list,
    encoder: TileEncoder,
    aoi_id: str,
    size_m: int,
    s: Settings,
    batch_size: int,
) -> dict[str, np.ndarray]:
    folder = embedding_dir(aoi_id, cache_layer, size_m, encoder.name, s)
    done = cached_ids(folder)
    todo = [(t, sl) for t, sl in kept if t not in done]
    for i in range(0, len(todo), 512):
        part = todo[i : i + 512]
        embs = [
            np.asarray(encoder.encode([arr[sl] for _, sl in part[j : j + batch_size]]))
            for j in range(0, len(part), batch_size)
        ]
        _write_shard(folder, [t for t, _ in part], np.concatenate(embs).astype("float32"))
    ids, X = load_embeddings(folder)
    return dict(zip(ids, X, strict=True))


def embed_scores(
    prep: PreparedPair,
    aoi_id: str,
    encoder: TileEncoder,
    size_m: int = 128,
    settings: Settings | None = None,
    name_a: str = "a",
    name_b: str = "b",
    batch_size: int = 32,
    write: bool = True,
) -> pd.DataFrame:
    """``change_embed`` table: cosine distance between A and B tile embeddings.

    Embeddings use the shared cache (`sbf.detect.embedding.embedding_dir`); the cache layer key
    includes the pair grid (resolution + origin), and B's key also names A because B is
    histogram-matched to A. No shift search (see module docstring): prefer 128 m tiles.
    """
    s = settings or load_settings()
    kept, _ = _pair_tiles(prep, aoi_id, size_m, s)
    if not kept:
        return _empty()
    ea = _embed_side(
        prep, prep.a, f"{name_a}-g{prep.grid_tag}", kept, encoder, aoi_id, size_m, s, batch_size
    )
    eb = _embed_side(
        prep,
        prep.b,
        f"{name_b}-g{prep.grid_tag}-hm{name_a}",
        kept,
        encoder,
        aoi_id,
        size_m,
        s,
        batch_size,
    )
    ids = [t for t, _ in kept if t in ea and t in eb]
    A = np.stack([ea[t] for t in ids]).astype("float64")
    B = np.stack([eb[t] for t in ids]).astype("float64")
    cos = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)
    df = pd.DataFrame(
        {
            "tile_id": ids,
            "aoi_id": aoi_id,
            "size_m": int(size_m),
            "detector": "change_embed",
            "layer": prep.name,
            "score": 1.0 - cos,
        }
    )
    if write:
        write_scores(df, s)
    return df


# --------------------------------------------------------------------------- orchestration


def run_change(
    aoi_id: str,
    pairs: Sequence[Pair],
    do_diff: bool = True,
    do_embed: bool = True,
    diff_size: int = 64,
    embed_size: int = 128,
    encoder: TileEncoder | None = None,
    settings: Settings | None = None,
    unstable_provider: UnstableProvider | None = None,
) -> dict[str, dict[str, int]]:
    """Score every pair; returns ``{pair: {"diff": n_tiles, "embed": n_tiles}}`` (counts only)."""
    s = settings or load_settings()
    out: dict[str, dict[str, int]] = {}
    for p in pairs:
        prep = prepare_pair(p, aoi_id, s, unstable_provider)
        res: dict[str, int] = {}
        if do_diff:
            res["diff"] = len(diff_scores(prep, aoi_id, diff_size, s))
        if do_embed:
            if encoder is None:
                raise ValueError("an encoder is required for change_embed")
            res["embed"] = len(
                embed_scores(prep, aoi_id, encoder, embed_size, s, p.a.name, p.b.name)
            )
        out[p.name] = res
        log.info("%s: %s", p.name, res)
    return out
