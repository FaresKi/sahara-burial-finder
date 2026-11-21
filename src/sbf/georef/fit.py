"""Step 5: transform models (pixel -> xy), own least-squares polynomials, TPS, selection.

`fit_transform(px, xy, method)` is the reusable seam (per frame or per strip). A model maps
frame pixels (corner convention) to map xy and is callable on (n, 2) arrays.

Overfit guard: a polynomial of order k has T = (k+1)(k+2)/2 coefficients per axis (each axis is
an independent regression); it is only eligible when n_train >= 2*T (poly1 >= 6, poly2 >= 12,
poly3 >= 20). TPS needs >= 10 points. `select_model` splits off a seeded 20% checkpoint set,
fits every eligible candidate on the rest, and picks the lowest checkpoint RMSE, preferring
simpler models unless a more complex one is better by more than 5% (IMPROVEMENT).

Honest accuracy (ticket 08 follow-up): a random 20% hold-out puts checkpoints next to training
GCPs, so it only measures local consistency (a frame-wide fit scored 4 m yet was ~85 m wrong where
GCPs were sparse). Default is now a SPATIALLY BLOCKED hold-out: the frame (or strip) is cut into a
k x k grid (k <= 5, shrunk until enough blocks hold GCPs) and whole blocks totalling ~20% of the
GCPs are held out. `cv_rmse_m` (spatial 5-fold over blocks, pooled) is the primary accuracy number;
the random hold-out is kept only as `random_check_rmse_m` for comparison. Candidate models are
ranked by their CV RMSE (blocked checkpoint RMSE when CV is not computable).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RBFInterpolator

METHODS = ("poly1", "poly2", "poly3", "tps")
IMPROVEMENT = 0.95
TPS_MIN_POINTS = 10
CHECK_FRACTION = 0.2
GRID_K = 5  # default blocks per side for the hold-out / CV / coverage grid
CV_FOLDS = 5
OCC_MIN_GCPS = 3  # a block counts as occupied (coverage metric) with at least this many GCPs


def n_terms(order: int) -> int:
    return (order + 1) * (order + 2) // 2


def min_points(method: str) -> int:
    """Minimum number of TRAIN points for a method (overfit guard)."""
    if method == "tps":
        return TPS_MIN_POINTS
    return 2 * n_terms(int(method[-1]))


def _design(u: np.ndarray, v: np.ndarray, order: int) -> np.ndarray:
    return np.column_stack([u**i * v**j for i in range(order + 1) for j in range(order + 1 - i)])


class PolyModel:
    """Total-degree polynomial in normalised pixel coordinates, independent per output axis."""

    def __init__(self, order: int, coef: np.ndarray, centre, scale: float, offset):
        self.order, self.coef, self.centre, self.scale, self.offset = (
            order,
            coef,
            np.asarray(centre),
            scale,
            np.asarray(offset),
        )
        self.method = f"poly{order}"

    @classmethod
    def fit(cls, px, xy, order: int) -> PolyModel:
        px, xy = np.asarray(px, float), np.asarray(xy, float)
        centre = px.mean(axis=0)
        scale = max(float(np.abs(px - centre).max()), 1e-9)
        n = (px - centre) / scale
        offset = xy.mean(axis=0)
        coef, *_ = np.linalg.lstsq(_design(n[:, 0], n[:, 1], order), xy - offset, rcond=None)
        return cls(order, coef, centre, scale, offset)

    def __call__(self, px) -> np.ndarray:
        n = (np.asarray(px, float) - self.centre) / self.scale
        return _design(n[:, 0], n[:, 1], self.order) @ self.coef + self.offset


class TpsModel:
    """Thin-plate spline (scipy RBFInterpolator) pixel -> xy. `smoothing` 0 interpolates."""

    method = "tps"

    def __init__(self, px, xy, smoothing: float = 0.0):
        px, xy = np.asarray(px, float), np.asarray(xy, float)
        self.centre = px.mean(axis=0)
        self.scale = max(float(np.abs(px - self.centre).max()), 1e-9)
        self.offset = xy.mean(axis=0)
        self.smoothing = smoothing
        self._rbf = RBFInterpolator(
            (px - self.centre) / self.scale,
            xy - self.offset,
            kernel="thin_plate_spline",
            smoothing=smoothing,
        )

    def __call__(self, px) -> np.ndarray:
        return self._rbf((np.asarray(px, float) - self.centre) / self.scale) + self.offset


TransformModel = PolyModel | TpsModel


def fit_transform(px, xy, method: str, *, smoothing: float = 0.0) -> TransformModel:
    """Fit one model. Raises ValueError if there are too few points for `method`."""
    px, xy = np.asarray(px, float), np.asarray(xy, float)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if len(px) < min_points(method):
        raise ValueError(f"{method} needs >= {min_points(method)} points, got {len(px)}")
    if method == "tps":
        return TpsModel(px, xy, smoothing)
    return PolyModel.fit(px, xy, int(method[-1]))


def residuals_m(model: TransformModel, px, xy) -> np.ndarray:
    """Euclidean distance (map units, metres in UTM) between model(px) and xy."""
    return np.hypot(*(model(px) - np.asarray(xy, float)).T)


def rmse(res: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(res)))) if len(res) else float("nan")


def split_checkpoints(
    n: int,
    frac: float = CHECK_FRACTION,
    seed: int = 0,
    *,
    px=None,
    extent: tuple[float, float, float, float] | None = None,
    strategy: str = "blocked",
    k: int = GRID_K,
):
    """Seeded (train_idx, check_idx). `strategy` "random": size round(frac*n), at least 1 when
    n >= 2. "blocked" (default; needs `px`, falls back to random without it): whole grid blocks are
    held out until ~frac*n GCPs, so checkpoints are spatially separated from training points.
    `extent` = (x0, y0, x1, y1) px of the frame/strip (default: bounding box of `px`)."""
    if strategy not in ("blocked", "random"):
        raise ValueError("strategy must be 'blocked' or 'random'")
    if strategy == "blocked" and px is not None and n >= 4:
        out = _blocked_split(np.asarray(px, float), frac, seed, extent, k)
        if out is not None:
            return out
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    kk = min(max(round(frac * n), 1), n - 1) if n >= 2 else 0
    return np.sort(perm[kk:]), np.sort(perm[:kk])


def _extent_of(px: np.ndarray, extent):
    if extent is not None:
        return tuple(float(v) for v in extent)
    return (*px.min(axis=0), *px.max(axis=0))


def block_ids(px, extent, k: int) -> np.ndarray:
    """Block index (row-major, 0..k*k-1) of each point on a k x k grid over `extent`."""
    px = np.asarray(px, float)
    x0, y0, x1, y1 = _extent_of(px, extent)
    cx = np.clip(((px[:, 0] - x0) / max(x1 - x0, 1e-9) * k).astype(int), 0, k - 1)
    cy = np.clip(((px[:, 1] - y0) / max(y1 - y0, 1e-9) * k).astype(int), 0, k - 1)
    return cy * k + cx


def choose_k(px, extent, k: int = GRID_K, min_blocks: int = 5) -> int:
    """Largest k' <= k (>= 2) whose grid has at least `min_blocks` occupied blocks."""
    for kk in range(k, 1, -1):
        if len(np.unique(block_ids(px, extent, kk))) >= min_blocks:
            return kk
    return 2


def _blocked_split(px, frac, seed, extent, k):
    n = len(px)
    ids = block_ids(px, extent, choose_k(px, extent, k))
    blocks = np.unique(ids)
    if len(blocks) < 2:
        return None
    rng = np.random.default_rng(seed)
    target = max(round(frac * n), 1)
    held: list[int] = []
    count = 0
    for b in rng.permutation(blocks):
        size = int((ids == b).sum())
        if count >= target:
            break
        if count + size > max(2 * target, target + 2) and held:
            continue  # would overshoot; try a smaller block
        held.append(int(b))
        count += size
    chk = np.isin(ids, held)
    if chk.all() or not chk.any():
        return None
    return np.flatnonzero(~chk), np.flatnonzero(chk)


def spatial_folds(
    px, extent=None, k: int = GRID_K, n_folds: int = CV_FOLDS, seed: int = 0
) -> list[np.ndarray]:
    """Spatial k-fold: occupied blocks are shuffled (seeded) and dealt round-robin into at most
    `n_folds` folds; each fold's GCP indices are returned (whole blocks never straddle folds)."""
    px = np.asarray(px, float)
    ids = block_ids(px, extent, choose_k(px, extent, k))
    blocks = np.random.default_rng(seed).permutation(np.unique(ids))
    nf = min(n_folds, len(blocks))
    if nf < 2:
        return []
    return [np.flatnonzero(np.isin(ids, blocks[f::nf])) for f in range(nf)]


def coverage_metrics(px, extent, k: int = GRID_K, min_gcps: int = OCC_MIN_GCPS):
    """(gcp_hull_fraction, gcp_block_occupancy) in frame pixel space.

    hull fraction = convex-hull area of the GCPs / area of `extent`; occupancy = fraction of the
    k x k grid of `extent` blocks that contain >= `min_gcps` GCPs."""
    px = np.asarray(px, float)
    x0, y0, x1, y1 = extent
    area = max((x1 - x0) * (y1 - y0), 1e-9)
    hull = 0.0
    if len(px) >= 3:
        from scipy.spatial import ConvexHull, QhullError

        try:
            hull = float(ConvexHull(px).volume) / area
        except QhullError:  # collinear / degenerate
            hull = 0.0
    occ = 0.0
    if len(px):
        cnt = np.bincount(block_ids(px, extent, k), minlength=k * k)
        occ = float((cnt >= min_gcps).mean())
    return min(hull, 1.0), occ


def cv_residuals(px, xy, method: str, folds, *, smoothing: float = 0.0) -> np.ndarray:
    """Pooled out-of-fold residuals (m). Folds whose training set is too small for `method` are
    skipped; empty array when no fold is usable."""
    out = []
    n = len(px)
    for f in folds:
        train = np.setdiff1d(np.arange(n), f)
        if len(train) < min_points(method) or len(f) == 0:
            continue
        model = fit_transform(px[train], xy[train], method, smoothing=smoothing)
        out.append(residuals_m(model, px[f], xy[f]))
    return np.concatenate(out) if out else np.array([])


@dataclass
class Selection:
    method: str
    model: TransformModel  # fitted on the train split only (the one whose RMSE is reported)
    train_idx: np.ndarray
    check_idx: np.ndarray
    train_rmse_m: float
    check_rmse_m: float  # blocked hold-out (strategy="blocked") or random
    candidate_rmse_m: dict[str, float | None] = field(default_factory=dict)
    cv_rmse_m: float | None = None  # spatial k-fold, primary accuracy number
    cv_res: np.ndarray = field(default_factory=lambda: np.array([]))  # pooled out-of-fold residuals
    random_check_rmse_m: float | None = None  # random 20% hold-out, for comparison only
    candidate_cv_rmse_m: dict[str, float | None] = field(default_factory=dict)


def select_model(
    px,
    xy,
    method: str = "auto",
    *,
    seed: int = 0,
    check_fraction: float = CHECK_FRACTION,
    smoothing: float = 0.0,
    extent: tuple[float, float, float, float] | None = None,
    strategy: str = "blocked",
    k: int = GRID_K,
) -> Selection:
    """Split, fit candidates, pick by CV RMSE (checkpoint RMSE if CV is not computable).
    `method` `auto` or one of METHODS; `extent` is the frame/strip box in px (default: bbox)."""
    px, xy = np.asarray(px, float), np.asarray(xy, float)
    train, check = split_checkpoints(
        len(px), check_fraction, seed, px=px, extent=extent, strategy=strategy, k=k
    )
    folds = spatial_folds(px, extent, k, seed=seed)
    names = METHODS if method == "auto" else (method,)
    cand: dict[str, float | None] = {}
    cand_cv: dict[str, float | None] = {}
    fitted: dict[str, tuple[TransformModel, float, float, np.ndarray]] = {}
    for m in names:
        if len(train) < min_points(m) or len(check) == 0:
            cand[m] = cand_cv[m] = None
            continue
        model = fit_transform(px[train], xy[train], m, smoothing=smoothing)
        c = rmse(residuals_m(model, px[check], xy[check]))
        t = rmse(residuals_m(model, px[train], xy[train]))
        cvr = cv_residuals(px, xy, m, folds, smoothing=smoothing)
        cand[m], fitted[m] = c, (model, t, c, cvr)
        cand_cv[m] = rmse(cvr) if len(cvr) else None
    if not fitted:
        raise ValueError(
            f"no eligible model: {len(train)} train / {len(check)} check points "
            f"(need >= {min_points('poly1')} train for poly1)"
        )

    def score(m: str) -> float:
        return cand_cv[m] if cand_cv[m] is not None else fitted[m][2]

    best = None
    for m in names:  # simple -> complex; complex must beat the incumbent by >5%
        if m in fitted and (best is None or score(m) < score(best) * IMPROVEMENT):
            best = m
    model, t, c, cvr = fitted[best]
    rnd = None
    r_train, r_check = split_checkpoints(len(px), check_fraction, seed, strategy="random")
    if len(r_train) >= min_points(best) and len(r_check):
        rm = fit_transform(px[r_train], xy[r_train], best, smoothing=smoothing)
        rnd = rmse(residuals_m(rm, px[r_check], xy[r_check]))
    return Selection(
        best,
        model,
        train,
        check,
        t,
        c,
        cand,
        cv_rmse_m=cand_cv[best],
        cv_res=cvr,
        random_check_rmse_m=rnd,
        candidate_cv_rmse_m=cand_cv,
    )


# --------------------------------------------------------------------------- inversion


def invert_on_lattice(
    model: TransformModel,
    shape: tuple[int, int],
    n: int = 40,
    extent: tuple[float, float, float, float] | None = None,
) -> TpsModel:
    """Approximate inverse (xy -> pixel) as an exact TPS through an n x n lattice of the forward
    model over the frame (or over `extent` = (x0, y0, x1, y1) px, e.g. one strip). Used by
    `warp.warp_frame` (polynomials have no closed-form inverse)."""
    h, w = shape[:2]
    x0, y0, x1, y1 = extent if extent is not None else (0.0, 0.0, float(w), float(h))
    u, v = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    px = np.column_stack([u.ravel(), v.ravel()])
    return TpsModel(model(px), px, smoothing=0.0)
