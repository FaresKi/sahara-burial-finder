"""Steps 2 and 4: preprocess, match frame vs reference, RANSAC -> GCPs.

Design: the (CLAHE'd, block-averaged) frame is warped by each coarse candidate into the
reference grid, so frame and reference are already roughly aligned and a scale/rotation
sensitive matcher only has to find local residuals. Matches are mapped back to full-res frame
pixels through the inverse coarse homography. All pixel coordinates handed around are in the
"corner" convention (centre = +0.5); cv2 uses centre = integer, converted at the boundary.

`Matcher` is a Protocol so a learned matcher (LoFTR) can be dropped in later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from rasterio.transform import Affine

from sbf.features.preprocess import clahe
from sbf.georef.coarse import CoarseCandidate, apply_h

MIN_INLIERS = 8
RANSAC_THRESH_PX = 5.0  # reference pixels; generous so genuine nonlinear distortion survives
RATIO = 0.8
MAX_RESIDUAL_ROT_DEG = 20.0


@dataclass
class Features:
    pts: np.ndarray  # (n, 2) cv2 pixel coords
    desc: np.ndarray | None


class Matcher(Protocol):
    name: str

    def detect(self, image: np.ndarray, mask: np.ndarray | None) -> Features: ...

    def match(self, a: Features, b: Features) -> tuple[np.ndarray, np.ndarray]:
        """Matched point arrays (pts_a, pts_b), each (m, 2) cv2 pixel coords."""
        ...


def _u8(image: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(np.asarray(image, dtype="float32"), nan=0.0)
    return np.clip(a * 255.0, 0, 255).astype("uint8")


class _CvMatcher:
    name = ""
    norm = cv2.NORM_L2

    def _make(self):  # pragma: no cover
        raise NotImplementedError

    def detect(self, image, mask=None) -> Features:
        m = None if mask is None else mask.astype("uint8") * 255
        kps, desc = self._make().detectAndCompute(_u8(image), m)
        pts = np.array([k.pt for k in kps], dtype="float64").reshape(-1, 2)
        return Features(pts, desc)

    def match(self, a: Features, b: Features):
        if a.desc is None or b.desc is None or len(a.pts) < 2 or len(b.pts) < 2:
            return np.empty((0, 2)), np.empty((0, 2))
        knn = cv2.BFMatcher(self.norm).knnMatch(a.desc, b.desc, k=2)
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < RATIO * m[1].distance]
        ia = np.array([g.queryIdx for g in good], dtype=int)
        ib = np.array([g.trainIdx for g in good], dtype=int)
        return a.pts[ia], b.pts[ib]


class SiftMatcher(_CvMatcher):
    name = "sift"

    def _make(self):
        return cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)


class OrbMatcher(_CvMatcher):
    name = "orb"
    norm = cv2.NORM_HAMMING

    def _make(self):
        return cv2.ORB_create(nfeatures=8000, fastThreshold=10)


def default_matchers() -> tuple[Matcher, ...]:
    """SIFT first, ORB as fallback when SIFT cannot reach MIN_INLIERS."""
    return (SiftMatcher(), OrbMatcher())


# --------------------------------------------------------------------------- preprocessing


def downsample_for_matching(frame: np.ndarray, factor: float) -> tuple[np.ndarray, float]:
    """Block-mean (INTER_AREA) the frame by `factor` (>= 1) and apply CLAHE.

    Returns (small_unit_image, scale) where full-res corner px = small corner px * scale.
    """
    f = np.asarray(frame, dtype="float32")
    factor = max(1.0, float(factor))
    if factor > 1.0:
        w = max(8, round(f.shape[1] / factor))
        h = max(8, round(f.shape[0] / factor))
        small = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
    else:
        small = f
    scale = f.shape[1] / small.shape[1]  # == f.shape[0]/small.shape[0] to rounding
    return clahe(small), float(scale)


_C2K = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1.0]])  # cv2 centre conv -> corner conv


def _to_cv2(M_corner: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_C2K) @ M_corner @ _C2K


def _affine3(a: Affine) -> np.ndarray:
    return np.array([[a.a, a.b, a.c], [a.d, a.e, a.f], [0, 0, 1.0]])


def warp_to_reference(
    small: np.ndarray, scale: float, H: np.ndarray, ref_transform: Affine, ref_shape
) -> tuple[np.ndarray, np.ndarray]:
    """Warp the small frame into the reference grid. Returns (image, valid_mask)."""
    S = np.diag([scale, scale, 1.0])
    M = np.linalg.inv(_affine3(ref_transform)) @ H @ S  # small corner px -> ref corner px
    Mc = _to_cv2(M)
    size = (ref_shape[1], ref_shape[0])
    img = cv2.warpPerspective(small, Mc, size, flags=cv2.INTER_LINEAR)
    ones = np.ones(small.shape, dtype="float32")
    valid = cv2.warpPerspective(ones, Mc, size, flags=cv2.INTER_NEAREST) > 0.5
    valid = cv2.erode(valid.astype("uint8"), np.ones((5, 5), "uint8")).astype(bool)
    return img, valid


# --------------------------------------------------------------------------- matching


def ransac_affine(
    pts_a: np.ndarray, pts_b: np.ndarray, thresh_px: float = RANSAC_THRESH_PX, seed: int = 0
) -> tuple[np.ndarray, np.ndarray | None]:
    """(inlier mask, 2x3 affine) of a RANSAC fit a -> b; (all False, None) if it fails."""
    n = len(pts_a)
    if n < 4:
        return np.zeros(n, dtype=bool), None
    cv2.setRNGSeed(int(seed))
    M, inl = cv2.estimateAffine2D(
        pts_a.astype("float32"),
        pts_b.astype("float32"),
        method=cv2.RANSAC,
        ransacReprojThreshold=thresh_px,
        maxIters=5000,
        confidence=0.999,
    )
    if M is None or inl is None:
        return np.zeros(n, dtype=bool), None
    return inl.ravel().astype(bool), M


def ransac_inliers(pts_a, pts_b, thresh_px: float = RANSAC_THRESH_PX, seed: int = 0):
    return ransac_affine(pts_a, pts_b, thresh_px, seed)[0]


def plausible_residual(M: np.ndarray | None) -> bool:
    """True if the residual affine left after coarse placement is small (rotation, scale)."""
    if M is None:
        return False
    rot = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    sc = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    return abs(rot) <= MAX_RESIDUAL_ROT_DEG and 0.6 <= sc <= 1.6


def dedupe(px: np.ndarray, xy: np.ndarray, min_dist: float = 2.0):
    """Drop GCPs closer than `min_dist` full-res px to an earlier one (multi-orientation SIFT)."""
    keep: list[int] = []
    seen: dict[tuple[int, int], list[int]] = {}
    for i, p in enumerate(px):
        c = (int(p[0] // min_dist), int(p[1] // min_dist))
        dup = any(
            np.hypot(*(px[j] - p)) < min_dist
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for j in seen.get((c[0] + dx, c[1] + dy), [])
        )
        if not dup:
            keep.append(i)
            seen.setdefault(c, []).append(i)
    k = np.array(keep, dtype=int)
    return px[k], xy[k]


@dataclass
class MatchOutcome:
    matcher: str
    variant: int
    px: np.ndarray  # (n, 2) full-res frame px, corner conv
    xy: np.ndarray  # (n, 2) reference xy
    inliers_per_variant: list[int]


def find_gcps(
    small: np.ndarray,
    scale: float,
    candidates: list[CoarseCandidate],
    ref_img: np.ndarray,
    ref_transform: Affine,
    ref_mask: np.ndarray | None = None,
    *,
    matchers: tuple[Matcher, ...] | None = None,
    thresh_px: float = RANSAC_THRESH_PX,
    min_inliers: int = MIN_INLIERS,
    seed: int = 0,
) -> MatchOutcome | None:
    """Try every coarse candidate with each matcher in turn; stop at the first matcher whose
    best candidate reaches `min_inliers`. The candidate with most RANSAC inliers wins (this
    resolves the footprint corner-order ambiguity). Returns None when nothing matches."""
    ref_u = np.nan_to_num(ref_img, nan=0.0).astype("float32")
    A_ref = _affine3(ref_transform)
    best_overall: MatchOutcome | None = None
    for matcher in matchers or default_matchers():
        fb = matcher.detect(ref_u, ref_mask)
        counts: list[int] = []
        best = None
        for cand in candidates:
            warped, valid = warp_to_reference(small, scale, cand.H, ref_transform, ref_u.shape)
            fa = matcher.detect(warped, valid)
            pa, pb = matcher.match(fa, fb)
            inl, M = ransac_affine(pa, pb, thresh_px, seed)
            counts.append(int(inl.sum()))
            # SIFT/ORB are rotation invariant, so inlier counts alone cannot tell a 90-degree
            # wrong corner order from the right one (and rotated detector bias shifts GCPs).
            # Candidates whose residual affine is near identity win; count breaks ties.
            key = (plausible_residual(M), counts[-1])
            if best is None or key > best[0]:
                best = (key, cand, pa[inl], pb[inl])
        (_, n), cand, pa, pb = best
        if n == 0:
            continue
        # cv2 centre conv -> corner conv, then back through the coarse homography to frame px
        a_xy = apply_h(A_ref, pa + 0.5)
        px_full = apply_h(np.linalg.inv(cand.H), a_xy)
        xy = apply_h(A_ref, pb + 0.5)
        px_full, xy = dedupe(px_full, xy)
        out = MatchOutcome(matcher.name, cand.variant, px_full, xy, counts)
        if best_overall is None or len(px_full) > len(best_overall.px):
            best_overall = out
        if n >= min_inliers:
            return out
    return best_overall
