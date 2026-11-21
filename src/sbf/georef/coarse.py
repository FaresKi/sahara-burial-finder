"""Step 1: coarse placement of a frame from its footprint corners.

USGS footprints list four corners with no guarantee about which is the frame's top-left, nor
about winding direction. We sort the corners into a canonical cyclic order (by angle around the
centroid) and then generate the 8 dihedral variants (4 rotations x optional mirror). Each gives a
homography frame-pixel -> UTM xy; the caller picks the variant with the most RANSAC inliers
(`gcps.find_gcps`). Pixel coordinates use the "corner" convention (pixel edges at integers,
centres at +0.5), like rasterio affines.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from pyproj import Transformer

from sbf.georef.models import UTM_CRS


@dataclass(frozen=True)
class CoarseCandidate:
    variant: int  # 0..7
    H: np.ndarray  # 3x3, frame px (corner conv) -> xy
    corners_xy: np.ndarray  # (4, 2) xy of frame TL, TR, BR, BL under this variant


def footprint_corners_xy(corners_lonlat, crs: str = UTM_CRS) -> np.ndarray:
    """(n, 2) lon/lat -> (n, 2) xy in `crs`. Rings with a repeated closing vertex are accepted."""
    c = np.asarray(corners_lonlat, dtype="float64")
    if len(c) == 5 and np.allclose(c[0], c[-1]):
        c = c[:-1]
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError("corners must be (n, 2) lon/lat")
    if len(c) != 4:
        # Not a quadrilateral: fall back to the minimum rotated rectangle of the vertices.
        from shapely.geometry import MultiPoint

        rect = MultiPoint([tuple(p) for p in c]).minimum_rotated_rectangle
        c = np.asarray(rect.exterior.coords)[:4]
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(c[:, 0], c[:, 1])
    return np.column_stack([x, y])


def canonical_order(xy: np.ndarray) -> np.ndarray:
    """Corners sorted counter-clockwise around their centroid (robust to bow-tie input)."""
    ctr = xy.mean(axis=0)
    ang = np.arctan2(xy[:, 1] - ctr[1], xy[:, 0] - ctr[0])
    return xy[np.argsort(ang)]


def quad_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def coarse_candidates(
    frame_shape: tuple[int, int], corners_xy: np.ndarray
) -> list[CoarseCandidate]:
    """The 8 candidate homographies (frame px -> xy) for an unordered footprint quad."""
    h, w = frame_shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float64")
    ring = canonical_order(np.asarray(corners_xy, dtype="float64"))
    out = []
    for flip in (False, True):
        base = ring[::-1] if flip else ring
        for k in range(4):
            dst = np.roll(base, -k, axis=0)
            # float32 loses precision at UTM magnitudes: solve in float64 around the centroid.
            c0 = dst.mean(axis=0)
            Hc = cv2.getPerspectiveTransform(src.astype("float32"), (dst - c0).astype("float32"))
            T = np.array([[1, 0, c0[0]], [0, 1, c0[1]], [0, 0, 1]], dtype="float64")
            H = T @ Hc.astype("float64")
            out.append(CoarseCandidate(variant=int(flip) * 4 + k, H=H, corners_xy=dst))
    return out


def coarse_transform(
    frame_shape: tuple[int, int],
    footprint_corners_lonlat,
    variant: int = 0,
    crs: str = UTM_CRS,
) -> np.ndarray:
    """Homography frame px -> `crs` xy for one corner-ordering `variant` (0..7)."""
    cands = coarse_candidates(frame_shape, footprint_corners_xy(footprint_corners_lonlat, crs))
    return cands[variant].H


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype="float64")
    q = np.column_stack([p, np.ones(len(p))]) @ H.T
    return q[:, :2] / q[:, 2:3]


def ground_resolution_m(frame_shape: tuple[int, int], corners_xy: np.ndarray) -> float:
    """Mean ground pixel size implied by the footprint area."""
    h, w = frame_shape[:2]
    return float(np.sqrt(quad_area(np.asarray(corners_xy)) / (h * w)))
