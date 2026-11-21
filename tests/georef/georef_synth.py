"""Synthetic modern raster + distorted 'historical frame' with known ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from rasterio.transform import Affine
from scipy import ndimage

from sbf.georef.reference import ReferenceLayer

X0, Y1 = 500_000.0, 2_900_000.0  # top-left of the 4 km x 4 km scene (EPSG:32631)
SCENE_RES = 2.5
SCENE_PX = 1600


def make_scene(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((SCENE_PX, SCENE_PX))
    s = sum(ndimage.gaussian_filter(n, sg) * sg for sg in (1.5, 4, 12, 30))
    s = (s - s.min()) / (s.max() - s.min())
    # hard-edged blobs give SIFT corners
    for _ in range(250):
        r, c = rng.integers(0, SCENE_PX, 2)
        h, w = rng.integers(3, 25, 2)
        s[r : r + h, c : c + w] = rng.uniform(0.1, 0.9)
    return ndimage.gaussian_filter(s, 0.8).astype("float32")


def make_reference(scene: np.ndarray) -> ReferenceLayer:
    k = int(10 / SCENE_RES)
    n = SCENE_PX // k
    ref = scene.reshape(n, k, n, k).mean(axis=(1, 3)).astype("float32")
    return ReferenceLayer("synthetic", ref, Affine(10, 0, X0, 0, -10, Y1))


@dataclass
class Frame:
    image: np.ndarray
    truth: callable  # (n, 2) frame corner-px -> xy
    corners_lonlat: np.ndarray  # shuffled, perturbed


def make_frame(scene, *, size=500, res=5.0, angle_deg=30.0, distort_m=30.0, seed=2) -> Frame:
    rng = np.random.default_rng(seed)
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    centre = np.array([X0 + 2000, Y1 - 2000])

    def truth(px):
        px = np.asarray(px, float)
        d = (px - size / 2) / (size / 2)  # [-1, 1]
        local = np.column_stack([d[:, 0] * size / 2 * res, -d[:, 1] * size / 2 * res])
        warp = distort_m * np.column_stack([d[:, 0] ** 2 - 0.5 * d[:, 0] * d[:, 1], d[:, 1] ** 2])
        return centre + local @ R.T + warp

    u, v = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    xy = truth(np.column_stack([u.ravel(), v.ravel()]))
    cols = (xy[:, 0] - X0) / SCENE_RES - 0.5
    rows = (Y1 - xy[:, 1]) / SCENE_RES - 0.5
    s = ndimage.map_coordinates(scene, [rows, cols], order=1).reshape(size, size)
    img = np.clip(1.8 * (s - 0.2) + 0.2, 0, 1) + rng.normal(0, 0.03, s.shape)  # saturation+noise
    img8 = np.clip(img * 255, 0, 255).astype("uint8")

    corners = truth(np.array([[0, 0], [size, 0], [size, size], [0, size]], float))
    corners = corners + rng.normal(0, 60, corners.shape)  # footprints are approximate
    order = [2, 1, 0, 3]  # wrong start + opposite winding
    to_ll = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(corners[order, 0], corners[order, 1])
    return Frame(img8, truth, np.column_stack([lon, lat]))


def make_panoramic_frame(
    scene,
    *,
    w=1200,
    h=420,
    k=800.0,
    theta_max_deg=65.0,
    res_y=3.0,
    bow_m=120.0,
    angle_deg=10.0,
    seed=3,
) -> Frame:
    """Scan-arm-like frame: ground X = k*tan(theta) with theta linear in column (scale ~ 1/cos^2),
    plus along-track bowing bow*u^2 and a small rotation. Known ground truth in `truth`."""
    rng = np.random.default_rng(seed)
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    centre = np.array([X0 + 2000, Y1 - 2000])
    tmax = np.radians(theta_max_deg)

    def truth(px):
        px = np.asarray(px, float)
        u = (px[:, 0] - w / 2) / (w / 2)
        v = (px[:, 1] - h / 2) / (h / 2)
        local = np.column_stack([k * np.tan(u * tmax), -v * h / 2 * res_y + bow_m * u**2])
        return centre + local @ R.T

    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    xy = truth(np.column_stack([uu.ravel(), vv.ravel()]))
    cols = (xy[:, 0] - X0) / SCENE_RES - 0.5
    rows = (Y1 - xy[:, 1]) / SCENE_RES - 0.5
    s = ndimage.map_coordinates(scene, [rows, cols], order=1).reshape(h, w)
    img = np.clip(1.8 * (s - 0.2) + 0.2, 0, 1) + rng.normal(0, 0.03, s.shape)
    img8 = np.clip(img * 255, 0, 255).astype("uint8")
    corners = truth(np.array([[0, 0], [w, 0], [w, h], [0, h]], float))
    corners = corners + rng.normal(0, 30, corners.shape)
    order = [2, 1, 0, 3]
    to_ll = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(corners[order, 0], corners[order, 1])
    return Frame(img8, truth, np.column_stack([lon, lat]))
