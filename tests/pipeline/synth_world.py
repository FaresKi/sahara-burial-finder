"""A clearly SYNTHETIC raw dataset for the end-to-end pipeline test (SPEC section 8).

Everything here is fabricated for tests: entity IDs start with ``TEST_``, the "historical frames"
are distorted resamplings of a random synthetic scene (not imagery of anywhere), and the
"trenches" are drawn stripes. A small AOI named REG is placed at a real UTM location only so that
the CRS maths is realistic; reference features GZ1/GZ2 are synthetic boxes in the test project's
own config (the repository config is never modified).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import Affine
from scipy import ndimage
from shapely.geometry import box, mapping

from sbf.catalog import FixtureClient, RawStore
from sbf.catalog.search import run_search
from sbf.core.aoi import load_aois, load_reference_features
from sbf.core.config import Settings
from sbf.core.crs import lonlat_to_xy
from sbf.ingest.modern import FakeExporter, modern_grid

REPO = Path(__file__).resolve().parents[2]
RES = 2.5
SCENE_PX = 1600  # 4 km
FRAME_PX = 500
FRAME_RES = 5.0
ENTITIES = {"1962": "TEST_FRAME_1962", "1965": "TEST_FRAME_1965"}
POST_ENTITY = "TEST_FRAME_1973_NOT_DOWNLOADED"  # catalogue only: supplies the post-1967 era


@dataclass
class World:
    x0: float
    y1: float
    trench_ground: list[tuple[float, float]]  # ground centres of the two true trenches (UTM)
    decoy_ground: tuple[float, float]


def make_scene(seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((SCENE_PX, SCENE_PX))
    s = sum(ndimage.gaussian_filter(n, sg) * sg for sg in (1.5, 4, 12, 30))
    s = (s - s.min()) / (s.max() - s.min())
    for _ in range(250):  # hard-edged blobs give feature-matching corners
        r, c = rng.integers(0, SCENE_PX, 2)
        h, w = rng.integers(3, 25, 2)
        s[r : r + h, c : c + w] = rng.uniform(0.1, 0.9)
    return ndimage.gaussian_filter(s, 0.8).astype("float32")


def _frame(scene, x0, y1, *, angle_deg, distort_m, seed):
    rng = np.random.default_rng(seed)
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    centre = np.array([x0 + 2000, y1 - 2000])
    size, res = FRAME_PX, FRAME_RES

    def truth(px):
        px = np.asarray(px, float)
        d = (px - size / 2) / (size / 2)
        local = np.column_stack([d[:, 0] * size / 2 * res, -d[:, 1] * size / 2 * res])
        warp = distort_m * np.column_stack([d[:, 0] ** 2 - 0.5 * d[:, 0] * d[:, 1], d[:, 1] ** 2])
        return centre + local @ R.T + warp

    u, v = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    xy = truth(np.column_stack([u.ravel(), v.ravel()]))
    cols = (xy[:, 0] - x0) / RES - 0.5
    rows = (y1 - xy[:, 1]) / RES - 0.5
    s = ndimage.map_coordinates(scene, [rows, cols], order=1).reshape(size, size)
    img = np.clip(1.8 * (s - 0.2) + 0.2, 0, 1) + rng.normal(0, 0.03, s.shape)
    corners = truth(np.array([[0, 0], [size, 0], [size, size], [0, size]], float))
    corners = corners + rng.normal(0, 60, corners.shape)  # catalogue footprints are approximate
    return np.clip(img * 255, 0, 255), truth, corners


def _stripes(img, r0, c0, n=60):
    """Planted trench pattern: parallel dark stripes, 3 px wide every 6 px (15 m / 30 m)."""
    for k in range(0, n, 6):
        img[r0 : r0 + n, c0 + k : c0 + k + 3] = 15
    return img


def setup_project(s: Settings) -> World:
    """Write config, modern raw, frames, catalogue raw into the tmp project. Returns ground truth."""
    # --- config: small synthetic AOI + synthetic reference features (project-local copy only)
    cx, cy = lonlat_to_xy(-0.0572, 26.3117, "EPSG:32631")
    x0, y1 = round((cx - 2000) / 10) * 10.0, round((cy + 2000) / 10) * 10.0
    to_ll = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)

    def ll_box(xa, ya, xb, yb):
        pts = [to_ll.transform(x, y) for x, y in ((xa, ya), (xb, ya), (xb, yb), (xa, yb))]
        return box(
            min(p[0] for p in pts),
            min(p[1] for p in pts),
            max(p[0] for p in pts),
            max(p[1] for p in pts),
        )

    aoi = ll_box(x0 + 200, y1 - 3800, x0 + 3800, y1 - 200)
    (s.config_dir / "aois.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"aoi_id": "REG", "name": "SYNTHETIC test AOI"},
                        "geometry": mapping(aoi),
                    }
                ],
            }
        )
    )

    scene = make_scene()
    img62, _truth62, corners62 = _frame(scene, x0, y1, angle_deg=30, distort_m=30, seed=2)
    img65, truth65, corners65 = _frame(scene, x0, y1, angle_deg=22, distort_m=25, seed=5)

    # trench blocks planted ONLY in the 1965 frame (frame px, row/col of the block's top-left)
    blocks = {"GZ1": (150, 140), "GZ2": (300, 300)}
    decoy = (160, 330)  # a disturbance that is not a reference feature
    ground = []
    feats = []
    for fid, (r0, c0) in blocks.items():
        _stripes(img65, r0, c0)
        gx, gy = truth65(np.array([[c0 + 30.0, r0 + 30.0]]))[0]
        ground.append((float(gx), float(gy)))
        g = ll_box(gx - 110, gy - 110, gx + 110, gy + 110)
        feats.append(
            {
                "type": "Feature",
                "properties": {
                    "feature_id": fid,
                    "name": f"SYNTHETIC {fid}",
                    "aoi_id": "REG",
                    "precision": "iaea",
                    "expected_area_km2": None,
                    "notes": "synthetic test feature",
                },
                "geometry": mapping(g),
            }
        )
    _stripes(img65, *decoy)
    dgx, dgy = truth65(np.array([[decoy[1] + 30.0, decoy[0] + 30.0]]))[0]
    (s.config_dir / "reference_features.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats})
    )

    # --- modern raw rasters: FakeExporter, then s2_median re-made from the synthetic scene
    grid = modern_grid("REG", s)
    raw = s.data_raw / "modern" / "REG"
    FakeExporter().export_aoi("REG", raw, grid=grid)
    tr, w, h, _ = grid
    k = int(10 / RES)
    n = SCENE_PX // k
    s10 = scene.reshape(n, k, n, k).mean(axis=(1, 3)).astype("float32")
    pad = 60
    padded = np.pad(s10, pad, mode="edge")
    c0 = round((tr.c - x0) / 10) + pad
    r0 = round((y1 - tr.f) / 10) + pad
    window = padded[r0 : r0 + h, c0 : c0 + w]
    assert window.shape == (h, w)
    with rasterio.open(raw / "s2_median.tif", "r+") as dst:
        for i, scale in enumerate((0.9, 1.0, 1.1, 1.0, 0.8, 0.7), start=1):
            dst.write((0.05 + 0.3 * window * scale).astype("float32"), i)

    # --- frames (raw, un-georeferenced scans)
    for year, img in (("1962", img62), ("1965", img65)):
        eid = ENTITIES[year]
        d = s.data_raw / "declass" / eid
        d.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            d / f"{eid}.tif", "w", driver="GTiff", count=1, dtype="uint8",
            height=FRAME_PX, width=FRAME_PX,
        ) as dst:  # fmt: skip
            dst.write(img.astype("uint8"), 1)

    # --- catalogue raw responses, saved exactly as a live run would (RawStore + manifest.jsonl)
    def scene_json(eid, date, corners, mission, camera, order):
        ring = [list(to_ll.transform(x, y)) for x, y in corners]
        ring.append(ring[0])
        return {
            "entityId": eid,
            "cloudCover": 5,
            "browse": [{"browsePath": f"https://example.invalid/{eid}.jpg"}],
            "spatialCoverage": {"type": "Polygon", "coordinates": [ring]},
            "temporalCoverage": {"startDate": f"{date} 00:00:00"},
            "metadata": [
                {"fieldName": "Mission", "value": mission},
                {"fieldName": "Camera Type", "value": camera},
            ],
            "options": {"download": not order, "order": order},
        }

    cx_, cy_ = x0 + 2000, y1 - 2000
    big = [
        (cx_ - 2500, cy_ - 2500),
        (cx_ + 2500, cy_ - 2500),
        (cx_ + 2500, cy_ + 2500),
        (cx_ - 2500, cy_ + 2500),
    ]
    scenes = {
        "test_declass_1": [
            scene_json(ENTITIES["1962"], "1962-06-01", corners62, "LANYARD", "Frame", False),
            scene_json(ENTITIES["1965"], "1965-03-01", corners65, "LANYARD", "Frame", False),
        ],
        "test_declass_2": [],
        "test_declass_3": [
            scene_json(POST_ENTITY, "1973-05-01", big, "KH-9", "Mapping Camera", True)
        ],
    }
    fx = REPO / "tests" / "fixtures" / "catalog" / "dataset-search.json"

    def responder(endpoint, payload):
        if endpoint == "dataset-search":
            return json.loads(fx.read_text())
        res = scenes[payload["datasetName"]]
        return {
            "requestId": 1, "version": "SYNTHETIC-TEST",
            "data": {"results": res, "nextRecord": 0, "recordsReturned": len(res)},
            "errorCode": None, "errorMessage": None,
        }  # fmt: skip

    cat = s.data_raw / "catalog"
    if cat.exists():
        shutil.rmtree(cat)
    run_search(
        FixtureClient(responder, store=RawStore(cat)),
        load_aois(s),
        load_reference_features(s),
        ["REG"],
    )
    return World(x0, y1, ground, (float(dgx), float(dgy)))


__all__ = ["ENTITIES", "POST_ENTITY", "Affine", "World", "setup_project"]
