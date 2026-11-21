"""SPEC section 8 integration: fake trenches on a synthetic desert; texture detector + harness."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import box
from typer.testing import CliRunner

from sbf.cli import app
from sbf.detect.harness import evaluate
from sbf.detect.scores import random_scores, read_scores, write_scores
from sbf.detect.texture_detector import run_texture_detector
from sbf.features.tiling import grid_origin

RES = 2.0
SIZE_PX = 512  # 1024 m
TRENCH_BLOCKS = [(256, 256), (640, 384), (384, 704)]  # lower-left (m) of 64 m blocks


@pytest.fixture
def desert(tmp_project):
    ox, oy = grid_origin("REG")
    x0, y0 = ox + 4096, oy + 4096
    rng = np.random.default_rng(42)
    dunes = ndimage.gaussian_filter(rng.normal(size=(SIZE_PX, SIZE_PX)), 40)
    dunes = 0.25 * dunes / dunes.std()
    img = 0.5 + dunes + rng.normal(0, 0.02, (SIZE_PX, SIZE_PX))
    for bx, by in TRENCH_BLOCKS:  # parallel trenches: 4 px wide dark bands every 8 px
        c0, r0 = int(bx / RES), SIZE_PX - int((by + 64) / RES)
        block = img[r0 : r0 + 32, c0 : c0 + 32]
        for k in range(0, 32, 8):
            block[:, k : k + 4] -= 0.35
    path = tmp_project.root / "synthetic.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SIZE_PX,
        width=SIZE_PX,
        count=1,
        dtype="float32",
        crs="EPSG:32631",
        transform=from_origin(x0, y0 + 1024, RES, RES),
    ) as dst:
        dst.write(img.astype("float32"), 1)
    ref_utm = gpd.GeoDataFrame(
        {
            "feature_id": [f"T{i}" for i in range(3)],
            "aoi_id": "REG",
            "precision": "iaea",
            "expected_area_km2": [None] * 3,
        },
        geometry=[
            box(x0 + bx + 8, y0 + by + 8, x0 + bx + 56, y0 + by + 56) for bx, by in TRENCH_BLOCKS
        ],
        crs="EPSG:32631",
    )
    return path, ref_utm.to_crs("EPSG:4326")


@pytest.mark.parametrize("method", ["iforest", "knn", "maha"])
def test_texture_finds_fake_trenches(tmp_project, desert, method):
    path, ref = desert
    df = run_texture_detector(
        "REG", "synthetic", path, 64, method, write=False, settings=tmp_project
    )
    assert len(df) > 500 and (df["detector"] == "texture").all()
    m = evaluate(df, ref, gpd.GeoDataFrame(), tmp_project, ks=(10, 50)).iloc[0]
    assert m["n_features"] == 3
    assert m["recall_at_50"] == 1.0
    assert m["recall_at_50"] > m["random_recall_at_50"] * 2
    assert m["ap"] > 5 * m["random_ap"]

    rnd = random_scores(df["tile_id"].tolist(), "synthetic", seed=3)
    r = evaluate(rnd, ref, gpd.GeoDataFrame(), tmp_project, ks=(10, 50)).iloc[0]
    assert m["ap"] > 3 * r["ap"]


def test_clahe_preprocess_and_roundtrip(tmp_project, desert):
    path, _ = desert
    df = run_texture_detector(
        "REG", "synthetic", path, 128, "iforest", preprocess="clahe", settings=tmp_project
    )
    back = read_scores("REG", "texture", "synthetic", tmp_project)
    assert len(back) == len(df) and (back["size_m"] == 128).all()
    write_scores(random_scores(back["tile_id"].tolist(), "synthetic"), tmp_project)


def test_cli_end_to_end(tmp_project, desert):
    path, _ = desert
    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "detect",
            "texture",
            "--aoi",
            "REG",
            "--layer",
            "synthetic",
            "--raster",
            str(path),
            "--size",
            "64",
        ],
    )
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["detect", "random", "--aoi", "REG", "--layer", "synthetic"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["detect", "score", "--aoi", "REG"])
    assert r.exit_code == 0, r.output
    assert "texture" in r.output and "random" in r.output and "POLYGON" not in r.output
    r = runner.invoke(app, ["features", "grid", "--aoi", "INE", "--size", "128"])
    assert r.exit_code == 0 and "tiles" in r.output
