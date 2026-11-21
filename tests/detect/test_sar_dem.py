"""Synthetic tests for the sar_z and dem_relief detectors."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import box
from typer.testing import CliRunner

from sbf.detect.harness import evaluate
from sbf.detect.sar_dem import (
    aggregator,
    blockwise,
    db_from_linear,
    local_z,
    relief_residual,
    run_dem_relief,
    run_sar_z,
)
from sbf.detect.sar_dem_cli import sar_dem_app
from sbf.detect.scores import random_scores, read_scores
from sbf.features.tiling import grid_origin

PX = 10.0
N = 120  # 1200 m
PATCH = (60, 40)  # row, col of a 3x3 px patch (pixel units)


def _write(path, arr, ox, oy):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32631",
        nodata=np.nan,
        transform=from_origin(ox, oy + arr.shape[0] * PX, PX, PX),
    ) as dst:
        dst.write(arr.astype("float32"), 1)


def _ref(ox, oy, r, c, half=2):
    top = oy + N * PX
    x0, y1 = ox + c * PX, top - r * PX
    g = box(x0 - half * PX, y1 - (half + 3) * PX, x0 + (half + 3) * PX, y1 + half * PX)
    return gpd.GeoDataFrame(
        {"feature_id": ["P"], "aoi_id": "REG", "precision": "iaea", "expected_area_km2": [None]},
        geometry=[g],
        crs="EPSG:32631",
    ).to_crs("EPSG:4326")


@pytest.fixture
def sar(tmp_project):
    ox, oy = grid_origin("REG")
    ox, oy = ox + 4096, oy + 4096
    rng = np.random.default_rng(1)
    base = -20 + 3 * ndimage.gaussian_filter(rng.normal(size=(N, N)), 15) * 5
    img = base + rng.normal(0, 1.0, (N, N))
    r, c = PATCH
    img[r : r + 3, c : c + 3] += 9
    path = tmp_project.root / "sar.tif"
    _write(path, img, ox, oy)
    return path, _ref(ox, oy, r, c), ox, oy


def test_sar_patch_ranks_top(tmp_project, sar):
    path, ref, *_ = sar
    for robust in (False, True):
        df = run_sar_z(
            "REG", "palsar_hv", path, 128, robust=robust, write=False, settings=tmp_project
        )
        assert (df["detector"] == "sar_z").all() and len(df) > 20
        m = evaluate(df, ref, gpd.GeoDataFrame(), tmp_project, ks=(5, 10)).iloc[0]
        assert m["recall_at_10"] == 1.0
        assert m["recall_at_5"] == 1.0
        assert m["ap"] > 3 * m["random_ap"]
        rnd = random_scores(df["tile_id"].tolist(), "palsar_hv", seed=2)
        r = evaluate(rnd, ref, gpd.GeoDataFrame(), tmp_project, ks=(5,)).iloc[0]
        assert m["ap"] > r["ap"]


def test_sar_linear_units_and_low_tail(tmp_project, sar):
    path, _, ox, oy = sar
    with rasterio.open(path) as s:
        lin = 10 ** (s.read(1) / 10)
    p2 = tmp_project.root / "lin.tif"
    _write(p2, lin, ox, oy)
    a = run_sar_z("REG", "palsar_hv", path, 128, write=False, settings=tmp_project)
    b = run_sar_z("REG", "palsar_hv", p2, 128, units="linear", write=False, settings=tmp_project)
    assert np.allclose(a["score"], b["score"], atol=1e-3)
    low = run_sar_z("REG", "palsar_hv", path, 128, tail="low", write=False, settings=tmp_project)
    hi = run_sar_z("REG", "palsar_hv", path, 128, tail="high", write=False, settings=tmp_project)
    assert hi["score"].max() > low["score"].max()


def test_block_equals_whole():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(70, 83)).astype("float32")
    a[10:20, 5:9] = np.nan
    a[40, 40] = np.nan
    for robust in (False, True):
        whole = local_z(a, 2, 6, robust)
        blk = blockwise(a, lambda x, rb=robust: local_z(x, 2, 6, rb), halo=6, block=25)
        assert np.allclose(whole, blk, equal_nan=True, atol=1e-5)
    whole = relief_residual(a, 3.0)
    blk = blockwise(a, lambda x: relief_residual(x, 3.0), halo=13, block=25)
    assert np.allclose(whole, blk, equal_nan=True, atol=1e-5)
    whole = relief_residual(a, 3.0, "median")
    blk = blockwise(a, lambda x: relief_residual(x, 3.0, "median"), halo=4, block=25)
    assert np.allclose(whole, blk, equal_nan=True, atol=1e-5)


def test_nodata_handling():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(40, 40)).astype("float32")
    a[5:15, 5:15] = np.nan
    for robust in (False, True):
        z = local_z(a, 2, 6, robust)
        assert np.isnan(z[5:15, 5:15]).all()  # centre nodata -> NaN
        assert np.isfinite(z[25:35, 25:35]).all()
        assert not np.isinf(z).any()
    z = local_z(np.full((20, 20), np.nan, dtype="float32"), 2, 6)
    assert np.isnan(z).all()
    d = relief_residual(a, 2.0)
    assert np.isnan(d[8:12, 8:12]).all() and np.isfinite(d[25:35, 25:35]).all()
    # flat background: sd floor keeps z finite
    assert np.isfinite(local_z(np.zeros((30, 30), "float32"), 2, 6, min_sd=0.1)[3:-3, 3:-3]).all()
    out = db_from_linear(np.array([1.0, 0.1, 0.0, -1.0, np.nan]))
    assert np.allclose(out[:2], [0, -10]) and np.isnan(out[2:]).all()


def test_aggregator():
    v = np.arange(101.0)
    assert aggregator("p95")(v) == pytest.approx(95)
    assert aggregator("max")(v) == 100 and aggregator("median")(v) == 50
    with pytest.raises(ValueError):
        aggregator("foo")


@pytest.fixture
def dem(tmp_project):
    ox, oy = grid_origin("REG")
    ox, oy = ox + 4096, oy + 4096
    rng = np.random.default_rng(5)
    z = 100 + 40 * ndimage.gaussian_filter(rng.normal(size=(N, N)), 25) / 0.03
    z = 100 + 5 * (z - 100) / (z - 100).std()
    z += rng.normal(0, 0.02, (N, N))
    r, c = 30, 80
    z[r : r + 4, c : c + 4] += 3.0  # mound 40 m wide
    z[90:92, 20:32] -= 2.5  # trench
    path = tmp_project.root / "dem.tif"
    _write(path, z, ox, oy)
    return path, _ref(ox, oy, r, c, 2), _ref(ox, oy, 90, 24, 3), ox, oy


def test_dem_mound_and_trench_rank_top(tmp_project, dem):
    path, ref_mound, ref_trench, *_ = dem
    for method in ("gaussian", "median"):
        df = run_dem_relief(
            "REG", path, "dem", 128, 50.0, method, write=False, settings=tmp_project
        )
        assert (df["detector"] == "dem_relief").all()
        for ref in (ref_mound, ref_trench):
            m = evaluate(df, ref, gpd.GeoDataFrame(), tmp_project, ks=(5, 10)).iloc[0]
            assert m["recall_at_10"] == 1.0
            assert m["ap"] > 3 * m["random_ap"]


def test_dem_roughness_variant(tmp_project, dem):
    path, ref_mound, *_ = dem
    with rasterio.open(path) as s:
        z = s.read(1)
        t = s.transform
    rough = ndimage.generic_filter(z, np.std, size=5, mode="nearest")
    p = tmp_project.root / "rough.tif"
    _write(p, rough, t.c, t.f - N * PX)
    df = run_dem_relief("REG", p, "dem_roughness", 128, write=False, settings=tmp_project)
    assert df["layer"].iloc[0] == "dem_roughness"
    m = evaluate(df, ref_mound, gpd.GeoDataFrame(), tmp_project, ks=(10,)).iloc[0]
    assert m["recall_at_10"] == 1.0


def test_write_and_cli(tmp_project, sar, dem):
    runner = CliRunner()
    r = runner.invoke(
        sar_dem_app, ["sar", "--aoi", "REG", "--layer", "s1_vv_mean", "--raster", str(sar[0])]
    )
    assert r.exit_code == 0, r.output
    r = runner.invoke(sar_dem_app, ["dem", "--aoi", "REG", "--raster", str(dem[0])])
    assert r.exit_code == 0, r.output
    assert "POLYGON" not in r.output
    assert len(read_scores("REG", "sar_z", "s1_vv_mean", tmp_project)) > 0
    assert len(read_scores("REG", "dem_relief", "dem", tmp_project)) > 0


def test_manifest_resolution(tmp_project, sar):
    import json

    d = tmp_project.data_processed / "modern" / "REG"
    d.mkdir(parents=True)
    (d / "palsar_hv.tif").write_bytes(sar[0].read_bytes())
    (d / "stack.json").write_text(
        json.dumps({"layers": {"palsar_hv": {"file": "palsar_hv.tif", "units": "dB"}}})
    )
    df = run_sar_z("REG", "palsar_hv", None, 128, write=False, settings=tmp_project)
    assert len(df) > 20
    with pytest.raises(FileNotFoundError):
        run_sar_z("REG", "s1_vv_mean", None, 128, write=False, settings=tmp_project)
