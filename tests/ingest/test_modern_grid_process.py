import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from sbf.core.provenance import read_prov
from sbf.ingest.modern import (
    LAYERS,
    FakeExporter,
    build_stack,
    grid_from_bounds,
    local_std,
    modern_grid,
    read_manifest,
    reproject_to_grid,
    slope_degrees,
    terrain_ruggedness_index,
)
from sbf.ingest.modern.chips import extract_chip, feature_chips
from sbf.ingest.modern.process import palsar_to_db, raw_dir, stack_dir

SMALL = (-0.0620, 26.3070, -0.0525, 26.3150)  # ~1 km box around GZ1


@pytest.fixture
def small_grid():
    return grid_from_bounds(SMALL, "EPSG:32631", 10.0)


def test_modern_grid_deterministic_and_snapped(tmp_project):
    g1 = modern_grid("REG", tmp_project)
    g2 = modern_grid("REG", tmp_project)
    assert g1 == g2
    t, w, h, crs = g1
    assert crs == "EPSG:32631"
    assert t.a == 10 and t.e == -10 and t.b == t.d == 0
    assert t.c % 10 == 0 and t.f % 10 == 0
    assert w > 4000 and h > 7000  # ~0.5 x 0.7 degrees


def test_modern_grid_covers_aoi_and_ine_in_zone31(tmp_project):
    from sbf.core.aoi import load_aois
    from sbf.core.crs import transform_geometry
    from sbf.ingest.modern.grid import grid_bounds

    for aoi in ("REG", "INE"):
        g = modern_grid(aoi, tmp_project)
        left, bottom, right, top = grid_bounds(g)
        geom = load_aois(tmp_project).set_index("aoi_id").loc[aoi, "geometry"]
        xmin, ymin, xmax, ymax = transform_geometry(geom, "EPSG:4326", g[3]).bounds
        assert left <= xmin and bottom <= ymin and right >= xmax and top >= ymax


def test_modern_grid_unknown_aoi(tmp_project):
    with pytest.raises(KeyError):
        modern_grid("XXX", tmp_project)


def test_slope_of_plane():
    _, xx = np.mgrid[0:20, 0:20]
    dem = (xx * 10.0 * np.tan(np.radians(15))).astype(np.float32)  # 15 deg east-rising, 10 m px
    s = slope_degrees(dem, 10.0)
    assert np.allclose(s, 15.0, atol=1e-3)
    assert np.allclose(slope_degrees(np.full((5, 5), 7.0), 10.0), 0)


def test_slope_nan_propagates():
    dem = np.zeros((6, 6), dtype=np.float32)
    dem[3, 3] = np.nan
    assert np.isnan(slope_degrees(dem, 10)[3, 3])


def test_tri_and_local_std():
    flat = np.full((7, 7), 5.0, dtype=np.float32)
    assert np.nanmax(terrain_ruggedness_index(flat)[1:-1, 1:-1]) == 0
    spike = flat.copy()
    spike[3, 3] = 8.0
    tri = terrain_ruggedness_index(spike)
    assert tri[3, 3] == pytest.approx(np.sqrt(8 * 9.0))
    assert np.isnan(tri[0, 0])
    st = local_std(spike, 3)
    assert st[3, 3] > 0 and st[0, 0] == 0
    # known value: 3x3 window with one outlier of +3 over 9 px
    assert st[3, 3] == pytest.approx(np.std([5] * 8 + [8]), rel=1e-4)


def test_local_std_nan_and_window_validation():
    a = np.arange(25, dtype=np.float32).reshape(5, 5)
    a[2, 2] = np.nan
    out = local_std(a, 3)
    assert np.isnan(out[2, 2]) and np.isfinite(out[0, 0])
    with pytest.raises(ValueError):
        local_std(a, 4)
    assert np.isnan(local_std(np.full((3, 3), np.nan), 3)).all()


def test_palsar_to_db():
    db, step = palsar_to_db(np.array([[1000.0, 0.0, np.nan]]))
    assert db[0, 0] == pytest.approx(10 * np.log10(1000.0**2) - 83)  # -23 dB
    assert np.isnan(db[0, 1]) and np.isnan(db[0, 2])
    assert "DN" in step
    already, step2 = palsar_to_db(np.array([[-20.0, -15.0]]))
    assert already.tolist() == [[-20.0, -15.0]] and "no conversion" in step2


def test_reproject_to_grid_aligns_offset_coarse_raster(tmp_path, small_grid):
    t, w, h, crs = small_grid
    # 30 m raster with a gentle ramp, misaligned origin, bigger than grid
    tr = Affine(30, 0, t.c - 45, 0, -30, t.f + 45)
    ww, hh = w // 3 + 4, h // 3 + 4
    ramp = np.tile(np.arange(ww, dtype=np.float32) * 30, (hh, 1))  # value == x offset in m
    p = tmp_path / "c.tif"
    with rasterio.open(
        p, "w", driver="GTiff", width=ww, height=hh, count=1, dtype="float32",
        crs=crs, transform=tr, nodata=-9999,
    ) as d:  # fmt: skip
        d.write(ramp, 1)
    out = reproject_to_grid(p, small_grid)
    assert out.shape == (1, h, w) and out.dtype == np.float32
    # centre columns: bilinear of a linear ramp reproduces x position
    x_center = t.c + (np.arange(w) + 0.5) * 10
    expect = x_center - (t.c - 45) - 15  # value at coarse pixel centres is index*30 at x=c0+15+30*i
    mid = out[0, h // 2, 5:-5]
    assert np.allclose(mid, expect[5:-5], atol=1e-3)


def test_reproject_nodata_becomes_nan(tmp_path, small_grid):
    t, w, h, crs = small_grid
    p = tmp_path / "n.tif"
    arr = np.ones((h, w), dtype=np.float32)
    arr[:10] = -9999
    with rasterio.open(
        p, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32",
        crs=crs, transform=t, nodata=-9999,
    ) as d:  # fmt: skip
        d.write(arr, 1)
    out = reproject_to_grid(p, small_grid, resampling="nearest")
    assert np.isnan(out[0, :10]).all() and (out[0, 12:] == 1).all()


@pytest.fixture
def built(tmp_project, small_grid):
    FakeExporter().export_aoi("REG", raw_dir("REG", tmp_project), grid=small_grid)
    return build_stack("REG", settings=tmp_project, grid=small_grid), tmp_project, small_grid


def test_build_stack_layers_manifest_and_cogs(built):
    m, s, grid = built
    assert set(m["layers"]) == set(LAYERS)
    assert m["missing"] == {}
    assert m["crs"] == "EPSG:32631" and m["resolution_m"] == 10
    assert (m["width"], m["height"]) == (grid[1], grid[2])
    assert Affine(*m["transform"]) == grid[0]
    on_disk = json.loads((stack_dir("REG", s) / "stack.json").read_text())
    assert on_disk["layers"].keys() == m["layers"].keys()
    for ent in m["layers"].values():
        path = stack_dir("REG", s) / ent["file"]
        with rasterio.open(path) as src:
            assert src.crs.to_string() == "EPSG:32631"
            assert src.transform == grid[0] and (src.width, src.height) == (grid[1], grid[2])
            assert src.count == len(ent["bands"])
            assert src.profile["tiled"]  # COG
        prov = read_prov(path)
        assert prov["processing_steps"] and prov["inputs"]
        assert prov["source"].startswith("FAKE")
        assert ent["units"] and all({"index", "name", "meaning"} <= b.keys() for b in ent["bands"])
    assert m["layers"]["s2_median"]["bands"][2]["name"] == "B4"
    assert m["layers"]["dem_slope"]["derived_from"] == "dem"


def test_build_stack_values_plausible(built):
    m, s, _ = built

    def arr(n):
        with rasterio.open(stack_dir("REG", s) / m["layers"][n]["file"]) as d:
            return d.read()

    assert -30 < np.nanmean(arr("palsar_hh")) < 0  # dB converted from DN
    assert np.nanmean(arr("s1_vv_std")) > 0
    slope = arr("dem_slope")[0]
    assert np.nanmax(slope) > 0 and np.nanmax(slope) < 45
    assert np.nanmax(arr("dem_roughness")) > 0
    assert 0 < np.nanmean(arr("s2_median")) < 1


def test_build_stack_records_missing_without_filling(tmp_project, small_grid):
    FakeExporter(fail={"dem", "palsar_multi"}).export_aoi(
        "REG", raw_dir("REG", tmp_project), grid=small_grid
    )
    m = build_stack("REG", settings=tmp_project, grid=small_grid)
    assert {"dem", "dem_slope", "dem_roughness", "palsar_hh_multi", "palsar_hv_multi"} == set(
        m["missing"]
    )
    assert "dataset not found" in m["missing"]["dem"]
    assert "dem" not in m["layers"] and not (stack_dir("REG", tmp_project) / "dem.tif").exists()
    assert "s2_median" in m["layers"]


def test_build_stack_unknown_layer(tmp_project, small_grid):
    with pytest.raises(KeyError):
        build_stack("REG", settings=tmp_project, grid=small_grid, layers=["nope"])


def test_read_manifest_absent(tmp_project):
    assert read_manifest("REG", tmp_project) is None


def test_layer_vocabulary_matches_architecture():
    want = {
        "s2_median", "s2_djf", "s2_mam", "s2_jja", "s2_son",
        "s1_vv_mean", "s1_vv_std", "s1_vh_mean", "s1_vh_std",
        "palsar_hh", "palsar_hv", "palsar_hh_multi", "palsar_hv_multi",
        "dem", "dem_slope", "dem_roughness",
    }  # fmt: skip
    assert set(LAYERS) == want


def test_chips(built):
    _, s, _ = built
    from sbf.core.crs import lonlat_to_xy

    chips = feature_chips("REG", -0.0572, 26.3117, 100, settings=s)
    assert set(chips) == set(LAYERS)
    assert chips["s2_median"].shape == (3, 21, 21) and chips["dem"].shape == (1, 21, 21)
    x, y = lonlat_to_xy(-0.0572, 26.3117, "EPSG:32631")
    assert extract_chip(stack_dir("REG", s) / "dem.tif", x + 1e6, y, 50) is None
    assert feature_chips("INE", 5.0, 24.0, settings=s) == {}
