import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from sbf.features.preprocess import clahe, local_contrast_normalise, preprocess_array
from sbf.features.texture import FEATURE_NAMES, chip_texture
from sbf.features.tiling import (
    apply_exclusions,
    grid_origin,
    make_grid,
    make_tile_id,
    parse_tile_id,
    tile_geometries,
    tile_windows,
)

BOUNDS = (500_000.0, 2_900_000.0, 500_000.0 + 512, 2_900_000.0 + 384)


def test_tile_id_format_roundtrip():
    assert make_tile_id("REG", 64, 3, 12) == "REG-64-00003-00012"
    assert parse_tile_id("REG-64-00003-00012") == ("REG", 64, 3, 12)
    with pytest.raises(ValueError):
        parse_tile_id("REG-64-3-12")


def test_origin_snapped_to_128():
    for aoi in ("REG", "INE"):
        ox, oy = grid_origin(aoi)
        assert ox % 128 == 0 and oy % 128 == 0


def test_grid_deterministic_and_ids():
    ox, oy = grid_origin("REG")
    b = (ox + 1024, oy + 1024, ox + 1024 + 256, oy + 1024 + 192)
    g1 = make_grid("REG", 64, 0.5, bounds=b)
    g2 = make_grid("REG", 64, 0.5, bounds=b)
    assert list(g1.tile_id) == list(g2.tile_id)
    assert list(g1.columns) == ["tile_id", "aoi_id", "size_m", "geometry"]
    assert str(g1.crs).endswith("32631")
    # step 32: (256-64)/32+1 = 7 cols, (192-64)/32+1 = 5 rows
    assert len(g1) == 35
    assert g1.tile_id.iloc[0] == make_tile_id("REG", 64, 1024 // 32, 1024 // 32)
    assert g1.geometry.iloc[0].bounds == (ox + 1024, oy + 1024, ox + 1088, oy + 1088)
    assert g1.tile_id.is_unique


def test_sizes_share_lattice_and_ids_match_full_grid():
    ox, oy = grid_origin("INE")
    b = (ox + 256, oy + 256, ox + 640, oy + 640)
    g128 = make_grid("INE", 128, 0.5, bounds=b)
    assert g128.tile_id.iloc[0] == "INE-128-00004-00004"
    geo = tile_geometries(list(g128.tile_id))
    assert all(a.equals(b_) for a, b_ in zip(geo.geometry, g128.geometry, strict=True))


def test_grid_counts_full_aoi():
    g = make_grid("INE", 128, 0.0)
    assert len(g) > 100_000
    assert g.tile_id.is_unique


def test_apply_exclusions():
    ox, oy = grid_origin("REG")
    g = make_grid("REG", 64, 0.0, bounds=(ox, oy, ox + 256, oy + 128))  # 4x2 tiles
    ex = gpd.GeoDataFrame(geometry=[box(ox + 10, oy + 10, ox + 20, oy + 20)], crs=g.crs)
    out = apply_exclusions(g, ex)
    assert out["excluded"].sum() == 1 and out["excluded"].iloc[0]
    assert apply_exclusions(g, None)["excluded"].sum() == 0
    # other CRS accepted
    assert apply_exclusions(g, ex.to_crs("EPSG:4326"))["excluded"].sum() == 1


def _raster(tmp_path, nodata=None):
    ox, oy = grid_origin("REG")
    path = tmp_path / "r.tif"
    arr = np.arange(100 * 100, dtype="float32").reshape(100, 100)
    t = from_origin(ox + 1024, oy + 1024 + 200, 2.0, 2.0)  # 200 m x 200 m
    if nodata is not None:
        arr[:, :20] = nodata
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="float32",
        crs="EPSG:32631",
        transform=t,
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)
    return path, (ox + 1024, oy + 1024), arr


def test_tile_windows_shapes_and_values(tmp_path):
    path, (x0, y0), arr = _raster(tmp_path)
    g = make_grid("REG", 64, 0.5, bounds=(x0 - 64, y0 - 64, x0 + 264, y0 + 264))
    chips = dict(tile_windows(g, path))
    assert chips and all(c.shape == (32, 32) for c in chips.values())
    # tile fully inside: first tile at (x0, y0) is the bottom-left 32x32 px block
    inside = chips[make_tile_id("REG", 64, 1024 // 32, 1024 // 32)]
    assert np.array_equal(inside, arr[-32:, :32])
    partial = [c for c in chips.values() if np.isnan(c).any()]
    assert partial  # tiles hanging over the edge are NaN-padded


def test_tile_windows_nodata_and_min_valid(tmp_path):
    path, (x0, y0), _ = _raster(tmp_path, nodata=-9999.0)
    g = make_grid("REG", 64, 0.5, bounds=(x0, y0, x0 + 200, y0 + 200))
    chips = dict(tile_windows(g, path))
    left = chips[make_tile_id("REG", 64, 1024 // 32, 1024 // 32)]
    assert np.isnan(left[:, :20]).all() and np.isfinite(left[:, 20:]).all()
    strict = dict(tile_windows(g, path, min_valid_frac=1.0))
    assert len(strict) < len(chips)


def test_preprocess_and_texture():
    rng = np.random.default_rng(0)
    img = rng.normal(0.9, 0.01, (64, 64)).astype("float32")
    img[10:20, 10:20] = np.nan
    for out in (clahe(img), local_contrast_normalise(img), preprocess_array(img, "none")):
        assert out.shape == img.shape and np.isnan(out[12, 12]) and np.isfinite(out[40, 40])
    assert 0 <= np.nanmin(clahe(img)) and np.nanmax(clahe(img)) <= 1
    with pytest.raises(ValueError):
        preprocess_array(img, "bogus")
    flat = chip_texture(np.full((32, 32), 0.5))
    stripes = np.zeros((32, 32))
    stripes[::2] = 1.0
    busy = chip_texture(stripes)
    assert flat.shape == (len(FEATURE_NAMES),)
    assert busy[0] > flat[0] and busy[1] >= flat[1] and busy[2] > flat[2]
    assert np.isnan(chip_texture(np.full((8, 8), np.nan))).all()
