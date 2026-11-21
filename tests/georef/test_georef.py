import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from georef_synth import X0, Y1, make_frame, make_reference, make_scene
from shapely.geometry import Point

from sbf.georef import fit as fitmod
from sbf.georef.coarse import (
    apply_h,
    coarse_candidates,
    coarse_transform,
    footprint_corners_xy,
)
from sbf.georef.mask import MaskParams, stable_terrain_mask
from sbf.georef.models import GeorefReport
from sbf.georef.overlay import gz_overlay
from sbf.georef.pipeline import is_panoramic, solve_frame
from sbf.georef.warp import warp_frame


@pytest.fixture(scope="module")
def world():
    scene = make_scene()
    return scene, make_reference(scene), make_frame(scene)


# ---------------------------------------------------------------- coarse


def test_coarse_resolves_corner_order(world):
    _, _ref, fr = world
    h, w = fr.image.shape
    corners = footprint_corners_xy(fr.corners_lonlat)
    cands = coarse_candidates(fr.image.shape, corners)
    assert len(cands) == 8
    px_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    truth = fr.truth(px_corners)
    # exactly one variant (up to footprint noise) maps frame corners onto the true corners
    errs = [np.abs(apply_h(c.H, px_corners) - truth).max() for c in cands]
    assert min(errs) < 250 and sorted(errs)[1] > 1000
    # H reproduces the ordered corners exactly
    c = cands[3]
    assert np.allclose(apply_h(c.H, px_corners), c.corners_xy, atol=1e-3)
    assert coarse_transform(fr.image.shape, fr.corners_lonlat, 2).shape == (3, 3)


# ---------------------------------------------------------------- fit


def test_poly_recovers_known_polynomial():
    rng = np.random.default_rng(0)
    px = rng.uniform(0, 1000, (80, 2))
    xy = np.column_stack(
        [5e5 + 2 * px[:, 0] + 1e-3 * px[:, 0] * px[:, 1], 3e6 - 2 * px[:, 1] + 2e-4 * px[:, 0] ** 2]
    )
    m = fitmod.fit_transform(px, xy, "poly2")
    assert fitmod.rmse(fitmod.residuals_m(m, px, xy)) < 1e-6
    assert fitmod.rmse(fitmod.residuals_m(fitmod.fit_transform(px, xy, "poly1"), px, xy)) > 1


def test_overfit_guard_and_selection():
    rng = np.random.default_rng(1)
    px = rng.uniform(0, 500, (14, 2))
    xy = px * 3 + rng.normal(0, 0.5, px.shape)
    with pytest.raises(ValueError):
        fitmod.fit_transform(px[:11], xy[:11], "poly2")  # needs >= 12
    sel = fitmod.select_model(px, xy, "auto", seed=3)
    # 14 pts -> 11 train: poly1 only (poly2 needs 12, poly3 20), TPS needs 10 -> eligible
    assert sel.candidate_rmse_m["poly2"] is None and sel.candidate_rmse_m["poly3"] is None
    assert sel.method in ("poly1", "tps")
    with pytest.raises(ValueError):
        fitmod.select_model(px[:5], xy[:5], "auto")


def test_split_is_seeded_and_20_percent():
    a = fitmod.split_checkpoints(50, seed=7)
    b = fitmod.split_checkpoints(50, seed=7)
    assert all((x == y).all() for x, y in zip(a, b))
    assert len(a[1]) == 10 and set(a[0]).isdisjoint(a[1])


def test_tps_fits_nonlinear():
    rng = np.random.default_rng(2)
    px = rng.uniform(0, 500, (60, 2))
    xy = px * 2 + 5 * np.sin(px / 80)
    sel = fitmod.select_model(px, xy, "tps", smoothing=1e-3, strategy="random")
    assert sel.check_rmse_m < 2
    blocked = fitmod.select_model(px, xy, "tps", smoothing=1e-3)
    assert blocked.cv_rmse_m < 4 and blocked.random_check_rmse_m < 2


# ---------------------------------------------------------------- mask


def test_stable_terrain_mask_rule():
    n = 100
    s2 = np.full((n, n), 0.4, "float32")
    _yy, xx = np.mgrid[:n, :n]
    s2[:, :50] += 0.04 * np.sin(xx[:, :50] / 1.2)  # rippled dunes (std ~0.03) in left half
    s2[:, 50:] += np.random.default_rng(0).normal(0, 0.1, (n, 50))  # rough bedrock: std too high
    slope = np.zeros((n, n), "float32")
    slope[:, :50] = 10.0
    s2[0, 0] = np.nan
    p = MaskParams(dune_buffer_px=0)
    m = stable_terrain_mask({"s2": s2, "slope": slope}, p)
    assert not m[50, 20] and m[50, 80] and not m[0, 0]
    # textured but flat -> kept; no DEM at all -> kept
    assert stable_terrain_mask({"s2": s2, "slope": np.zeros_like(slope)}, p)[50, 20]
    assert stable_terrain_mask({"s2": s2}, p)[50, 20]
    # roughness fallback
    rough = np.where(xx < 50, 5.0, 0.0)
    assert not stable_terrain_mask({"s2": s2, "roughness": rough}, p)[50, 20]


# ---------------------------------------------------------------- end to end


def test_pipeline_recovers_distortion(world):
    _scene, ref, fr = world
    sol = solve_frame(
        fr.image, fr.corners_lonlat, ref, entity_id="SYN1", camera="", aoi_id="REG", seed=0
    )
    r = sol.report
    assert r.method != "none" and r.n_gcps >= 40, r
    assert r.method in ("poly2", "poly3", "tps")
    assert r.checkpoint_rmse_m < 10.0  # < 1 reference pixel
    assert r.included_for_change_detection
    # mirrored variants (0-3) cannot match; rotation-invariant matchers inlier-match 4-7, and
    # the near-identity residual picks the right rotation
    assert max(r.corner_variant_inliers[:4]) < 20 and r.coarse_variant in (4, 5, 6, 7)
    # accuracy against ground truth (not just against the matches)
    h, w = fr.image.shape
    g = np.array([(x, y) for x in range(50, w, 80) for y in range(50, h, 80)], float)
    err = np.hypot(*(sol.final_model(g) - fr.truth(g)).T)
    assert np.sqrt(np.mean(err**2)) < 3.0
    assert set(sol.gcps["role"]) == {"train", "check"}
    assert list(sol.gcps.columns) == ["id", "px", "py", "x", "y", "role", "residual_m"]
    # report schema roundtrip
    rep = GeorefReport.model_validate_json(r.model_dump_json())
    assert rep.entity_id == "SYN1" and rep.n_checkpoints == (sol.gcps["role"] == "check").sum()


def test_pipeline_panoramic_flag_and_low_overlap(world):
    _, ref, fr = world
    sol = solve_frame(fr.image, fr.corners_lonlat + [3.0, 0.0], ref, entity_id="X", camera="KH-4B")
    assert "panoramic_needs_strips" in sol.report.warnings
    assert "low_overlap" in sol.report.warnings
    assert sol.report.method == "none" and not sol.report.included_for_change_detection


def test_is_panoramic():
    assert is_panoramic("KH-4B Forward")
    assert is_panoramic("KH-9 Panoramic Camera")
    assert not is_panoramic("KH-9 Mapping Camera")
    assert not is_panoramic(None, "")


# ---------------------------------------------------------------- warp


def test_warp_geotransform_and_content(world):
    scene, _ref, fr = world
    h, w = fr.image.shape
    g = np.array([(x, y) for x in range(0, w + 1, 50) for y in range(0, h + 1, 50)], float)
    fit = fitmod.fit_transform(g, fr.truth(g), "poly2")
    res = warp_frame(fr.image, fit, resolution_m=5.0)
    assert res.crs == "EPSG:32631" and res.transform.a == 5.0 and res.transform.e == -5.0
    assert res.transform.c % 5.0 == 0 and res.transform.f % 5.0 == 0
    assert res.array.dtype == np.uint8 and res.nodata == 0
    # warped frame should correlate with the scene resampled onto the same grid
    H, W = res.array.shape
    cs = ((res.transform.c + 5 * (np.arange(W) + 0.5)) - X0) / 2.5 - 0.5
    rs = (Y1 - (res.transform.f - 5 * (np.arange(H) + 0.5))) / 2.5 - 0.5
    from scipy import ndimage

    ref_img = ndimage.map_coordinates(scene, np.meshgrid(rs, cs, indexing="ij"), order=1)
    valid = res.array > 0
    valid[:3] = valid[-3:] = False
    valid[:, :3] = valid[:, -3:] = False
    assert valid.mean() > 0.3
    cc = np.corrcoef(res.array[valid].astype(float), ref_img[valid])[0, 1]
    assert cc > 0.9


# ---------------------------------------------------------------- overlay


def test_gz_overlay(world, tmp_path):
    _scene, _ref, fr = world
    h, w = fr.image.shape
    g = np.array([(x, y) for x in range(0, w + 1, 50) for y in range(0, h + 1, 50)], float)
    res = warp_frame(fr.image, fitmod.fit_transform(g, fr.truth(g), "poly2"), resolution_m=5.0)
    from pyproj import Transformer

    lon, lat = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True).transform(
        X0 + 2000, Y1 - 2000
    )
    gz = gpd.GeoDataFrame(
        {"feature_id": ["GZ1", "GZ2"]},
        geometry=[Point(lon, lat), Point(lon + 1, lat)],
        crs="EPSG:4326",
    )
    png, inside = gz_overlay(res, gz, "SYN1", tmp_path)
    assert inside == ["GZ1"] and png.exists()
    assert gz_overlay(res, gz.iloc[1:], "SYN1", tmp_path) == (None, [])


# ---------------------------------------------------------------- IO / CLI


def _build_project(tmp_project, world):
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import Polygon

    _scene, ref, fr = world
    s = tmp_project
    sd = s.data_processed / "modern" / "REG"
    sd.mkdir(parents=True)
    arr = np.stack([ref.array] * 4)
    prof = {
        "driver": "GTiff",
        "count": 4,
        "dtype": "float32",
        "crs": "EPSG:32631",
        "nodata": np.nan,
        "transform": ref.transform,
        "height": arr.shape[1],
        "width": arr.shape[2],
    }
    with rasterio.open(sd / "s2_median.tif", "w", **prof) as dst:
        dst.write(arr)
    layers = {
        "s2_median": {
            "file": "s2_median.tif",
            "bands": [{"index": i + 1, "name": n} for i, n in enumerate(["B2", "B3", "B4", "B8"])],
        }
    }
    (sd / "stack.json").write_text(json.dumps({"layers": layers}))
    ed = s.data_raw / "declass" / "SYN1"
    ed.mkdir(parents=True)
    fprof = {"driver": "GTiff", "count": 1, "dtype": "uint8", "height": 500, "width": 500}
    with rasterio.open(ed / "SYN1.tif", "w", **fprof) as dst:
        dst.write(fr.image, 1)
    poly = Polygon(fr.corners_lonlat)
    pd.DataFrame(
        [
            {
                "entity_id": "SYN1",
                "dataset": "CORONA2",
                "mission": "KH-4B",
                "camera": "",
                "acquisition_date": "1965-01-01",
                "cloud_cover": 0,
                "availability": "scanned",
                "browse_url": "",
                "footprint_wkt": poly.wkt,
                "intersecting_features": "GZ1",
                "resolution_class": "<=3m",
                "aoi_id": "REG",
                "raw_file": "",
            }
        ]
    ).to_csv(s.outputs_public / "coverage.csv", index=False)
    lon, lat = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True).transform(
        X0 + 2000, Y1 - 2000
    )
    return gpd.GeoDataFrame({"feature_id": ["GZ1"]}, geometry=[Point(lon, lat)], crs="EPSG:4326")


def test_run_frame_outputs(tmp_project, world):
    import rasterio

    from sbf.core.provenance import read_prov
    from sbf.georef.pipeline import run_frame

    gz = _build_project(tmp_project, world)
    rep = run_frame("SYN1", settings=tmp_project, gz=gz, strips=False)
    d = tmp_project.data_processed / "historical" / "SYN1"
    on_disk = json.loads((d / "georef_report.json").read_text())
    assert on_disk["entity_id"] == "SYN1" and on_disk["aoi_id"] == "REG"
    assert (
        on_disk["included_for_change_detection"] and "panoramic_needs_strips" in on_disk["warnings"]
    )
    gcps = pd.read_csv(d / "gcps.csv")
    assert len(gcps) == rep.n_gcps and {"train", "check"} == set(gcps.role)
    with rasterio.open(d / "SYN1_georef.tif") as src:
        assert src.crs.to_string() == "EPSG:32631" and src.res == (rep.resolution_m,) * 2
    assert "download_date" in read_prov(d / "SYN1_georef.tif")
    assert on_disk["cv_rmse_m"] is not None and on_disk["gcp_block_occupancy"] is not None
    with rasterio.open(d / "SYN1_georef.tif") as g, rasterio.open(d / "SYN1_support.tif") as m:
        assert (g.transform, g.shape) == (m.transform, m.shape)
        sup = m.read(1)
    assert on_disk["support_file"] == str(d / "SYN1_support.tif") and 0 < sup.mean() <= 1
    assert (tmp_project.outputs_public / "georef" / "SYN1_gz.png").exists()
    assert rep.gz_offsets["GZ1"].dx_m is None  # offsets are human-measured placeholders


def test_cli_frame(tmp_project, world):
    from typer.testing import CliRunner

    from sbf.cli import app

    _build_project(tmp_project, world)
    res = CliRunner().invoke(
        app, ["georef", "frame", "--entity-id", "SYN1", "--method", "poly2", "--no-strips"]
    )
    assert res.exit_code == 0, res.output
    assert "method=poly2" in res.output
    bad = CliRunner().invoke(app, ["georef", "frame", "--entity-id", "NOPE"])
    assert bad.exit_code == 1
