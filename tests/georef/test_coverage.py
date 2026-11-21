"""Honest accuracy: spatially blocked hold-out, spatial CV, coverage metrics, support mask, gate."""

import numpy as np
import pytest
import rasterio
from georef_synth import make_panoramic_frame, make_reference, make_scene

from sbf.georef import fit as fitmod
from sbf.georef import strips as st
from sbf.georef import support as sup
from sbf.georef.models import MIN_BLOCK_OCCUPANCY, GeorefReport
from sbf.georef.pipeline import fit_gcps

SHAPE = (1000, 1000)
EXTENT = (0.0, 0.0, 1000.0, 1000.0)


def _truth(px):
    """Smooth non-polynomial distortion: a TPS through sparse clumps cannot know it in between."""
    px = np.asarray(px, float)
    return np.column_stack(
        [
            5e5 + 5 * px[:, 0] + 60 * np.sin(px[:, 0] / 150),
            3e6 - 5 * px[:, 1] + 40 * np.cos(px[:, 1] / 200),
        ]
    )


def _clumps(seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[150, 150], [800, 200], [250, 800], [850, 850]])
    return np.vstack([c + rng.normal(0, 30, (12, 2)) for c in centres]).clip(0, 999)


def _fit(px, method):
    rep = GeorefReport(entity_id="T", method="none", georef_file="x.tif")
    sol = fit_gcps(rep, px, _truth(px), SHAPE, method=method, trim=False)
    return rep, sol


# ---------------------------------------------------------------- ticket-08 situation


def test_clustered_gcps_random_checkpoints_are_optimistic():
    px = _clumps()
    rep, sol = _fit(px, "tps")
    assert rep.random_checkpoint_rmse_m < 3.0  # what the old gate saw: "fine"
    assert rep.cv_rmse_m > 20.0 and rep.cv_rmse_m > 10 * rep.random_checkpoint_rmse_m
    assert rep.gcp_block_occupancy < MIN_BLOCK_OCCUPANCY
    assert not rep.included_for_change_detection
    for w in ("rmse_gt_20m_excluded", "low_gcp_coverage", "cv_vs_checkpoint_gap"):
        assert w in rep.warnings, rep.warnings
    # the truth over the whole frame is indeed far worse than the random checkpoints claimed
    g = np.random.default_rng(1).uniform(0, 1000, (2000, 2))
    true_err = fitmod.rmse(fitmod.residuals_m(sol.final_model, g, _truth(g)))
    assert true_err > 5 * rep.random_checkpoint_rmse_m


def test_well_covered_frame_stays_included():
    px = np.random.default_rng(0).uniform(0, 1000, (100, 2))
    rep, _ = _fit(px, "auto")
    assert rep.gcp_block_occupancy >= 0.6 and rep.gcp_hull_fraction > 0.8
    assert rep.cv_rmse_m <= 20.0 and rep.included_for_change_detection
    assert "low_gcp_coverage" not in rep.warnings and "cv_vs_checkpoint_gap" not in rep.warnings


def test_gate_requires_both_conditions():
    rep = GeorefReport(
        entity_id="G", method="poly2", cv_rmse_m=8.0, checkpoint_rmse_m=7.0, gcp_block_occupancy=0.3
    )
    from sbf.georef.models import apply_quality_gate

    apply_quality_gate(rep)
    assert not rep.included_for_change_detection and "low_gcp_coverage" in rep.warnings
    rep = GeorefReport(
        entity_id="G",
        method="poly2",
        cv_rmse_m=15.0,
        checkpoint_rmse_m=7.0,
        gcp_block_occupancy=0.5,
    )
    apply_quality_gate(rep)
    assert rep.included_for_change_detection and "rmse_gt_10m" in rep.warnings
    assert "cv_vs_checkpoint_gap" in rep.warnings  # 15 > 2 x 7


# ---------------------------------------------------------------- split / coverage


def test_blocked_split_is_spatially_separated_and_seeded():
    px = np.random.default_rng(3).uniform(0, 1000, (150, 2))
    tr, ck = fitmod.split_checkpoints(150, seed=4, px=px, extent=EXTENT)
    _tr2, ck2 = fitmod.split_checkpoints(150, seed=4, px=px, extent=EXTENT)
    assert (ck == ck2).all() and set(tr).isdisjoint(ck) and len(tr) + len(ck) == 150
    assert 0.1 * 150 <= len(ck) <= 0.4 * 150
    rtr, rck = fitmod.split_checkpoints(150, seed=4, strategy="random")
    assert len(rck) == 30

    def nn(a, b):
        d = np.hypot(*(px[a][:, None, :] - px[b][None, :, :]).transpose(2, 0, 1))
        return d.min(axis=1).mean()

    assert nn(ck, tr) > 1.5 * nn(rck, rtr)  # blocked checkpoints are far from training points
    with pytest.raises(ValueError):
        fitmod.split_checkpoints(10, strategy="bogus")


def test_spatial_folds_keep_blocks_whole():
    px = np.random.default_rng(5).uniform(0, 1000, (120, 2))
    folds = fitmod.spatial_folds(px, EXTENT)
    assert len(folds) == fitmod.CV_FOLDS
    assert sorted(np.concatenate(folds)) == list(range(120))
    ids = fitmod.block_ids(px, EXTENT, 5)
    owner = {b: i for i, f in enumerate(folds) for b in np.unique(ids[f])}
    assert sum(len(np.unique(ids[f])) for f in folds) == len(owner)  # no block in two folds


def test_coverage_metrics():
    rng = np.random.default_rng(0)
    corner = rng.uniform(0, 150, (40, 2))  # one corner of the frame
    hull, occ = fitmod.coverage_metrics(corner, EXTENT)
    assert hull < 0.05 and occ == pytest.approx(1 / 25)
    full = rng.uniform(0, 1000, (400, 2))
    hull, occ = fitmod.coverage_metrics(full, EXTENT)
    assert hull > 0.9 and occ == 1.0
    line = np.column_stack([np.linspace(0, 999, 30), np.linspace(0, 999, 30)])
    assert fitmod.coverage_metrics(line, EXTENT)[0] == 0.0  # degenerate hull


# ---------------------------------------------------------------- support mask


def _frame_model_and_grid():
    g = np.random.default_rng(2).uniform(0, 300, (40, 2))  # GCPs in the lower-left 300x300 px
    xy = np.column_stack([5e5 + 5 * g[:, 0], 3e6 - 5 * g[:, 1]])
    model = fitmod.fit_transform(g, xy, "poly1")
    from sbf.georef.warp import Grid, frame_bounds_xy

    grid = Grid.from_bounds(frame_bounds_xy(model, SHAPE), 10.0)
    return g, model, grid


def _cell(grid, xy):
    return int((grid.y1 - xy[1]) // grid.res), int((xy[0] - grid.x0) // grid.res)


def test_support_mask_geometry(tmp_path):
    g, model, grid = _frame_model_and_grid()
    poly = sup.support_polygon(g, EXTENT)
    d = 0.1 * np.hypot(1000, 1000)
    fn = sup.frame_support_fn(model, SHAPE, poly)
    path = sup.write_support(tmp_path / "s.tif", grid, fn, block=64)
    with rasterio.open(path) as src:
        m = src.read(1)
        assert src.crs.to_string() == "EPSG:32631" and src.transform == grid.transform
        assert src.profile["driver"] == "GTiff" and m.shape == (grid.H, grid.W)
    assert set(np.unique(m)) == {0, 1}

    def at(p):
        return m[_cell(grid, model(np.array([p], float))[0])]

    assert at(g[0]) == 1 and at((150, 150)) == 1
    hx = g[:, 0].max()
    assert at((hx + 0.5 * d, g[np.argmax(g[:, 0]), 1])) == 1  # within D of the hull
    assert at((hx + 1.5 * d, g[np.argmax(g[:, 0]), 1])) == 0  # beyond D
    assert at((900, 900)) == 0 and at((900, 100)) == 0
    frac = m.mean()
    assert 0.0 < frac < 0.6
    assert sup.support_polygon(np.empty((0, 2)), EXTENT) is None


def test_load_support_roundtrip(tmp_project):
    from sbf.georef.pipeline import processed_dir
    from sbf.georef.registry import load_support

    assert load_support("NOPE", tmp_project) is None
    g, model, grid = _frame_model_and_grid()
    d = processed_dir("E1", tmp_project)
    d.mkdir(parents=True)
    sup.write_support(
        d / "E1_support.tif",
        grid,
        sup.frame_support_fn(model, SHAPE, sup.support_polygon(g, EXTENT)),
    )
    arr, tr = load_support("E1", tmp_project)
    assert arr.dtype == np.uint8 and arr.shape == (grid.H, grid.W) and tr == grid.transform


# ---------------------------------------------------------------- strips


@pytest.fixture(scope="module")
def pano():
    scene = make_scene()
    return make_reference(scene), make_panoramic_frame(scene)


def test_strips_report_cv_coverage_and_support(pano):
    ref, fr = pano
    sol = st.solve_strips(
        fr.image, fr.corners_lonlat, ref, entity_id="P", camera="KH-4B", aoi_id="REG", n_strips=4
    )
    r = sol.report
    assert r.cv_rmse_m is not None and r.random_checkpoint_rmse_m is not None
    assert 0 < r.gcp_hull_fraction <= 1 and 0 < r.gcp_block_occupancy <= 1
    assert r.included_for_change_detection == (
        r.cv_rmse_m <= 20 and r.gcp_block_occupancy >= MIN_BLOCK_OCCUPANCY
    )
    assert all(s.gcp_hull_fraction is not None and s.cv_rmse_m is not None for s in r.strips)
    assert sorted(sol.support_px) == [0, 1, 2, 3]
    res = st.warp_strips(fr.image, sol.strip_models, sol.plan, resolution_m=r.resolution_m * 2)
    grid = sup.grid_of(res.transform, res.array.shape)
    fn = st.strips_support_fn(sol.strip_models, sol.plan, sol.shape, sol.support_px)
    m = fn(grid, 0, 0, grid.H, grid.W)
    assert m.shape == res.array.shape and 0.2 < m.mean() < 1.0
    assert (m & ~(res.array > 0)).mean() < 0.02


def test_strip_fallback_is_excluded_from_support(pano, monkeypatch):
    ref, fr = pano
    real = st.find_gcps
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        return None if calls["n"] == 4 else real(*a, **kw)

    monkeypatch.setattr(st, "find_gcps", flaky)
    sol = st.solve_strips(
        fr.image, fr.corners_lonlat, ref, entity_id="P", camera="KH-4B", n_strips=4
    )
    assert 3 not in sol.support_px and sorted(sol.support_px) == [0, 1, 2]
    fn = st.strips_support_fn(sol.strip_models, sol.plan, sol.shape, sol.support_px)
    h, w = sol.shape
    xy = sol.strip_models[3](np.array([[0.95 * w, h / 2]]))[0]  # core of the fallback strip
    from sbf.georef.warp import Grid

    grid = Grid(xy[0] - 50, xy[1] + 50, 10.0, 10, 10)
    assert not fn(grid, 0, 0, 10, 10).any()
