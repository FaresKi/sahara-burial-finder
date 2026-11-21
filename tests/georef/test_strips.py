"""Ticket 08: panoramic strips on a synthetic 1/cos^2 + bowing distortion."""

import numpy as np
import pytest
from georef_synth import make_panoramic_frame, make_reference, make_scene

from sbf.georef import fit as fitmod
from sbf.georef import strips as st
from sbf.georef.pipeline import solve_frame


@pytest.fixture(scope="module")
def pano():
    scene = make_scene()
    return make_reference(scene), make_panoramic_frame(scene)


def _grid_pts(fr, step=40):
    h, w = fr.image.shape
    return np.array([(x, y) for x in range(0, w + 1, step) for y in range(0, h + 1, step)], float)


def test_plan_strips_overlap():
    p = st.plan_strips((400, 1200), 8, 0.1)
    assert p.axis == "x" and p.n == 8 and p.ranges[0][0] == 0 and p.ranges[-1][1] == 1200
    assert p.overlap_px == pytest.approx(15.0)
    a = p.ranges[0][1] - p.ranges[1][0]
    assert a == pytest.approx(15.0)
    t = np.array([p.ranges[1][0] + 7.5])
    w1, w0 = st.strip_weights(p, 1, t), st.strip_weights(p, 0, t)
    assert w1[0] == pytest.approx(0.5, abs=1e-3) and w0[0] == pytest.approx(0.5, abs=1e-3)
    assert st.strip_weights(p, 0, np.array([5.0]))[0] == 1.0  # outer frame edge is not feathered
    assert st.strip_weights(p, 1, np.array([0.0]))[0] == 0.0  # outside the strip


def test_frame_wide_poly_fails_but_strips_pass(pano):
    ref, fr = pano
    g = _grid_pts(fr)
    xy = fr.truth(g)
    for m in ("poly2", "poly3"):  # even with perfect GCPs the frame-wide polynomial misses
        mod = fitmod.fit_transform(g, xy, m)
        assert fitmod.rmse(fitmod.residuals_m(mod, g, xy)) > 20.0, m
    wide = solve_frame(fr.image, fr.corners_lonlat, ref, entity_id="PANO", camera="KH-4B")
    sol = st.solve_strips(
        fr.image, fr.corners_lonlat, ref, entity_id="PANO", camera="KH-4B", aoi_id="REG", n_strips=4
    )
    r = sol.report
    assert r.method == "strips" and len(r.strips) == 4 and r.strip_axis == "x"
    assert r.checkpoint_rmse_m < 10.0, r.model_dump()
    assert r.included_for_change_detection
    assert (
        "panoramic_needs_strips" not in r.warnings
        and "panoramic_needs_strips" in wide.report.warnings
    )
    assert r.candidate_rmse_m["frame_wide"] == wide.report.checkpoint_rmse_m
    assert all("strip_fallback" not in x.warnings and x.checkpoint_rmse_m < 12 for x in r.strips)
    # pooled checkpoint RMSE = RMSE over every check row of the GCP table
    chk = sol.gcps[sol.gcps["role"] == "check"]
    assert r.n_checkpoints == len(chk) == sum(x.n_checkpoints for x in r.strips)
    assert r.checkpoint_rmse_m == pytest.approx(np.sqrt((chk["residual_m"] ** 2).mean()), abs=1e-2)
    # Against ground truth over the whole frame the frame-wide model is wrong (its checkpoints sit
    # where the GCPs are, not at the scan edges) while the strips follow the 1/cos^2 profile.
    g = _grid_pts(fr)
    wide_err = fitmod.rmse(fitmod.residuals_m(wide.final_model, g, fr.truth(g)))
    se = []
    for k, m in enumerate(sol.strip_models):
        a, b = sol.plan.ranges[k]
        sel = g[(g[:, 0] >= a + 0.1 * (b - a)) & (g[:, 0] <= b - 0.1 * (b - a))]
        se.append(fitmod.residuals_m(m, sel, fr.truth(sel)))
    strip_err = fitmod.rmse(np.concatenate(se))
    assert wide_err > 20.0 and strip_err < wide_err / 2, (wide_err, strip_err)
    inner = [np.sqrt(np.mean(e**2)) for e in se[1:3]]
    assert max(inner) < 6.0
    res = st.warp_strips(fr.image, sol.strip_models, sol.plan, resolution_m=r.resolution_m * 2)
    assert res.array.dtype == np.uint8 and (res.array > 0).mean() > 0.5


def test_strips_stream_to_file_matches_memory(pano, tmp_path):
    import rasterio

    _, fr = pano
    plan = st.plan_strips(fr.image.shape, 4, 0.1)
    g = _grid_pts(fr, 60)
    models = []
    for k in range(4):
        a, b = plan.ranges[k]
        sel = g[(g[:, 0] >= a) & (g[:, 0] <= b)]
        models.append(fitmod.fit_transform(sel, fr.truth(sel), "poly3"))
    mem = st.warp_strips(fr.image, models, plan, resolution_m=6.0)
    wf = st.warp_strips_to_file(
        fr.image, models, plan, tmp_path / "s.tif", resolution_m=6.0, block=200
    )
    with rasterio.open(wf.path) as s:
        disk = s.read(1)
    assert disk.shape == mem.array.shape
    d = np.abs(disk.astype(int) - mem.array.astype(int))
    assert (d > 8).mean() < 0.01  # lattice phase differs per block: only edge pixels may flip
    # seams are invisible with exact models: blended result matches single exact warp closely
    from sbf.georef.warp import warp_frame

    full = warp_frame(
        fr.image,
        fitmod.fit_transform(g, fr.truth(g), "tps"),
        resolution_m=6.0,
        bounds=(
            wf.transform.c,
            wf.transform.f - 6.0 * disk.shape[0],
            wf.transform.c + 6.0 * disk.shape[1],
            wf.transform.f,
        ),
    )
    both = (full.array > 0) & (mem.array > 0)
    both[:4] = both[-4:] = False
    both[:, :4] = both[:, -4:] = False
    assert both.mean() > 0.3
    assert np.abs(full.array[both].astype(int) - mem.array[both].astype(int)).mean() < 6


def test_strip_fallback_inherits_neighbour(pano, monkeypatch):
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
    flagged = [s for s in sol.report.strips if "strip_fallback" in s.warnings]
    # solve order is centre-out (1, 2, 0, 3): the 4th strip call (index 3) fails
    assert [s.index for s in flagged] == [3] and flagged[0].fallback_from == "strip:2"
    assert "strip_fallback" in sol.report.warnings
    assert all(m is not None for m in sol.strip_models)
