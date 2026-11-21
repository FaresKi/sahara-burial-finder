import json

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from typer.testing import CliRunner

from sbf.detect.scores import write_scores
from sbf.features.tiling import grid_origin, make_grid
from sbf.report import blind_test as bt
from sbf.report.blind_test_figures import MASK_RGB, recall_curve, score_panel
from sbf.report.cli import app


def row(
    detector="texture",
    layer="kh4_D1",
    aoi="REG",
    size=64,
    rec=0.8,
    rrec=0.1,
    ap=0.3,
    rap=0.05,
    n=5,
    status="reviewed",
):
    return {
        "aoi_id": aoi,
        "detector": detector,
        "layer": layer,
        "size_m": size,
        "n_tiles": 1000,
        "n_pos_tiles": 20,
        "n_features": n,
        "approximate_labels": True,
        "recall_at_10": rec,
        "random_recall_at_10": rrec,
        "recall_at_50": rec,
        "random_recall_at_50": rrec,
        "recall_at_100": rec,
        "random_recall_at_100": rrec,
        "ap": ap,
        "random_ap": rap,
        "exclusions_status": status,
    }


def gate(*rows):
    return bt.evaluate_gate(pd.DataFrame(list(rows)))


def test_pass_and_fail_edges():
    assert gate(row()).status == "PASS"
    assert gate(row(rec=0.49, rrec=0.0)).status == "FAIL"  # recall too low
    assert gate(row(rec=0.5, rrec=0.31)).status == "FAIL"  # margin 0.19
    assert gate(row(rec=0.5, rrec=0.3)).status == "PASS"  # boundaries inclusive
    assert gate(row(ap=0.09, rap=0.05)).status == "FAIL"  # AP < 2x random
    assert gate(row(ap=0.10, rap=0.05)).status == "PASS"
    assert gate(row(rec=float("nan"))).status == "FAIL"
    assert gate(row(rap=0.0)).status == "FAIL"


def test_modern_only_and_random_prior_do_not_pass():
    r = gate(row(detector="texture", layer="s2_median"))
    assert r.status == "NOT_RUN" and r.modern_best is not None
    assert gate(row(detector="sar_z", layer="s1_vv")).status == "NOT_RUN"
    r = gate(row(detector="random", layer="kh4_D1"), row(detector="prior", layer="kh4_D1"))
    assert r.status == "NOT_RUN"
    r = gate(row(layer="s2_median"), row(rec=0.1, rrec=0.1, layer="kh4_D1"))
    assert r.status == "FAIL" and r.modern_best["layer"] == "s2_median"
    assert gate(row(detector="change_diff", layer="pair_kh4_s2_median")).status == "PASS"
    assert bt.evaluate_gate(pd.DataFrame()).status == "NOT_RUN"


def test_gate_small_n_warning_and_margin():
    r = gate(row(n=3), row(aoi="INE", n=9, rec=0.6, rrec=0.1))
    assert r.n_reference_features_scored == {"REG": 3, "INE": 9}
    assert any("REG" in w for w in r.warnings) and not any("INE" in w for w in r.warnings)
    assert r.margin == pytest.approx(0.7)
    assert r.best_rows[0]["aoi"] == "REG"


def test_gate_json_schema(tmp_project):
    res = gate(row(status="draft"))
    p = bt.write_gate(res, tmp_project)
    d = json.loads(p.read_text())
    assert {"status", "criteria", "best_rows", "generated_at", "exclusions_status"} <= set(d)
    assert d["status"] == "PASS" and d["exclusions_status"] == "draft"
    assert set(d["best_rows"][0]) == {"detector", "layer", "aoi", "size_m", "metrics"}
    assert "geometry" not in p.read_text() and "tile_id" not in p.read_text()


def test_report_pass_and_fail_text():
    ok = pd.DataFrame([row(status="draft")])
    txt = bt.build_report(ok, bt.evaluate_gate(ok))
    assert "Decision: PASS" in txt and "Honest interpretation" in txt
    assert "PROVISIONAL" in txt and "Negative result" not in txt
    bad = pd.DataFrame([row(rec=0.1, rrec=0.1, status="reviewed")])
    txt = bt.build_report(bad, bt.evaluate_gate(bad))
    assert "Negative result" in txt and bt.STOP_NOTICE in txt
    for t in (txt, bt.build_report(ok, bt.evaluate_gate(ok))):
        assert "waste found" not in t.lower()
        assert "label noise" in t.lower() and "invisible" in t.lower()


def test_figure_does_not_leak_outside_footprint():
    foot = box(0, 0, 64, 64)
    inside = (0.0, 0.0, 64.0, 64.0)
    outside = (192.0, 0.0, 256.0, 64.0)  # planted high-score tile away from the footprint
    bounds = (-100.0, -100.0, 364.0, 164.0)
    panel = score_panel(foot, {"in": inside, "out": outside}, {"in": 1.0, "out": 99.0}, bounds, 4.0)
    out_px = panel[(164 - 60) // 4, (192 + 100 + 8) // 4]  # inside the outside tile (row from top)
    assert tuple(out_px) == MASK_RGB
    in_px = panel[(164 - 32) // 4, (100 + 32) // 4]
    assert tuple(in_px) != MASK_RGB
    # no tile touches the footprint -> entirely masked
    allmask = score_panel(foot, {"out": outside}, {"out": 5.0}, bounds, 4.0)
    assert (allmask == np.array(MASK_RGB, dtype=np.uint8)).all()


def test_recall_curve_uses_positive_ranks_only():
    _ks, obs, rnd = recall_curve([1, 30, 500], 1000, [2, 2, 2])
    assert (
        obs[0] == pytest.approx(1 / 3)
        and obs[29] == pytest.approx(2 / 3)
        and obs[-1] == pytest.approx(2 / 3)
    )
    assert (np.diff(rnd) >= 0).all()


# ----------------------------------------------------------------------------- CLI
def _seed_scores(s):
    ox, oy = grid_origin("REG")
    g = make_grid("REG", 64, 0.0, s, bounds=(ox + 4096, oy + 4096, ox + 4096 + 256, oy + 4096 + 64))
    df = pd.DataFrame(
        {
            "tile_id": g["tile_id"],
            "aoi_id": "REG",
            "size_m": 64,
            "detector": "texture",
            "layer": "kh4_D1",
            "score": np.arange(len(g), dtype=float),
        }
    )
    write_scores(df, s)


@pytest.mark.parametrize(("rec", "code"), [(0.9, 0), (0.1, 4)])
def test_cli_exit_codes(tmp_project, monkeypatch, rec, code):
    _seed_scores(tmp_project)
    monkeypatch.setattr(
        "sbf.detect.harness.evaluate",
        lambda *a, **k: pd.DataFrame([row(rec=rec, rrec=0.1)]),
    )
    res = CliRunner().invoke(app, ["blind-test", "--no-figures"])
    assert res.exit_code == code, res.output
    d = json.loads((tmp_project.outputs_public / "blind_test_gate.json").read_text())
    assert d["status"] == ("PASS" if code == 0 else "FAIL")
    assert (tmp_project.outputs_public / "blind_test.md").exists()
    assert ("STOP" in res.output) == (code == 4)


def test_cli_not_run_without_scores(tmp_project):
    res = CliRunner().invoke(app, ["blind-test", "--no-figures"])
    assert res.exit_code == 5
    d = json.loads((tmp_project.outputs_public / "blind_test_gate.json").read_text())
    assert d["status"] == "NOT_RUN"


def test_cli_sam_review_requires_flag(tmp_project):
    res = CliRunner().invoke(
        app, ["sam-review", "--aoi", "REG", "--detector", "texture", "--layer", "L"]
    )
    assert res.exit_code == 2


def test_write_figures_synthetic(tmp_project):
    import geopandas as gpd
    from shapely.geometry import Point

    from sbf.report.blind_test_figures import write_figures

    ox, oy = grid_origin("REG")
    x0, y0 = ox + 4096, oy + 4096
    g = make_grid("REG", 64, 0.5, tmp_project, bounds=(x0, y0, x0 + 320, y0 + 64))
    g = g[g.geometry.bounds.minx.mod(64) == 0].reset_index(drop=True)
    ref = gpd.GeoDataFrame(
        {"feature_id": ["F1"], "aoi_id": "REG", "precision": "iaea", "expected_area_km2": [None]},
        geometry=[Point(g.geometry[1].centroid.x, g.geometry[1].centroid.y)],
        crs="EPSG:32631",
    ).to_crs("EPSG:4326")
    sc = pd.DataFrame(
        {"tile_id": g["tile_id"], "aoi_id": "REG", "size_m": 64, "detector": "texture",
         "layer": "kh4_D1", "score": [5.0, 4.0, 3.0, 2.0, 1.0], "score_norm": 0.5}
    )  # fmt: skip
    paths = write_figures(sc, ref, None, tmp_project)
    names = {p.name for p in paths}
    assert {"blind_test_recall_curves.png", "blind_test_rank_hist.png"} <= names
    assert all(p.is_relative_to(tmp_project.outputs_public) for p in paths)
