import re
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from typer.testing import CliRunner

from sbf.core.restricted import restricted_path
from sbf.detect.scores import write_scores
from sbf.features.tiling import make_tile_id
from sbf.fuse.candidates import (
    LABEL_MULTIPLIER,
    build_candidates,
    date_bracket,
    rank_candidates,
    select_top_tiles,
)
from sbf.fuse.cli import app
from sbf.fuse.families import FAMILIES, family_of, family_scores, fuse, weights_from_metrics
from sbf.fuse.pipeline import run_fusion, write_outputs


def T(c, r, size=128):
    return make_tile_id("REG", size, c, r)


def test_family_mapping():
    assert family_of("change_diff", "pair_a_b") == "hist_change"
    assert family_of("change_embed", "kh7_x") is None
    assert family_of("texture", "kh7_ds1") == "hist_anomaly"
    assert family_of("embed_knn", "s2_median") == "modern_optical"
    assert family_of("texture", "palsar_hv") is None
    assert family_of("sar_z", "palsar_hv") == "SAR_L"
    assert family_of("sar_z", "s1_vv") == "SAR_C"
    assert family_of("dem_relief", "dem_srtm") == "DEM"
    assert family_of("prior", "docs_prior") == "prior"
    assert family_of("random", "none") is None


def test_weights_from_metrics_non_beating_zero():
    m = pd.DataFrame(
        {
            "detector": ["texture", "embed_knn", "sar_z", "dem_relief", "prior"],
            "layer": ["kh7_a", "kh7_a", "palsar_hv", "dem", "docs_prior"],
            "ap": [0.5, 0.3, 0.01, 0.02, np.nan],
            "random_ap": [0.1, 0.1, 0.05, 0.02, 0.1],
        }
    )
    w = weights_from_metrics(m)
    assert w["hist_anomaly"] == pytest.approx(1.0)  # best detector (0.4) represents the family
    assert w["SAR_L"] == 0 and w["DEM"] == 0 and w["prior"] == 0
    m.loc[0, "ap"] = 0.3
    m.loc[2, "ap"] = 0.25
    w = weights_from_metrics(m)
    assert w["hist_anomaly"] == pytest.approx(0.2 / 0.4) and w["SAR_L"] == pytest.approx(0.2 / 0.4)


def test_fusion_renormalises_over_present_families():
    fam = pd.DataFrame(
        {"hist_anomaly": [1.0, 1.0], "DEM": [0.0, np.nan]},
        index=pd.Index(["a", "b"], name="tile_id"),
    )
    out = fuse(fam, {"hist_anomaly": 1.0, "DEM": 1.0})
    assert out.loc["a", "S"] == pytest.approx(0.5) and out.loc["a", "n_families"] == 2
    assert out.loc["b", "S"] == pytest.approx(1.0) and out.loc["b", "n_families"] == 1
    with pytest.raises(ValueError):
        fuse(fam, {"SAR_L": 1.0})


def test_family_scores_max_vs_mean():
    df = pd.DataFrame(
        {
            "tile_id": [T(0, 0)] * 2,
            "aoi_id": "REG",
            "size_m": 128,
            "detector": "texture",
            "layer": ["kh7_a", "kh7_b"],
            "score": [1.0, 2.0],
            "score_norm": [0.2, 0.6],
        }
    )
    assert family_scores(df, 128, "max").iloc[0]["hist_anomaly"] == pytest.approx(0.6)
    assert family_scores(df, 128, "mean").iloc[0]["hist_anomaly"] == pytest.approx(0.4)


def test_polygon_merge_adjacent_tiles(tmp_project):
    ids = [T(5, 5), T(6, 5), T(20, 20)]  # first two overlap (50%), third isolated
    top = pd.DataFrame({"S": [0.9, 0.8, 0.7], "n_families": 1}, index=pd.Index(ids, name="tile_id"))
    fam = pd.DataFrame({"hist_anomaly": [0.9, 0.9, 0.1]}, index=top.index)
    gdf, members = build_candidates(top, fam, settings=tmp_project)
    assert len(gdf) == 2
    big = gdf.sort_values("n_tiles").iloc[-1]
    assert big.n_tiles == 2 and big.area_m2 == pytest.approx(128 * 192)
    assert big.mean_S == pytest.approx(0.85) and big.families == "hist_anomaly"
    assert sorted(t for v in members.values() for t in v) == sorted(ids)


def test_select_top_tiles_excludes():
    f = pd.DataFrame(
        {"S": np.linspace(0, 1, 100), "n_families": 1}, index=[f"t{i}" for i in range(100)]
    )
    f.index.name = "tile_id"
    assert len(select_top_tiles(f, [], 1.0)) == 1
    assert select_top_tiles(f, ["t99"], 2.0).index[0] == "t98"


REG = pd.DataFrame(
    {
        "entity_id": ["E0", "E1", "E2", "E3"],
        "acquisition_date": [
            date(1959, 11, 1),
            date(1962, 5, 1),
            date(1965, 5, 1),
            date(1971, 3, 1),
        ],
        "era": ["pre_1967"] * 3 + ["post_1967"],
    }
)


def _layers(vals, ids=("t1", "t2"), extra=None):
    rows = []
    for layer, v in vals.items():
        for t in ids:
            rows.append(
                {"tile_id": t, "aoi_id": "REG", "size_m": 128, "detector": "texture",
                 "layer": layer, "score": v, "score_norm": v}
            )  # fmt: skip
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("vals", "label", "first", "absent"),
    [
        (
            {"kh7_E0": 0.95, "kh7_E1": 0.95, "kh7_E2": 0.95},
            "pre_1960_natural_or_unknown",
            date(1959, 11, 1),
            None,
        ),
        (
            {"kh7_E0": 0.1, "kh7_E1": 0.1, "kh7_E2": 0.9, "kh7_E3": 0.9},
            "ops_1961_1968",
            date(1965, 5, 1),
            date(1962, 5, 1),
        ),
        ({"kh7_E1": 0.95, "kh7_E2": 0.9}, "ops_1961_1968", date(1962, 5, 1), None),
        (
            {"kh7_E1": 0.1, "kh7_E2": 0.1, "kh7_E3": 0.9},
            "ops_1961_1968",
            date(1971, 3, 1),
            date(1965, 5, 1),
        ),  # bracket overlaps window
        (
            {"kh7_E0": 0.1, "kh7_E1": 0.1, "kh7_E2": 0.1, "kh7_E3": 0.9},
            "ops_1961_1968",
            date(1971, 3, 1),
            date(1965, 5, 1),
        ),
        ({"kh7_E0": 0.1}, "other", None, date(1959, 11, 1)),
    ],
)
def test_date_bracket_labels(vals, label, first, absent):
    r = date_bracket(["t1", "t2"], _layers(vals), REG)
    assert r["label"] == label and r["first_seen"] == first and r["last_absent_before"] == absent


def test_label_other_when_bracket_misses_window():
    reg = REG.copy()
    reg.loc[3, "acquisition_date"] = date(1975, 1, 1)
    reg.loc[2, "acquisition_date"] = date(1970, 1, 1)
    vals = {"kh7_E2": 0.1, "kh7_E3": 0.9}
    assert date_bracket(["t1"], _layers(vals), reg)["label"] == "other"


def test_label_post_2025():
    s = pd.concat(
        [
            _layers({"kh7_E1": 0.1, "kh7_E2": 0.1}),
            _layers({"s2_new": 0.95, "s2_old": 0.1}).assign(detector="texture"),
        ]
    )
    md = {"s2_new": "2025-09-01", "s2_old": "2023-01-01"}
    assert date_bracket(["t1"], s, REG, md)["label"] == "post_2025_likely_cleanup"
    md_old = {"s2_new": "2023-09-01", "s2_old": "2022-01-01"}
    assert date_bracket(["t1"], s, REG, md_old)["label"] == "other"


def test_ranking_multiplier():
    c = gpd.GeoDataFrame(
        {"cand_id": ["a", "b", "c"], "mean_S": [0.9, 0.8, 0.95]},
        geometry=[box(0, 0, 1, 1)] * 3,
        crs="EPSG:32631",
    )
    br = {
        "a": {
            "label": "post_2025_likely_cleanup",
            "first_seen": None,
            "last_absent_before": None,
            "bracket": "",
        },
        "b": {
            "label": "ops_1961_1968",
            "first_seen": None,
            "last_absent_before": None,
            "bracket": "",
        },
        "c": {
            "label": "pre_1960_natural_or_unknown",
            "first_seen": None,
            "last_absent_before": None,
            "bracket": "",
        },
    }
    r = rank_candidates(c, br)
    assert list(r["cand_id"]) == ["b", "c", "a"]
    assert r.loc[0, "final_rank_score"] == pytest.approx(0.8 * LABEL_MULTIPLIER["ops_1961_1968"])


# ------------------------------------------------------------------ synthetic end to end
def _synthetic(tmp_project):
    rng = np.random.default_rng(0)
    n = 20
    tiles = [(c, r) for c in range(n) for r in range(n)]
    planted = {(10, 10), (11, 10), (10, 11), (11, 11)}  # appears 1965, absent 1959/1962
    decoy = {(3, 3), (4, 3), (3, 4), (4, 4)}  # present in every frame (incl. 1959)
    rows = []

    def add(det, layer, hot):
        base = rng.random(len(tiles))
        for i, (c, r) in enumerate(tiles):
            hi = (c, r) in hot
            rows.append(
                {"tile_id": T(c, r), "aoi_id": "REG", "size_m": 128, "detector": det,
                 "layer": layer, "score": 10.0 + rng.random() if hi else base[i]}
            )  # fmt: skip

    both = planted | decoy
    add("texture", "kh7_E0", decoy)
    add("texture", "kh7_E1", decoy)
    add("texture", "kh7_E2", both)
    add("texture", "kh7_E3", both)
    add("change_diff", "pair_E0_E1", decoy)
    add("change_diff", "pair_E1_E2", planted)
    write_scores(pd.DataFrame(rows), tmp_project)
    from sbf.detect.scores import read_scores

    return read_scores("REG", settings=tmp_project)


W = {"hist_anomaly": 1.0, "hist_change": 1.0}


def test_end_to_end_planted_ranks_first(tmp_project):
    scores = _synthetic(tmp_project)
    res = run_fusion(
        "REG",
        scores,
        REG,
        weights=W,
        top_pct=2.0,
        exclusions=gpd.GeoDataFrame(),
        settings=tmp_project,
    )
    r = res.ranked
    assert len(r) == 2
    top = r.iloc[0]
    assert top["label"] == "ops_1961_1968" and top["n_tiles"] == 4
    assert top["first_seen"] == "1965-05-01" and top["last_absent_before"] == "1962-05-01"
    assert r.iloc[1]["label"] == "pre_1960_natural_or_unknown"
    assert top["multiplier"] == 1.5

    paths = write_outputs(res, {"top_pct": 2.0}, tmp_project)
    base = tmp_project.outputs_restricted.resolve()
    assert all(p.resolve().is_relative_to(base) for p in paths)
    assert (base / "candidates_REG.gpkg").exists() and (base / "ranking.csv").exists()
    # nothing but the two public aggregate files outside restricted
    pub = sorted(p.name for p in tmp_project.outputs_public.iterdir())
    assert pub == ["fusion_summary.md", "fusion_weights.md"]
    for p in tmp_project.outputs_public.iterdir():
        txt = p.read_text()
        assert not re.search(r"\d+\.\d{4,}", txt)  # decimal degrees
        assert not re.search(r"\d{6,}", txt)  # UTM-like
        assert "REG-128" not in txt  # no tile ids
    summary = (tmp_project.outputs_public / "fusion_summary.md").read_text()
    assert "ops_1961_1968 | 1" in summary
    back = gpd.read_file(base / "candidates_REG.gpkg")
    assert len(back) == 2


def test_restricted_path_refuses_outside(tmp_project):
    from sbf.core.restricted import RestrictedPathError

    with pytest.raises(RestrictedPathError):
        restricted_path("..", "x.csv", settings=tmp_project)


def test_cli_refuses_without_flag(tmp_project):
    res = CliRunner().invoke(app, ["run", "--aoi", "REG"])
    assert res.exit_code == 2
    assert "human sign-off" in res.output
    assert not any(tmp_project.outputs_restricted.iterdir())


# ------------------------------------------------------------------ family means / contributions
def test_family_means_and_contributions_sum(tmp_project):
    ids = [T(5, 5), T(6, 5)]
    top = pd.DataFrame({"S": [0.9, 0.8], "n_families": 2}, index=pd.Index(ids, name="tile_id"))
    fam = pd.DataFrame(
        {"hist_anomaly": [0.9, 0.7], "DEM": [0.5, np.nan], "prior": [0.4, 0.4]}, index=top.index
    )
    w = {"hist_anomaly": 2.0, "DEM": 1.0, "prior": 1.0, "SAR_L": 3.0}
    gdf, _ = build_candidates(top, fam, settings=tmp_project, weights=w)
    r = gdf.iloc[0]
    assert r["fam_hist_anomaly"] == pytest.approx(0.8)
    assert r["fam_DEM"] == pytest.approx(0.5)  # NaN tile skipped
    assert r["fam_prior"] == pytest.approx(0.4)
    assert np.isnan(r["fam_SAR_L"]) and np.isnan(r["contrib_SAR_L"])  # absent -> NaN
    contrib = r[[c for c in gdf.columns if c.startswith("contrib_")]].dropna()
    assert contrib.sum() == pytest.approx(1.0)
    assert r["contrib_hist_anomaly"] == pytest.approx(1.6 / (1.6 + 0.5 + 0.4))
    assert r["contrib_prior"] == pytest.approx(0.4 / 2.5)


def test_fusion_columns_in_gpkg_and_ranking(tmp_project):
    scores = _synthetic(tmp_project)
    res = run_fusion(
        "REG", scores, REG, weights=W, top_pct=2.0, exclusions=gpd.GeoDataFrame(),
        settings=tmp_project,
    )  # fmt: skip
    top = res.ranked.iloc[0]
    assert top["n_frames"] == 4 and top["n_present"] == 2  # planted: E2, E3 present
    assert top["fam_hist_anomaly"] > 0.8 and np.isnan(top["fam_prior"])
    cc = top[[f"contrib_{f}" for f in FAMILIES]].astype(float)
    assert cc.sum() == pytest.approx(1.0)
    write_outputs(res, {"top_pct": 2.0}, tmp_project)
    base = tmp_project.outputs_restricted
    back = gpd.read_file(base / "candidates_REG.gpkg")
    csv = pd.read_csv(base / "ranking.csv")
    for tab in (back, csv):
        for f in ("hist_anomaly", "prior", "SAR_L", "DEM"):
            assert f"fam_{f}" in tab.columns and f"contrib_{f}" in tab.columns
        assert "n_present" in tab.columns and "n_frames" in tab.columns
    assert back.loc[back["cand_id"] == top["cand_id"], "n_present"].iloc[0] == 2
