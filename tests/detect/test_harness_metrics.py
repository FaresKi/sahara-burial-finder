import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from sbf.detect.exclusions import (
    DRAFT,
    build_draft_exclusions,
    load_exclusions,
    write_draft,
)
from sbf.detect.harness import (
    average_precision,
    evaluate,
    feature_recall_at_k,
    random_feature_recall_at_k,
    reference_footprints,
    tile_recall_at_k,
    to_markdown,
)
from sbf.features.tiling import grid_origin, make_grid


def test_ap_hand_computed():
    assert average_precision([1, 0, 1, 0]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([0, 0, 1]) == pytest.approx(1 / 3)
    assert average_precision([1, 1]) == 1.0
    assert np.isnan(average_precision([0, 0]))


def test_recall_hand_computed():
    y = [0, 1, 0, 1, 1, 0]
    assert tile_recall_at_k(y, 2) == pytest.approx(1 / 3)
    assert tile_recall_at_k(y, 5) == 1.0
    assert feature_recall_at_k([1, 4, 30], 3) == pytest.approx(1 / 3)
    assert feature_recall_at_k([1, 4, 30], 50) == 1.0
    # N=10, one feature with m=2 tiles, K=3: 1 - C(8,3)/C(10,3) = 1 - 56/120
    assert random_feature_recall_at_k(10, [2], 3) == pytest.approx(1 - 56 / 120)
    assert random_feature_recall_at_k(10, [2], 50) == 1.0


def _setup(tmp_project, order):
    """4 x 1 strip of 64 m tiles (no overlap); feature covers tile 1 only."""
    ox, oy = grid_origin("REG")
    x0, y0 = ox + 4096, oy + 4096
    grid = make_grid("REG", 64, 0.5, tmp_project, bounds=(x0, y0, x0 + 64 * 4 + 64, y0 + 64))
    grid = grid[grid.geometry.bounds.minx.mod(64) == 0].reset_index(drop=True)  # drop half steps
    assert len(grid) == 5
    ref = gpd.GeoDataFrame(
        {"feature_id": ["F1"], "aoi_id": "REG", "precision": "iaea", "expected_area_km2": [None]},
        geometry=[Point(grid.geometry[1].centroid.x, grid.geometry[1].centroid.y)],
        crs="EPSG:32631",
    ).to_crs("EPSG:4326")
    ids = grid["tile_id"].tolist()
    df = pd.DataFrame(
        {
            "tile_id": ids,
            "aoi_id": "REG",
            "size_m": 64,
            "detector": "texture",
            "layer": "L",
            "score": [order[i] for i in range(5)],
        }
    )
    df["score_norm"] = 0.5
    return df, ref, grid


def test_evaluate_hand_computed(tmp_project):
    # positive is tile index 1; scores rank it 2nd of 5
    df, ref, _ = _setup(tmp_project, [5.0, 4.0, 3.0, 2.0, 1.0])
    # tiles at half step overlap the 64 m grid points; only integer-aligned tiles are scored
    df = df.iloc[:5]
    r = evaluate(df, ref, exclusions=gpd.GeoDataFrame(), settings=tmp_project, ks=(1, 2)).iloc[0]
    assert r["n_pos_tiles"] == 1 and r["n_features"] == 1
    assert r["recall_at_2"] == 1.0 and r["recall_at_1"] == 0.0
    assert r["ap"] == pytest.approx(0.5)
    assert not r["approximate_labels"]
    assert "geometry" not in to_markdown(pd.DataFrame([r]), ks=(1, 2))


def test_exclusions_removed_before_ranking(tmp_project):
    df, ref, grid = _setup(tmp_project, [5.0, 4.0, 3.0, 2.0, 1.0])
    ex = gpd.GeoDataFrame(
        geometry=[grid.geometry[0].buffer(-1)], crs=grid.crs
    )  # top-ranked tile only
    r = evaluate(df, ref, ex, tmp_project, ks=(1,)).iloc[0]
    assert r["n_excluded"] >= 1 and r["recall_at_1"] == 1.0
    assert r["exclusions_status"] == "draft"  # no status column means unreviewed


def test_footprints_skip_null_and_flag_approx():
    ref = gpd.GeoDataFrame(
        {
            "feature_id": ["A", "B", "C"],
            "aoi_id": "REG",
            "precision": ["iaea", "approximate", "to_locate"],
            "expected_area_km2": [1.0, None, None],
        },
        geometry=[Point(0, 26), Point(0.1, 26), None],
        crs="EPSG:4326",
    )
    fp = reference_footprints("REG", ref, "EPSG:32631")
    assert list(fp["feature_id"]) == ["A", "B"]
    assert fp.geometry[0].area == pytest.approx(1e6, rel=1e-2)
    assert fp.geometry[1].area == 0


def test_draft_exclusions(tmp_project):
    gdf = build_draft_exclusions()
    assert set(gdf["status"]) == {DRAFT} and gdf["approximate"].all()
    (tmp_project.config_dir / "exclusions.geojson").unlink(missing_ok=True)
    p = write_draft(settings=tmp_project)
    assert load_exclusions(tmp_project)["status"].tolist() == [DRAFT]
    with pytest.raises(FileExistsError):
        write_draft(settings=tmp_project)
    assert p.exists() and isinstance(box(0, 0, 1, 1), object)
