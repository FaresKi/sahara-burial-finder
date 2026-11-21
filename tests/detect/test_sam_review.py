import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from sbf.detect.sam_review import FakeSegmenter, Sam2Segmenter, run_sam_review
from sbf.features.tiling import grid_origin, make_grid


def _scores(tmp_project, n=6):
    ox, oy = grid_origin("REG")
    g = make_grid(
        "REG",
        128,
        0.0,
        tmp_project,
        bounds=(ox + 8192, oy + 8192, ox + 8192 + 128 * n, oy + 8192 + 128),
    )
    g = g[:n]
    return pd.DataFrame(
        {
            "tile_id": g["tile_id"],
            "aoi_id": "REG",
            "size_m": 128,
            "detector": "texture",
            "layer": "L",
            "score": np.arange(n, dtype=float),
            "score_norm": 0.5,
        }
    )


def test_outputs_only_restricted(tmp_project):
    sc = _scores(tmp_project)
    out = run_sam_review(
        sc,
        FakeSegmenter(0.5),
        lambda b: np.ones((8, 8), "float32"),
        "REG",
        "texture",
        "L",
        128,
        top_k=3,
        settings=tmp_project,
    )
    assert out is not None and out.is_relative_to(tmp_project.outputs_restricted.resolve())
    gdf = gpd.read_file(out)
    assert set(gdf["tile_id"]) == set(sc.sort_values("score").tail(3)["tile_id"])
    assert not any(tmp_project.outputs_public.rglob("*.gpkg"))


def test_no_polygons_writes_nothing(tmp_project):
    out = run_sam_review(
        _scores(tmp_project),
        FakeSegmenter(0.5),
        lambda b: np.zeros((8, 8), "float32"),
        "REG",
        "texture",
        "L",
        128,
        settings=tmp_project,
    )
    assert out is None and not list(tmp_project.outputs_restricted.rglob("*.gpkg"))


def test_sam2_missing_checkpoint_is_clear(monkeypatch):
    monkeypatch.delenv("SBF_SAM2_CHECKPOINT", raising=False)
    with pytest.raises(RuntimeError, match="human step"):
        Sam2Segmenter()
