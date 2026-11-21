import numpy as np
import pandas as pd
import pytest

from sbf.detect.background import fit_score
from sbf.detect.scores import (
    SCORE_COLUMNS,
    ScoreSchemaError,
    random_scores,
    rank_normalise,
    read_scores,
    validate_scores,
    write_scores,
)


def _df(n=5, **over):
    df = pd.DataFrame(
        {
            "tile_id": [f"REG-64-{i:05d}-00001" for i in range(n)],
            "aoi_id": "REG",
            "size_m": 64,
            "detector": "texture",
            "layer": "s2_median",
            "score": np.arange(n, dtype=float),
        }
    )
    for k, v in over.items():
        df[k] = v
    return df


def test_rank_normalise():
    assert rank_normalise([10, 30, 20]).tolist() == [0.0, 1.0, 0.5]
    assert rank_normalise([1, 1, 3]).tolist() == [0.25, 0.25, 1.0]


def test_write_read_roundtrip_computes_norm(tmp_project):
    paths = write_scores(_df(), tmp_project)
    assert paths[0].name == "texture__s2_median__64m.parquet"
    back = read_scores("REG", settings=tmp_project)
    assert list(back.columns) == SCORE_COLUMNS
    assert back["score_norm"].tolist() == [0, 0.25, 0.5, 0.75, 1.0]
    assert read_scores("REG", "texture", "nope", tmp_project).empty
    assert len(read_scores("REG", detector="texture", settings=tmp_project)) == 5


@pytest.mark.parametrize(
    "bad",
    [
        {"detector": "magic"},
        {"score": np.nan},
        {"aoi_id": "INE"},  # disagrees with tile_id
        {"size_m": 128},  # disagrees with tile_id
        {"tile_id": "bogus"},
        {"score_norm": 2.0},
    ],
)
def test_schema_rejects(bad):
    df = _df()
    df["score_norm"] = rank_normalise(df["score"])
    for k, v in bad.items():
        df[k] = v
    with pytest.raises(ScoreSchemaError):
        validate_scores(df)


def test_schema_missing_and_duplicates():
    with pytest.raises(ScoreSchemaError):
        validate_scores(_df().drop(columns=["layer"]))
    df = pd.concat([_df(), _df()], ignore_index=True)
    df["score_norm"] = 0.5
    with pytest.raises(ScoreSchemaError):
        validate_scores(df)


def test_random_baseline_seeded(tmp_project):
    ids = _df(20)["tile_id"].tolist()
    a, b, c = random_scores(ids, seed=1), random_scores(ids, seed=1), random_scores(ids, seed=2)
    assert a.equals(b) and not a["score"].equals(c["score"])
    assert (a["detector"] == "random").all()
    validate_scores(a)


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    X[:5] += 8
    return X


@pytest.mark.parametrize("method", ["iforest", "knn", "maha"])
def test_background_flags_outliers(blobs, method):
    s = fit_score(blobs, method, seed=0)
    assert s.shape == (300,)
    assert set(np.argsort(-s)[:5]) == set(range(5))
    assert np.array_equal(s, fit_score(blobs, method, seed=0))


def test_background_validation_and_high_dim_maha():
    with pytest.raises(ValueError):
        fit_score(np.zeros((2, 3)), "knn")
    with pytest.raises(ValueError):
        fit_score(np.random.rand(10, 2), "nope")
    X = np.random.default_rng(1).normal(size=(40, 100))  # n < 5d -> shrinkage path
    assert np.isfinite(fit_score(X, "maha")).all()


def test_tile_sizes_do_not_overwrite(tmp_project):
    from sbf.detect.scores import read_scores, write_scores

    def table(size):
        return pd.DataFrame(
            {
                "tile_id": [f"REG-{size}-00000-0000{i}" for i in range(3)],
                "aoi_id": "REG",
                "size_m": size,
                "detector": "texture",
                "layer": "s2_median",
                "score": [0.1, 0.5, 0.9],
            }
        )

    write_scores(table(64))
    write_scores(table(128))
    got = read_scores("REG", "texture", "s2_median")
    assert sorted(got["size_m"].unique().tolist()) == [64, 128]
