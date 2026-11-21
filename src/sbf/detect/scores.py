"""Score-table contract (ARCHITECTURE.md "Score table") plus the seeded random baseline.

Tables live at ``data/processed/scores/{aoi}/{detector}__{layer}.parquet``. They hold tile IDs,
never coordinates. ``score_norm`` is the rank-normalisation of ``score`` to [0, 1] computed per
(aoi, detector, layer, size_m): tile sizes have different raw score scales, so they are ranked
separately (identical to the contract whenever a table holds a single tile size).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sbf.core.config import Settings, load_settings
from sbf.features.tiling import parse_tile_id

SCORE_COLUMNS = ["tile_id", "aoi_id", "size_m", "detector", "layer", "score", "score_norm"]
DETECTORS = {
    "texture",
    "embed_iforest",
    "embed_knn",
    "embed_maha",
    "change_embed",
    "change_diff",
    "sar_z",
    "dem_relief",
    "random",
    "prior",
}
_KEY = ["aoi_id", "detector", "layer", "size_m"]


class ScoreSchemaError(ValueError):
    """The frame violates the score-table contract."""


def rank_normalise(score: Sequence[float] | np.ndarray) -> np.ndarray:
    """Average-rank normalisation to [0, 1]: lowest 0, highest 1 (a single value maps to 0.5)."""
    s = np.asarray(score, dtype="float64")
    if len(s) == 1:
        return np.array([0.5])
    return (rankdata(s, method="average") - 1) / (len(s) - 1)


def validate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Check the contract; raise `ScoreSchemaError` on the first problem, else return ``df``."""
    missing = [c for c in SCORE_COLUMNS if c not in df.columns]
    if missing:
        raise ScoreSchemaError(f"missing columns: {missing}")
    for c in ("tile_id", "aoi_id", "detector", "layer"):
        if not df[c].map(lambda v: isinstance(v, str) and v != "").all():
            raise ScoreSchemaError(f"column {c} must hold non-empty strings")
    if not pd.api.types.is_integer_dtype(df["size_m"]) or (df["size_m"] <= 0).any():
        raise ScoreSchemaError("size_m must be positive integers")
    bad = set(df["detector"]) - DETECTORS
    if bad:
        raise ScoreSchemaError(f"unknown detector(s): {sorted(bad)}")
    for c in ("score", "score_norm"):
        if not pd.api.types.is_float_dtype(df[c]) or not np.isfinite(df[c]).all():
            raise ScoreSchemaError(f"{c} must be finite floats")
    if ((df["score_norm"] < 0) | (df["score_norm"] > 1)).any():
        raise ScoreSchemaError("score_norm must lie in [0, 1]")
    try:
        parsed = [parse_tile_id(t) for t in df["tile_id"]]
    except ValueError as e:
        raise ScoreSchemaError(str(e)) from None
    if any(
        a != aoi or s != size
        for (a, s, _, _), aoi, size in zip(parsed, df["aoi_id"], df["size_m"], strict=True)
    ):
        raise ScoreSchemaError("tile_id disagrees with aoi_id/size_m")
    if df.duplicated(["tile_id", "detector", "layer"]).any():
        raise ScoreSchemaError("duplicate (tile_id, detector, layer) rows")
    return df


def _prepared(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "score" in df and pd.api.types.is_integer_dtype(df["score"]):
        df["score"] = df["score"].astype("float64")
    if "score_norm" not in df.columns or df["score_norm"].isna().any():
        if "score" not in df.columns:
            raise ScoreSchemaError("missing columns: ['score']")
        df["score_norm"] = df.groupby(_KEY, group_keys=False)["score"].transform(rank_normalise)
    if "size_m" in df and pd.api.types.is_integer_dtype(df["size_m"]):
        df["size_m"] = df["size_m"].astype("int64")
    return df[SCORE_COLUMNS].reset_index(drop=True)


def scores_path(
    aoi_id: str, detector: str, layer: str, size_m: int, settings: Settings | None = None
):
    """Parquet path for one (aoi, detector, layer, size_m) table."""
    s = settings or load_settings()
    return s.data_processed / "scores" / aoi_id / f"{detector}__{layer}__{int(size_m)}m.parquet"


def write_scores(df: pd.DataFrame, settings: Settings | None = None) -> list:
    """Validate and write one parquet per (aoi, detector, layer, size_m); returns the paths.

    ``score_norm`` is computed (rank-normalised) when absent. An existing table for the same
    key is replaced; other tile sizes of the same detector/layer are kept.
    """
    df = validate_scores(_prepared(df))
    paths = []
    for (aoi, det, layer, size), g in df.groupby(["aoi_id", "detector", "layer", "size_m"]):
        p = scores_path(aoi, det, layer, size, settings)
        p.parent.mkdir(parents=True, exist_ok=True)
        g.to_parquet(p, index=False)
        paths.append(p)
    return paths


def read_scores(
    aoi: str,
    detector: str | None = None,
    layer: str | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Read and validate score tables for an AOI (optionally one detector and/or layer).

    Returns an empty frame with ``SCORE_COLUMNS`` when nothing matches.
    """
    s = settings or load_settings()
    folder = s.data_processed / "scores" / aoi
    pattern = f"{detector or '*'}__{layer or '*'}__*m.parquet"
    files = sorted(folder.glob(pattern)) if folder.is_dir() else []
    if not files:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in SCORE_COLUMNS})
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return validate_scores(df)


def random_scores(tile_ids: Sequence[str], layer: str = "none", seed: int = 0) -> pd.DataFrame:
    """Seeded random-baseline score table (detector ``random``) for the given tiles."""
    ids = list(tile_ids)
    parsed = [parse_tile_id(t) for t in ids]
    aois = {p[0] for p in parsed}
    sizes = {p[1] for p in parsed}
    if len(aois) != 1 or len(sizes) != 1:
        raise ValueError("random_scores needs tiles of a single AOI and tile size")
    score = np.random.default_rng(seed).random(len(ids))
    df = pd.DataFrame(
        {
            "tile_id": ids,
            "aoi_id": aois.pop(),
            "size_m": sizes.pop(),
            "detector": "random",
            "layer": layer,
            "score": score,
        }
    )
    return _prepared(df)
