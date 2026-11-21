"""Embedding anomaly detector: tile chips -> encoder -> cached embeddings -> background models.

Cache layout (resumable, streamed): ``data/interim/embeddings/{aoi}/{layer}__{size}__{encoder}/``
holds ``part-000001.parquet, ...``; each shard has ``tile_id`` (string) and ``emb``
(fixed-size list<float32>). Re-running skips tile_ids already cached. Shards are written
atomically, so an interrupted run loses at most the current shard.

Scoring fits `sbf.detect.background.fit_score` per (aoi, layer, size) on the cached embeddings,
optionally after a seeded PCA (default 64 dims; embeddings are 768-d, far above the sample sizes
where Mahalanobis/kNN behave). Labels are never touched (D7); only counts are logged (D8).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio

from sbf.core.config import Settings, load_settings
from sbf.detect.background import METHODS, fit_score
from sbf.detect.scores import _prepared, validate_scores, write_scores
from sbf.features.embeddings import TileEncoder
from sbf.features.tiling import make_grid, tile_windows

log = logging.getLogger(__name__)
MIN_TILE_PX = 16
DEFAULT_PCA = 64


def embedding_dir(
    aoi_id: str, layer: str, size_m: int, encoder: str, settings: Settings | None = None
) -> Path:
    """Cache directory for one (aoi, layer, size, encoder)."""
    s = settings or load_settings()
    return s.data_interim / "embeddings" / aoi_id / f"{layer}__{int(size_m)}__{encoder}"


def _shards(folder: Path) -> list[Path]:
    return sorted(folder.glob("part-*.parquet")) if folder.is_dir() else []


def cached_ids(folder: Path) -> set[str]:
    """Tile IDs already cached (reads only the id column)."""
    ids: set[str] = set()
    for p in _shards(folder):
        ids.update(pq.read_table(p, columns=["tile_id"]).column("tile_id").to_pylist())
    return ids


def load_embeddings(folder: Path) -> tuple[list[str], np.ndarray]:
    """All cached ``(tile_ids, (n, d) float32)``, de-duplicated by tile_id (first wins)."""
    ids: list[str] = []
    mats: list[np.ndarray] = []
    for p in _shards(folder):
        t = pq.read_table(p)
        col = t.column("emb").combine_chunks()
        d = col.type.list_size
        mats.append(col.values.to_numpy(zero_copy_only=False).reshape(len(t), d))
        ids.extend(t.column("tile_id").to_pylist())
    if not ids:
        return [], np.zeros((0, 0), dtype="float32")
    X = np.concatenate(mats).astype("float32")
    _, first = np.unique(np.asarray(ids), return_index=True)
    keep = np.sort(first)
    return [ids[i] for i in keep], X[keep]


def _write_shard(folder: Path, ids: list[str], emb: np.ndarray) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    n = max((int(p.stem.split("-")[1]) for p in _shards(folder)), default=0) + 1
    arr = pa.FixedSizeListArray.from_arrays(
        pa.array(np.ascontiguousarray(emb, dtype="float32").ravel()), emb.shape[1]
    )
    tmp = folder / f"part-{n:06d}.tmp"
    pq.write_table(pa.table({"tile_id": pa.array(ids), "emb": arr}), str(tmp))
    tmp.rename(folder / f"part-{n:06d}.parquet")


def check_tile_scale(size_m: float, pixel_m: float) -> bool:
    """Warn (and return False) when a tile is under 16 px across: embeddings would be upsampling."""
    px = size_m / pixel_m
    if px < MIN_TILE_PX:
        msg = (
            f"tile is only {px:.1f} px across (<{MIN_TILE_PX}): embeddings will mostly encode "
            "upsampling noise. Use 128 m tiles for modern layers (S2 10 m) and 64 m for "
            "historical frames of 3 m or finer."
        )
        warnings.warn(msg, stacklevel=3)
        log.warning(msg)
        return False
    return True


def compute_embeddings(
    aoi_id: str,
    layer: str,
    raster_path: str | Path,
    encoder: TileEncoder,
    size_m: int = 128,
    band: int = 1,
    min_valid_frac: float = 0.5,
    max_tiles: int | None = None,
    batch_size: int = 32,
    shard_tiles: int = 2048,
    progress: bool = True,
    settings: Settings | None = None,
) -> dict[str, int]:
    """Embed tiles of the raster footprint into the cache; resumable; returns counts.

    Only tiles fully inside the raster bounds are considered (``make_grid(bounds=...)``), tiles
    below ``min_valid_frac`` valid pixels are skipped, and tile_ids already cached are not
    re-encoded. ``max_tiles`` caps NEW tiles encoded in this call (debug). Chips are streamed in
    batches of ``batch_size`` and flushed every ``shard_tiles`` tiles.
    """
    s = settings or load_settings()
    folder = embedding_dir(aoi_id, layer, size_m, encoder.name, s)
    with rasterio.open(raster_path) as src:
        b = src.bounds
        pixel_m = abs(src.transform.a)
    check_tile_scale(size_m, pixel_m)
    grid = make_grid(aoi_id, size_m, s.overlap, s, bounds=(b.left, b.bottom, b.right, b.top))
    done = cached_ids(folder)
    todo = grid[~grid["tile_id"].isin(done)]
    log.info(
        "embed %s/%s size=%d: %d tiles in footprint, %d cached, %d to do",
        aoi_id,
        layer,
        size_m,
        len(grid),
        len(done),
        len(todo),
    )
    new = 0
    ids: list[str] = []
    chips: list[np.ndarray] = []
    pending_ids: list[str] = []
    pending: list[np.ndarray] = []

    def encode_batch() -> None:
        nonlocal chips, ids
        if chips:
            pending.append(np.asarray(encoder.encode(chips), dtype="float32"))
            pending_ids.extend(ids)
            chips, ids = [], []

    def flush() -> None:
        nonlocal pending, pending_ids
        encode_batch()
        if pending_ids:
            _write_shard(folder, pending_ids, np.concatenate(pending))
        pending, pending_ids = [], []

    bar = None
    if progress:
        from tqdm import tqdm

        bar = tqdm(
            total=len(todo) if max_tiles is None else min(len(todo), max_tiles),
            desc=f"embed {layer}/{size_m}",
            unit="tile",
        )
    try:
        for tid, chip in tile_windows(todo, raster_path, band, min_valid_frac):
            if max_tiles is not None and new >= max_tiles:
                break
            ids.append(tid)
            chips.append(chip)
            new += 1
            if len(chips) >= batch_size:
                encode_batch()
                if bar:
                    bar.update(batch_size)
            if len(pending_ids) + len(ids) >= shard_tiles:
                flush()
        flush()
    finally:
        if bar:
            bar.close()
    return {
        "grid": len(grid),
        "cached_before": len(done),
        "encoded": new,
        "cached_total": len(done) + new,
    }


def run_embedding_scores(
    aoi_id: str,
    layer: str,
    size_m: int = 128,
    encoder: str = "dinov2-small_clahe",
    methods: Sequence[str] = METHODS,
    pca: int | None = DEFAULT_PCA,
    seed: int = 0,
    write: bool = True,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Fit the shared background models on cached embeddings; one score table per method.

    Detectors are ``embed_{method}``. ``pca`` (None/0 = off) reduces to at most that many
    components (seeded; capped at n-1 and d) before the robust-scaled background fit.
    Fit is per (aoi, layer, size) and never sees labels (D7).
    """
    s = settings or load_settings()
    bad = set(methods) - set(METHODS)
    if bad:
        raise ValueError(f"unknown method(s) {sorted(bad)}; choose from {METHODS}")
    ids, X = load_embeddings(embedding_dir(aoi_id, layer, size_m, encoder, s))
    if len(ids) < 3:
        raise ValueError(f"only {len(ids)} cached embeddings; run `embed` first (need >= 3)")
    X = X.astype("float64")
    if pca:
        from sklearn.decomposition import PCA

        X = PCA(n_components=min(pca, len(X) - 1, X.shape[1]), random_state=seed).fit_transform(X)
    frames = []
    for m in methods:
        frames.append(
            pd.DataFrame(
                {
                    "tile_id": ids,
                    "aoi_id": aoi_id,
                    "size_m": int(size_m),
                    "detector": f"embed_{m}",
                    "layer": layer,
                    "score": fit_score(X, m, seed=seed),
                }
            )
        )
    df = validate_scores(_prepared(pd.concat(frames, ignore_index=True)))
    log.info(
        "embed scores %s/%s size=%d: %d tiles x %d methods",
        aoi_id,
        layer,
        size_m,
        len(ids),
        len(methods),
    )
    if write:
        write_scores(df, s)
    return df
