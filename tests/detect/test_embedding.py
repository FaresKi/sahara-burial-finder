"""Embedding detector with an injected fake encoder (no model weights, no network)."""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import box
from typer.testing import CliRunner

from sbf.detect.embedding import (
    cached_ids,
    compute_embeddings,
    embedding_dir,
    load_embeddings,
    run_embedding_scores,
)
from sbf.detect.embedding_cli import embed_app
from sbf.detect.harness import evaluate
from sbf.detect.scores import read_scores
from sbf.features.embeddings import TileEncoder, chip_to_rgb
from sbf.features.tiling import grid_origin

RES = 2.0
SIZE_PX = 512  # 1024 m
TRENCH_BLOCKS = [(256, 256), (640, 384), (384, 704)]


class FakeEncoder:
    """Deterministic: 4x4 pooled gradient-energy stats, randomly projected to 16 dims."""

    name = "fake"
    dim = 16

    def __init__(self):
        self.calls = 0
        self.proj = np.random.default_rng(0).normal(size=(32, self.dim)).astype("float32")

    def encode(self, chips):
        self.calls += len(chips)
        out = []
        for c in chips:
            c = np.nan_to_num(c)
            g = np.abs(np.diff(c, axis=1))[:, :, None][:, : c.shape[1] - 1, 0]
            g = g[: (g.shape[0] // 4) * 4, : (g.shape[1] // 4) * 4]
            cells = g.reshape(4, g.shape[0] // 4, 4, g.shape[1] // 4)
            f = np.concatenate([cells.mean(axis=(1, 3)).ravel(), cells.std(axis=(1, 3)).ravel()])
            out.append(f @ self.proj)
        return np.asarray(out, dtype="float32")


@pytest.fixture
def desert(tmp_project):
    ox, oy = grid_origin("REG")
    x0, y0 = ox + 4096, oy + 4096
    rng = np.random.default_rng(42)
    dunes = ndimage.gaussian_filter(rng.normal(size=(SIZE_PX, SIZE_PX)), 40)
    img = 0.5 + 0.25 * dunes / dunes.std() + rng.normal(0, 0.02, (SIZE_PX, SIZE_PX))
    for bx, by in TRENCH_BLOCKS:
        c0, r0 = int(bx / RES), SIZE_PX - int((by + 64) / RES)
        for k in range(0, 32, 8):
            img[r0 : r0 + 32, c0 + k : c0 + k + 4] -= 0.35
    path = tmp_project.root / "synthetic.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SIZE_PX,
        width=SIZE_PX,
        count=1,
        dtype="float32",
        crs="EPSG:32631",
        nodata=-9999.0,
        transform=from_origin(x0, y0 + 1024, RES, RES),
    ) as dst:
        dst.write(img.astype("float32"), 1)
    ref = gpd.GeoDataFrame(
        {
            "feature_id": [f"T{i}" for i in range(3)],
            "aoi_id": "REG",
            "precision": "iaea",
            "expected_area_km2": [None] * 3,
        },
        geometry=[
            box(x0 + bx + 8, y0 + by + 8, x0 + bx + 56, y0 + by + 56) for bx, by in TRENCH_BLOCKS
        ],
        crs="EPSG:32631",
    )
    return path, ref.to_crs("EPSG:4326"), (x0, y0)


def test_protocol_and_chip_to_rgb():
    assert isinstance(FakeEncoder(), TileEncoder)
    chip = np.random.default_rng(0).random((20, 20)).astype("float32")
    chip[:3] = np.nan
    x = chip_to_rgb(chip, "clahe")
    assert x.shape == (3, 224, 224) and np.isfinite(x).all() and x.dtype == np.float32
    assert chip_to_rgb(np.stack([chip] * 3), "none").shape == (3, 224, 224)


def test_cache_resume_and_max_tiles(tmp_project, desert):
    path, _, _ = desert
    enc = FakeEncoder()
    r1 = compute_embeddings(
        "REG",
        "syn",
        path,
        enc,
        128,
        max_tiles=40,
        shard_tiles=16,
        batch_size=7,
        progress=False,
        settings=tmp_project,
    )
    assert r1["encoded"] == 40 and enc.calls == 40
    folder = embedding_dir("REG", "syn", 128, "fake", tmp_project)
    assert len(cached_ids(folder)) == 40 and len(list(folder.glob("part-*.parquet"))) == 3
    r2 = compute_embeddings(
        "REG", "syn", path, enc, 128, shard_tiles=64, progress=False, settings=tmp_project
    )
    assert r2["cached_before"] == 40 and r2["encoded"] == r1["grid"] - 40
    assert enc.calls == r1["grid"]  # nothing re-encoded
    r3 = compute_embeddings("REG", "syn", path, enc, 128, progress=False, settings=tmp_project)
    assert r3["encoded"] == 0 and enc.calls == r1["grid"]
    ids, X = load_embeddings(folder)
    assert len(ids) == len(set(ids)) == r1["grid"] and X.shape == (len(ids), 16)
    assert folder.parts[-3:-1] == ("embeddings", "REG")


def test_footprint_restriction_and_nodata(tmp_project, desert):
    path, _, (x0, y0) = desert
    compute_embeddings("REG", "syn", path, FakeEncoder(), 128, progress=False, settings=tmp_project)
    ids, _ = load_embeddings(embedding_dir("REG", "syn", 128, "fake", tmp_project))
    from sbf.features.tiling import tile_geometries

    g = tile_geometries(ids, settings=tmp_project)
    b = g.total_bounds
    assert b[0] >= x0 - 1e-6 and b[1] >= y0 - 1e-6 and b[2] <= x0 + 1024 + 1e-6
    assert b[3] <= y0 + 1024 + 1e-6
    assert len(ids) == 225  # 15x15 at 64 m step: footprint only, not the AOI grid


def test_small_tiles_warn(tmp_project, desert):
    path, _, _ = desert
    with pytest.warns(UserWarning, match="upsampling"):
        compute_embeddings(
            "REG", "syn", path, FakeEncoder(), 16, max_tiles=3, progress=False, settings=tmp_project
        )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*upsampling.*")
        compute_embeddings(
            "REG",
            "syn",
            path,
            FakeEncoder(),
            128,
            max_tiles=3,
            progress=False,
            settings=tmp_project,
        )


def test_end_to_end_scores_and_harness(tmp_project, desert):
    path, ref, _ = desert
    compute_embeddings("REG", "syn", path, FakeEncoder(), 128, progress=False, settings=tmp_project)
    df = run_embedding_scores("REG", "syn", 128, "fake", pca=8, settings=tmp_project)
    assert set(df["detector"]) == {"embed_iforest", "embed_knn", "embed_maha"}
    back = read_scores("REG", layer="syn", settings=tmp_project)
    assert len(back) == len(df) and (back["size_m"] == 128).all()
    again = run_embedding_scores(
        "REG", "syn", 128, "fake", pca=8, write=False, settings=tmp_project
    )
    assert np.allclose(df["score"], again["score"])  # seeded
    m = evaluate(back, ref, gpd.GeoDataFrame(), tmp_project, ks=(10, 50))
    assert len(m) == 3 and (m["n_features"] == 3).all()
    assert (m["ap"] > m["random_ap"]).all()
    assert (m["recall_at_50"] >= m["random_recall_at_50"]).all()


def test_scores_need_cache(tmp_project):
    with pytest.raises(ValueError, match="embed"):
        run_embedding_scores("REG", "nothing", 128, "fake", settings=tmp_project)
    with pytest.raises(ValueError, match="unknown method"):
        run_embedding_scores("REG", "nothing", 128, "fake", methods=["bogus"], settings=tmp_project)


def test_cli_score(tmp_project, desert):
    path, _, _ = desert
    compute_embeddings("REG", "syn", path, FakeEncoder(), 128, progress=False, settings=tmp_project)
    r = CliRunner().invoke(
        embed_app,
        [
            "score",
            "--aoi",
            "REG",
            "--layer",
            "syn",
            "--encoder",
            "fake",
            "--methods",
            "knn,maha",
            "--pca",
            "8",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "embed_knn" in r.output and "POLYGON" not in r.output
    assert not read_scores("REG", "embed_knn", "syn", tmp_project).empty
    r = CliRunner().invoke(embed_app, ["score", "--aoi", "REG", "--layer", "missing"])
    assert r.exit_code != 0


@pytest.mark.heavy
def test_dinov2_real_weights_if_cached():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    hub = Path.home() / ".cache/huggingface/hub/models--facebook--dinov2-small"
    if not hub.exists():
        pytest.skip("dinov2-small weights not cached locally")
    from sbf.features.embeddings import DinoV2Encoder

    enc = DinoV2Encoder(device="cpu")
    out = enc.encode([np.random.default_rng(0).random((40, 40)).astype("float32")] * 2)
    assert out.shape == (2, 768) == (2, enc.dim) and np.isfinite(out).all()
