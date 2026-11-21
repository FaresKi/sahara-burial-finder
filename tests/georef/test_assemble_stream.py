"""Ticket 08: multi-part assembly and streaming warp."""

import json

import numpy as np
import rasterio
from georef_synth import make_scene

from sbf.georef import fit as fitmod
from sbf.georef.assemble import assemble_frame, detect_parts
from sbf.georef.frame_io import ArraySource, RasterSource
from sbf.georef.warp import warp_frame, warp_to_file


def _write(path, arr):
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=arr.dtype
    ) as d:
        d.write(arr, 1)


def _strip_image():
    scene = make_scene(seed=5)[:620, :1500]
    return np.clip(scene * 255, 0, 255).astype("uint8")


def test_detect_parts_pattern():
    from pathlib import Path

    fs = [Path(n) for n in ("X_b.tif", "X_a.tif", "X_c.tif", "Y.tif", "X_browse.tif")]
    assert [p.name for p in detect_parts(fs)] == ["X_a.tif", "X_b.tif", "X_c.tif"]
    assert detect_parts([Path("Y.tif")]) == []


def test_single_part_passthrough(tmp_path):
    _write(tmp_path / "ENT.tif", _strip_image())
    asm = assemble_frame(tmp_path, tmp_path / "out")
    assert asm.passthrough and asm.path == tmp_path / "ENT.tif" and asm.shape == (620, 1500)
    assert not (tmp_path / "out").exists()


def test_three_overlapping_parts_assemble(tmp_path):
    img = _strip_image()
    # parts overlap by 150 px; part b has a different vertical offset (film wander)
    _write(tmp_path / "E_a.tif", img[0:600, 0:600])
    _write(tmp_path / "E_b.tif", img[8:608, 450:1050])
    _write(tmp_path / "E_c.tif", img[0:600, 900:1500])
    out = tmp_path / "proc"
    asm = assemble_frame(tmp_path, out)
    assert not asm.passthrough and asm.json_path.exists()
    meta = json.loads(asm.json_path.read_text())
    assert [p["file"] for p in meta["parts"]] == ["E_a.tif", "E_b.tif", "E_c.tif"]
    assert all(p["inliers"] >= 12 for p in meta["parts"][1:])
    assert [p["side"] for p in meta["parts"][1:]] == ["right", "right"]
    assert asm.shape == (608, 1500)
    assert asm.parts[1].offset_xy == (450.0, 8.0) and asm.parts[2].offset_xy == (900.0, 0.0)
    with rasterio.open(asm.path) as s:
        got = s.read(1)
        assert s.crs is None
    assert np.abs(got[10:598].astype(int) - img[10:598].astype(int)).mean() < 1.0
    # cached second call
    assert assemble_frame(tmp_path, out).parts[1].offset_xy == asm.parts[1].offset_xy


def test_stream_warp_matches_in_memory(tmp_path):
    img = _strip_image()
    g = np.array([(x, y) for x in range(0, 1501, 100) for y in range(0, 621, 100)], float)
    xy = np.column_stack([500000 + 3.0 * g[:, 0] + 0.002 * g[:, 1] ** 1.2, 2.9e6 - 3.0 * g[:, 1]])
    model = fitmod.fit_transform(g, xy, "poly2")
    big = warp_frame(img, model, resolution_m=3.0)
    tiny = warp_frame(img, model, resolution_m=3.0, block=64, step=8)
    assert big.array.shape == tiny.array.shape
    assert np.abs(big.array.astype(int) - tiny.array.astype(int)).max() <= 1
    _write(tmp_path / "f.tif", img)
    wf = warp_to_file(
        RasterSource(tmp_path / "f.tif"), model, tmp_path / "w.tif", resolution_m=3.0, block=128
    )
    with rasterio.open(wf.path) as s:
        disk = s.read(1)
    assert np.abs(disk.astype(int) - big.array.astype(int)).max() <= 1
    # ArraySource and lazily-read source agree
    a = warp_frame(ArraySource(img), model, resolution_m=3.0, block=100)
    assert np.abs(a.array.astype(int) - big.array.astype(int)).max() <= 1
