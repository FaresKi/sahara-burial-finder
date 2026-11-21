import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from typer.testing import CliRunner

from sbf.cli import app
from sbf.core.aoi import load_aois, load_reference_features
from sbf.core.crs import aoi_crs, buffer_point_m, lonlat_to_xy, xy_to_lonlat
from sbf.core.provenance import append_step, read_prov, script_commit, write_prov
from sbf.core.raster import write_cog
from sbf.core.restricted import RestrictedPathError, encrypt_if_key, restricted_path


def test_settings(tmp_project):
    s = tmp_project
    assert s.root.name == tmp_project.root.name
    assert s.outputs_restricted == s.root / "outputs" / "restricted"
    assert s.tile_sizes_m == [64, 128] and s.overlap == 0.5 and s.modern_resolution_m == 10
    assert s.crs("REG") == s.crs("INE") == "EPSG:32631"
    with pytest.raises(KeyError):
        s.crs("XXX")


def test_crs_roundtrip(tmp_project):
    x, y = lonlat_to_xy(-0.0572, 26.3117, "EPSG:32631")
    assert 150_000 < x < 800_000 and 2_800_000 < y < 3_000_000
    lon, lat = xy_to_lonlat(x, y, "EPSG:32631")
    assert lon == pytest.approx(-0.0572, abs=1e-8) and lat == pytest.approx(26.3117, abs=1e-8)
    assert aoi_crs("INE", tmp_project) == "EPSG:32631"
    circle = buffer_point_m(5.05, 24.03, 100, "EPSG:32631")
    assert circle.area == pytest.approx(np.pi * 100**2, rel=0.01)


def test_aois(tmp_project):
    g = load_aois(tmp_project)
    assert set(g.aoi_id) == {"REG", "INE"} and g.crs.to_epsg() == 4326
    reg = g[g.aoi_id == "REG"].geometry.iloc[0]
    assert reg.bounds == (-0.25, 26.05, 0.25, 26.75)


def test_reference_features(tmp_project):
    g = load_reference_features(tmp_project).set_index("feature_id")
    assert set(g.index) == {"GZ1", "GZ2", "GZ3", "GZ4", "TA1", "PO1", "BASE"}
    assert (g.loc["GZ1"].geometry.x, g.loc["GZ1"].geometry.y) == (-0.0572, 26.3117)
    assert g.loc["BASE"].geometry is None and g.loc["BASE"].precision == "to_locate"
    assert g.loc["TA1"].precision == g.loc["PO1"].precision == "approximate"
    assert g.loc["GZ4"].expected_area_km2 == 0.1
    assert g.loc["GZ3"].aoi_id == "REG" and g.loc["PO1"].aoi_id == "INE"


def test_provenance_roundtrip(tmp_path):
    r = tmp_path / "a.tif"
    r.write_bytes(b"x")
    write_prov(r, source="test", source_id="ID1", acquisition_date=None, steps=["s1"], inputs=["i"])
    rec = read_prov(r)
    assert rec["source_id"] == "ID1" and rec["acquisition_date"] is None
    assert rec["inputs"] == ["i"] and rec["processing_steps"] == ["s1"]
    assert {"download_date", "script_commit", "sbf_version"} <= rec.keys()
    append_step(r, "s2")
    assert read_prov(r)["processing_steps"] == ["s1", "s2"]


def test_script_commit_no_git(tmp_path):
    assert script_commit(tmp_path) == "no-git"


def test_write_cog(tmp_path):
    arr = np.arange(64 * 64, dtype="float32").reshape(64, 64)
    p = write_cog(
        tmp_path / "x.tif",
        arr,
        {"crs": "EPSG:32631", "transform": from_origin(500000, 2900000, 10, 10)},
        prov={
            "source": "t",
            "source_id": "t1",
            "acquisition_date": "2024-01-01",
            "steps": ["synthetic"],
        },
    )
    with rasterio.open(p) as src:
        assert src.count == 1 and src.crs.to_epsg() == 32631
        assert np.array_equal(src.read(1), arr)
    assert json.loads((tmp_path / "x.tif.prov.json").read_text())["source_id"] == "t1"


def test_restricted_path(tmp_project):
    p = restricted_path("candidates", "a.gpkg", settings=tmp_project)
    assert p.is_relative_to(tmp_project.outputs_restricted.resolve())
    assert p.parent.is_dir()
    for bad in (("..", "public", "x"), ("/etc/passwd",), ("a", "..", "..", "x")):
        with pytest.raises(RestrictedPathError):
            restricted_path(*bad, settings=tmp_project)
    link = tmp_project.outputs_restricted / "link"
    link.symlink_to(tmp_project.outputs_public)
    with pytest.raises(RestrictedPathError):
        restricted_path("link", "x.txt", settings=tmp_project)


def test_encrypt_if_key_warns_without_key(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("RESTRICTED_KEY", raising=False)
    f = tmp_path / "f.txt"
    f.write_text("x")
    assert encrypt_if_key(f) == f
    assert "RESTRICTED_KEY" in caplog.text


def test_cli_mounts_all(tmp_project):  # isolated: the real project now holds live data
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0
    for name in (
        "catalog",
        "ingest",
        "georef",
        "features",
        "detect",
        "docs",
        "fuse",
        "report",
        "core",
        "run-all",
    ):
        assert name in r.output
    assert CliRunner().invoke(app, ["run-all"]).exit_code == 0
    assert CliRunner().invoke(app, ["catalog", "status"]).exit_code == 0


def test_load_dotenv_does_not_override(tmp_path, monkeypatch):
    from sbf.core.config import load_dotenv

    env = tmp_path / ".env"
    env.write_text('# c\nSBF_T_A=one\nexport SBF_T_B="two"\nSBF_T_C=\nSBF_T_D=new\n')
    monkeypatch.delenv("SBF_T_A", raising=False)
    monkeypatch.delenv("SBF_T_B", raising=False)
    monkeypatch.setenv("SBF_T_D", "shell")
    assert sorted(load_dotenv(env)) == ["SBF_T_A", "SBF_T_B"]
    import os

    assert os.environ["SBF_T_A"] == "one" and os.environ["SBF_T_B"] == "two"
    assert os.environ["SBF_T_D"] == "shell" and "SBF_T_C" not in os.environ
    for k in ("SBF_T_A", "SBF_T_B"):
        monkeypatch.delenv(k)
