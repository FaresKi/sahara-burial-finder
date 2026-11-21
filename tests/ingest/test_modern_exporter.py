import json
import subprocess
import sys

import numpy as np
import pytest
from typer.testing import CliRunner

from sbf.cli import app
from sbf.ingest.modern import (
    EarthEngineExporter,
    FakeExporter,
    GeeExporter,
    GeeNotConfiguredError,
    grid_from_bounds,
)
from sbf.ingest.modern.exporter import _read_tile_bytes, plan_tiles, write_tiles

GRID = grid_from_bounds((-0.0620, 26.3070, -0.0525, 26.3150), "EPSG:32631", 10.0)


def test_protocol_conformance():
    assert isinstance(FakeExporter(), GeeExporter)
    assert isinstance(EarthEngineExporter(project="x", ee_module=object()), GeeExporter)


def test_ee_import_is_lazy():
    code = "import sbf.ingest.modern, sys; assert 'ee' not in sys.modules"
    r = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_missing_gee_project_raises(monkeypatch):
    monkeypatch.delenv("GEE_PROJECT", raising=False)
    exp = EarthEngineExporter(ee_module=type("EE", (), {})())
    exp._initialised = False
    with pytest.raises(GeeNotConfiguredError, match="GEE_PROJECT"):
        exp.verify_datasets(GRID)


def test_plan_tiles_cover_exactly_and_respect_size():
    tiles = plan_tiles(2500, 1700, 6, max_bytes=20_000_000)
    assert sum(w * h for _, _, w, h in tiles) == 2500 * 1700
    assert all(w * h * 6 * 4 <= 20_000_000 for _, _, w, h in tiles)
    covered = np.zeros((1700, 2500), bool)
    for c, r, w, h in tiles:
        assert not covered[r : r + h, c : c + w].any()
        covered[r : r + h, c : c + w] = True
    assert covered.all()
    assert len(plan_tiles(10, 10, 1)) == 1


def test_write_tiles_roundtrip(tmp_path):
    import rasterio

    _, w, h, _ = GRID
    full = np.random.default_rng(0).random((2, h, w)).astype(np.float32)
    tiles = plan_tiles(w, h, 2, max_bytes=2 * 4 * 40 * 40)
    assert len(tiles) > 1
    out = tmp_path / "t.tif"
    write_tiles(
        out, GRID, ["a", "b"], tiles, lambda t: full[:, t[1] : t[1] + t[3], t[0] : t[0] + t[2]]
    )
    with rasterio.open(out) as d:
        assert d.descriptions == ("a", "b") and d.transform == GRID[0]
        assert np.array_equal(d.read(), full)
    with pytest.raises(ValueError):
        write_tiles(out, GRID, ["a", "b"], tiles[:1], lambda t: full[:, :3, :3])


def test_read_tile_bytes_tif_and_zip(tmp_path):
    import io
    import zipfile

    import rasterio

    arrs = []
    blobs = []
    for i in range(2):
        with rasterio.MemoryFile() as mf:
            with mf.open(driver="GTiff", width=4, height=3, count=1, dtype="float32") as d:
                a = np.full((3, 4), float(i), dtype=np.float32)
                d.write(a, 1)
            blobs.append(mf.read())
        arrs.append(a)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.b0.tif", blobs[0])
        z.writestr("x.b1.tif", blobs[1])
    assert _read_tile_bytes(buf.getvalue(), 2).shape == (2, 3, 4)
    assert _read_tile_bytes(blobs[0], 1).shape == (1, 3, 4)
    with pytest.raises(ValueError):
        _read_tile_bytes(blobs[0], 2)


class _Chain:
    """Stub of an ee.ImageCollection chain returning a fixed count/bands."""

    def __init__(self, n, bands):
        self.n, self.bands = n, bands

    def filterBounds(self, *_):
        return self

    def filterDate(self, *_):
        return self

    def filter(self, *_):
        return self

    def size(self):
        return self

    def first(self):
        return self

    def bandNames(self):
        return self

    def getInfo(self):
        return self.bands if isinstance(self.bands, list) and self.n == "bands" else self._info()

    def _info(self):
        raise AssertionError


class StubEE:
    """Minimal ee stand-in: only the verification path."""

    class Filter:
        @staticmethod
        def Or(*a):
            return None

        @staticmethod
        def calendarRange(*a):
            return None

        @staticmethod
        def eq(*a):
            return None

        @staticmethod
        def listContains(*a):
            return None

    class Geometry:
        @staticmethod
        def Rectangle(*a, **k):
            return "rect"

    def __init__(self, assets):
        self.assets = assets  # id -> (count, bands)
        outer = self

        class _Data:
            @staticmethod
            def getAsset(i):
                if i not in outer.assets:
                    raise RuntimeError(f"Asset '{i}' not found")

        class _Col:
            def __init__(self, i):
                self.i = i

            def filterBounds(self, *_):
                return self

            filterDate = filter = filterBounds

            def size(self):
                return _Val(outer.assets[self.i][0])

            def first(self):
                return _First(outer.assets[self.i][1])

        class _Val:
            def __init__(self, v):
                self.v = v

            def getInfo(self):
                return self.v

        class _First:
            def __init__(self, b):
                self.b = b

            def bandNames(self):
                return _Val(self.b)

        self.data = _Data()
        self.ImageCollection = _Col


def test_verify_records_failures_and_never_fills(tmp_path):
    ee = StubEE(
        {
            "COPERNICUS/S2_SR_HARMONIZED": (
                12,
                ["B2", "B3", "B4", "B8", "B11", "B12", "SCL", "QA60"],
            ),
            "COPERNICUS/S1_GRD": (0, ["VV", "VH"]),  # empty over AOI
            "JAXA/ALOS/PALSAR/YEARLY/SAR": (5, ["HH", "HV"]),  # only the 2nd candidate exists
            # DEM: asset missing entirely
        }
    )
    exp = EarthEngineExporter(project="p", ee_module=ee)
    checks = {c.product: c for c in exp.verify_datasets(GRID)}
    assert checks["s2_median"].ok and checks["s2_djf"].ok
    assert not checks["s1"].ok and "no images" in checks["s1"].detail
    assert checks["palsar_latest"].ok and checks["palsar_latest"].dataset_id.endswith("/SAR")
    assert not checks["dem"].ok and "not found" in checks["dem"].detail
    # missing band is a failure too
    ee2 = StubEE({"COPERNICUS/DEM/GLO30": (3, ["WRONG"])})
    c = EarthEngineExporter(project="p", ee_module=ee2).verify_datasets(GRID, ["dem"])[0]
    assert not c.ok and "missing bands" in c.detail


def test_export_aoi_failed_verification_writes_nothing_but_report(tmp_path):
    exp = EarthEngineExporter(project="p", ee_module=StubEE({}))
    rep = exp.export_aoi("REG", tmp_path / "raw", grid=GRID, products=["dem", "s1"])
    assert [r.status for r in rep.results] == ["failed", "failed"]
    assert not list((tmp_path / "raw").glob("*.tif"))
    saved = json.loads((tmp_path / "raw" / "export_report.json").read_text())
    assert saved["results"][0]["error"]


def test_fake_exporter_report_and_files(tmp_path):
    rep = FakeExporter(fail={"s1"}).export_aoi("REG", tmp_path, grid=GRID, products=["s1", "dem"])
    assert [(r.product, r.status) for r in rep.results] == [("s1", "failed"), ("dem", "ok")]
    assert (tmp_path / "dem.tif").exists() and (tmp_path / "dem.tif.prov.json").exists()
    assert not (tmp_path / "s1.tif").exists()


# ------------------------------------------------------------------ CLI


def test_cli_export_gated_by_gee_project(tmp_project, monkeypatch):
    monkeypatch.delenv("GEE_PROJECT", raising=False)
    r = CliRunner().invoke(app, ["ingest", "modern", "export", "--aoi", "REG"])
    assert r.exit_code == 2
    assert "GEE_PROJECT" in r.output


def test_cli_unknown_aoi(tmp_project):
    r = CliRunner().invoke(app, ["ingest", "modern", "build-stack", "--aoi", "XXX"])
    assert r.exit_code == 2


def test_cli_build_stack_without_raw_reports_missing(tmp_project, monkeypatch):
    # tiny grid to avoid building huge arrays: no raw data => nothing is read anyway
    r = CliRunner().invoke(app, ["ingest", "modern", "build-stack", "--aoi", "INE"])
    assert r.exit_code == 0, r.output
    assert "0 layers written, 16 missing" in r.output
