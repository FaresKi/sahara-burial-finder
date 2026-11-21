"""Ticket 08: QGIS fallback round trip, strips through run_frame, batch + registry."""

import json

import numpy as np
import pandas as pd
import pytest
import rasterio
from georef_synth import make_panoramic_frame, make_reference, make_scene
from shapely.geometry import Polygon

from sbf.georef import qgis as q
from sbf.georef import registry as reg
from sbf.georef.models import GeorefReport
from sbf.georef.pipeline import run_frame


@pytest.fixture(scope="module")
def pano():
    scene = make_scene()
    return make_reference(scene), make_panoramic_frame(scene, w=2400, h=800, res_y=2.2)


def _project(s, ref, fr, parts=1, eid="PAN1", camera="KH-4B", date="1965-03-01"):
    sd = s.data_processed / "modern" / "REG"
    sd.mkdir(parents=True, exist_ok=True)
    arr = np.stack([ref.array] * 4)
    prof = {
        "driver": "GTiff",
        "count": 4,
        "dtype": "float32",
        "crs": "EPSG:32631",
        "nodata": np.nan,
        "transform": ref.transform,
        "height": arr.shape[1],
        "width": arr.shape[2],
    }
    with rasterio.open(sd / "s2_median.tif", "w", **prof) as dst:
        dst.write(arr)
    names = ["B2", "B3", "B4", "B8"]
    layers = {
        "s2_median": {
            "file": "s2_median.tif",
            "bands": [{"index": i + 1, "name": n} for i, n in enumerate(names)],
        }
    }
    (sd / "stack.json").write_text(json.dumps({"layers": layers}))
    ed = s.data_raw / "declass" / eid
    ed.mkdir(parents=True, exist_ok=True)
    h, w = fr.image.shape
    cuts = np.linspace(0, w, parts + 1).astype(int)
    for i in range(parts):
        a, b = max(cuts[i] - 120, 0), min(cuts[i + 1] + 120, w)
        name = f"{eid}.tif" if parts == 1 else f"{eid}_{'abcd'[i]}.tif"
        p = {"driver": "GTiff", "count": 1, "dtype": "uint8", "height": h, "width": b - a}
        with rasterio.open(ed / name, "w", **p) as dst:
            dst.write(fr.image[:, a:b], 1)
    return eid, Polygon(fr.corners_lonlat), camera, date


def _coverage(s, items):
    rows = [
        {
            "entity_id": eid,
            "dataset": "CORONA2",
            "mission": "KH-4B",
            "camera": camera,
            "acquisition_date": date,
            "cloud_cover": 0,
            "availability": "scanned",
            "browse_url": "",
            "footprint_wkt": poly.wkt,
            "intersecting_features": "GZ1",
            "resolution_class": "<=3m",
            "aoi_id": "REG",
            "raw_file": "",
        }
        for eid, poly, camera, date in items
    ]
    s.outputs_public.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(s.outputs_public / "coverage.csv", index=False)


# ---------------------------------------------------------------- QGIS


def test_points_roundtrip_and_sign(tmp_path):
    g = pd.DataFrame(
        {
            "px": [10.5, 300.25, 77.0],
            "py": [20.0, 410.75, 5.5],
            "x": [500010.0, 500400.5, 500100.25],
            "y": [2899900.0, 2899500.5, 2899950.75],
            "residual_m": [1.0, 2.0, 0.5],
        }
    )
    p = q.write_points(g, tmp_path / "a.points")
    text = p.read_text().splitlines()
    assert text[0].startswith("#CRS:") and text[1] == q.HEADER
    assert text[2].split(",")[2:4] == ["10.5000", "-20.0000"]  # QGIS: sourceY = -row
    back = q.read_points(p).gcps
    np.testing.assert_allclose(back[["px", "py", "x", "y"]], g[["px", "py", "x", "y"]], atol=1e-3)


def test_points_read_disabled_wgs84_and_positive_rows(tmp_path):
    p = tmp_path / "m.points"
    p.write_text(
        "#CRS: EPSG:4326\n" + q.HEADER + "\n3.0,26.0,100,-200,1,0,0,0\n3.1,26.1,110,-210,0,0,0,0\n"
    )
    pf = q.read_points(p)
    assert len(pf.gcps) == 1 and pf.gcps.iloc[0]["py"] == 200
    assert 400000 < pf.gcps.iloc[0]["x"] < 600000 and any("reprojected" in n for n in pf.notes)
    p.write_text(q.HEADER + "\n500000,2900000,1,5,1,0,0,0\n500010,2900010,2,6,1,0,0,0\n")
    assert q.read_points(p).gcps["py"].tolist() == [5, 6]
    p.write_text(q.HEADER + "\n500000,2900000,1,5,1,0,0,0\n500010,2900010,2,-6,1,0,0,0\n")
    with pytest.raises(ValueError, match="mixed"):
        q.read_points(p)
    p.write_text("garbage\n")
    with pytest.raises(ValueError, match="not a QGIS"):
        q.read_points(p)


def test_qgis_export_import_roundtrip(tmp_project, pano):
    ref, fr = pano
    items = [_project(tmp_project, ref, fr)]
    _coverage(tmp_project, items)
    rep = run_frame("PAN1", settings=tmp_project, strips=True, n_strips=4, make_overlay=False)
    assert rep.method == "strips" and rep.included_for_change_detection
    d = tmp_project.data_processed / "historical" / "PAN1"
    r = q.export_qgis("PAN1", settings=tmp_project)
    assert r["n_seed"] == rep.n_gcps and r["readme"].read_text().count("PAN1") >= 2
    assert "Georeferencer" in r["readme"].read_text()
    # the "human" keeps a subset of the seeds (as if re-picked in QGIS) and saves the file
    pts = q.read_points(r["points"]).gcps
    assert len(pts) == rep.n_gcps
    keep = pts.iloc[::2]
    q.write_points(keep, d / "qgis" / "PAN1_manual.points")
    rep2 = q.import_qgis(
        "PAN1",
        d / "qgis" / "PAN1_manual.points",
        settings=tmp_project,
        make_overlay=False,
        gz=None,
        method="poly3",
    )
    assert rep2.gcp_source == "manual_gcps" and rep2.method == "poly3"
    assert rep2.n_gcps == len(keep)
    assert rep2.n_checkpoints == round(0.2 * len(keep))  # 20% still held out
    on_disk = GeorefReport.model_validate_json((d / "georef_report.json").read_text())
    assert on_disk.gcp_source == "manual_gcps" and on_disk.georef_file
    gc = pd.read_csv(d / "gcps.csv")
    assert set(gc.role) == {"train", "check"} and len(gc) == len(keep)
    np.testing.assert_allclose(
        np.sort(gc["px"].to_numpy()), np.sort(keep["px"].to_numpy()), atol=1e-2
    )
    with rasterio.open(on_disk.georef_file) as s:
        assert s.crs.to_string() == "EPSG:32631"


# ---------------------------------------------------------------- run_frame strips + multipart


def test_run_frame_panoramic_multipart(tmp_project, pano):
    ref, fr = pano
    items = [_project(tmp_project, ref, fr, parts=3)]
    _coverage(tmp_project, items)
    rep = run_frame("PAN1", settings=tmp_project, n_strips=4, make_overlay=False)
    d = tmp_project.data_processed / "historical" / "PAN1"
    assert (d / "assembly.json").exists() and rep.assembly_file
    assert rep.method == "strips" and len(rep.strips) == 4 and rep.checkpoint_rmse_m < 12
    assert rep.georef_file and (d / "PAN1_georef.tif").exists()
    assert rep.cv_rmse_m is not None and rep.gcp_block_occupancy is not None
    assert rep.support_file == str(d / "PAN1_support.tif")
    with rasterio.open(d / "PAN1_georef.tif") as g, rasterio.open(rep.support_file) as m:
        assert (g.transform, g.shape, g.crs) == (m.transform, m.shape, m.crs)
        assert 0 < m.read(1).mean() < 1
    arr, tr = reg.load_support("PAN1", tmp_project)
    assert arr.dtype == np.uint8 and tr == g.transform
    assert not list(d.glob(".*tiled.tif"))


# ---------------------------------------------------------------- registry / batch


def test_era_boundaries():
    assert reg.era_of("1966-12-31") == "pre_1967"
    assert reg.era_of("1967-06-01") == "dismantling_1967"
    assert reg.era_of("1968-01-01") == "post_1967"
    assert reg.era_of(None) == "undated" and reg.era_of("") == "undated"


def _fake_runner(table):
    def run(eid, settings=None, aoi=None, **kw):
        if eid == "BOOM":
            raise RuntimeError("scan unreadable")
        rmse, gf = table[eid]
        return GeorefReport(
            entity_id=eid,
            aoi_id=aoi,
            method="poly2" if rmse is not None else "none",
            n_gcps=25,
            checkpoint_rmse_m=rmse,
            cv_rmse_m=rmse,
            gcp_block_occupancy=0.6,
            resolution_m=2.0,
            georef_file=gf,
            warnings=["rmse_gt_10m"] if rmse and rmse > 10 else [],
        )

    return run


def test_batch_registry(tmp_project, pano):
    ref, fr = pano
    s = tmp_project
    ids = {"A": ("1965-01-01", 4.0), "B": ("1967-05-05", 15.0), "C": ("1970-01-01", 25.0)}
    items = []
    for eid, (date, _) in ids.items():
        e = _project(s, ref, fr, eid=eid, date=date)
        items.append(e)
    items.append(_project(s, ref, fr, eid="BOOM"))
    items.append(_project(s, ref, fr, eid="NOSOLVE"))
    _coverage(s, items)
    (s.data_raw / "declass" / "ORPHAN").mkdir()
    pd.DataFrame().to_csv(s.data_raw / "declass" / "ORPHAN" / "x.txt")
    table = {k: (v[1], f"/f/{k}.tif") for k, v in ids.items()}
    table["NOSOLVE"] = (None, None)
    df = reg.run_batch(s, runner=_fake_runner(table))
    assert list(df.columns) == reg.REGISTRY_COLUMNS
    by = df.set_index("entity_id")
    assert bool(by.loc["A", "included_for_change_detection"])
    assert (
        bool(by.loc["B", "included_for_change_detection"])
        and "rmse_gt_10m" in by.loc["B", "warnings"]
    )
    assert not by.loc["C", "included_for_change_detection"]  # > 20 m excluded
    assert by.loc["BOOM", "status"] == "failed" and "scan unreadable" in by.loc["BOOM", "error"]
    assert by.loc["NOSOLVE", "status"] == "unsolved"
    assert by.loc["A", "era"] == "pre_1967" and by.loc["B", "era"] == "dismantling_1967"
    assert by.loc["C", "era"] == "post_1967" and by.loc["A", "aoi"] == "REG"
    assert (s.data_processed / "historical" / "BOOM" / "georef_error.txt").exists()
    inc = reg.load_registry(settings=s)
    assert sorted(inc.entity_id) == ["A", "B"]
    assert str(inc["acquisition_date"].dtype).startswith("datetime64")
    assert len(reg.load_registry(included_only=False, settings=s)) == 5
    md = (s.outputs_public / "georef_summary.md").read_text()
    assert (
        "Excluded (CV RMSE > 20 m, low GCP coverage or no CV): 1" in md and "scan unreadable" in md
    )


def test_batch_only_missing_and_aoi(tmp_project, pano):
    ref, fr = pano
    s = tmp_project
    items = [_project(s, ref, fr, eid="A"), _project(s, ref, fr, eid="B")]
    _coverage(s, items)
    calls = []

    def runner(eid, settings=None, aoi=None, **kw):
        calls.append(eid)
        r = GeorefReport(
            entity_id=eid,
            method="poly1",
            n_gcps=9,
            checkpoint_rmse_m=3.0,
            cv_rmse_m=3.0,
            gcp_block_occupancy=0.6,
            georef_file=f"/f/{eid}.tif",
        )
        d = s.data_processed / "historical" / eid
        d.mkdir(parents=True, exist_ok=True)
        (d / "georef_report.json").write_text(r.model_dump_json())
        return r

    reg.run_batch(s, runner=runner)
    assert calls == ["A", "B"]
    df = reg.run_batch(s, runner=runner, only_missing=True)
    assert calls == ["A", "B"] and len(df) == 2 and df["included_for_change_detection"].all()
    assert len(reg.run_batch(s, runner=runner, aoi="INE")) == 0


def test_cli_batch_and_qgis(tmp_project, pano):
    from typer.testing import CliRunner

    from sbf.cli import app

    ref, fr = pano
    _coverage(tmp_project, [_project(tmp_project, ref, fr)])
    r = CliRunner().invoke(app, ["georef", "batch", "--aoi", "REG"])
    assert r.exit_code == 0, r.output
    assert "1 frames" in r.output
    assert len(reg.load_registry(included_only=False, settings=tmp_project)) == 1
    ex = CliRunner().invoke(app, ["georef", "qgis-export", "--entity-id", "PAN1"])
    assert ex.exit_code == 0, ex.output
    pts = tmp_project.data_processed / "historical" / "PAN1" / "qgis" / "PAN1_seed.points"
    assert pts.exists()
    bad = CliRunner().invoke(
        app,
        ["georef", "qgis-import", "--entity-id", "PAN1", "--points", str(pts.parent / "no.points")],
    )
    assert bad.exit_code == 1


def test_registry_new_columns_and_gate(tmp_project, pano):
    ref, fr = pano
    s = tmp_project
    _coverage(s, [_project(s, ref, fr, eid=e) for e in ("OK", "SPARSE", "BADCV")])
    table = {  # cv, occupancy
        "OK": (9.0, 0.7),
        "SPARSE": (4.0, 0.2),  # accurate where it has GCPs, but covers too little of the frame
        "BADCV": (30.0, 0.9),
    }

    def runner(eid, settings=None, aoi=None, **kw):
        cv, occ = table[eid]
        return GeorefReport(
            entity_id=eid,
            method="poly2",
            n_gcps=40,
            checkpoint_rmse_m=2.0,
            random_checkpoint_rmse_m=1.5,
            cv_rmse_m=cv,
            gcp_hull_fraction=0.8,
            gcp_block_occupancy=occ,
            support_file=f"/f/{eid}_support.tif",
            georef_file=f"/f/{eid}.tif",
        )

    df = reg.run_batch(s, runner=runner)
    base = [
        "entity_id", "dataset", "camera", "acquisition_date", "era", "aoi", "method", "n_gcps",
        "checkpoint_rmse_m", "resolution_m", "included_for_change_detection", "warnings",
        "georef_file", "status", "error",
    ]  # fmt: skip
    assert list(df.columns)[: len(base)] == base  # existing columns and order untouched
    assert list(df.columns)[len(base) :] == [
        "cv_rmse_m",
        "random_checkpoint_rmse_m",
        "gcp_hull_fraction",
        "gcp_block_occupancy",
        "support_file",
    ]
    by = df.set_index("entity_id")
    assert by.loc["OK", "included_for_change_detection"]
    assert not by.loc["SPARSE", "included_for_change_detection"]
    assert not by.loc["BADCV", "included_for_change_detection"]
    assert by.loc["OK", "support_file"].endswith("OK_support.tif")
    assert by.loc["OK", "random_checkpoint_rmse_m"] == 1.5
    assert list(reg.load_registry(settings=s).entity_id) == ["OK"]
    md = (s.outputs_public / "georef_summary.md").read_text()
    assert "CV RMSE (m)" in md and "low GCP coverage: 1" in md
