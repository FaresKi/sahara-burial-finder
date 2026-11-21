import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Point, box
from typer.testing import CliRunner

from sbf.core.aoi import load_reference_features
from sbf.detect.harness import reference_footprints
from sbf.ingest.refine import (
    RefinementError,
    add_refinement,
    export_feature_chip,
    import_polygon,
    list_refinements,
)

TA1 = (5.05, 24.03)


@pytest.fixture
def ev(tmp_path):
    """Synthetic EPSG:4326 raster around TA1 and a screenshot file."""
    r = tmp_path / "ev.tif"
    b = (5.00, 23.98, 5.10, 24.08)
    data = np.random.default_rng(0).random((3, 100, 100)).astype("float32")
    with rasterio.open(
        r,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=3,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(*b, 100, 100),
    ) as d:
        d.write(data)
    s = tmp_path / "shot.png"
    s.write_bytes(b"png")
    return r, s


def _poly(dx=0.0):
    return box(TA1[0] + dx - 0.002, TA1[1] - 0.002, TA1[0] + dx + 0.002, TA1[1] + 0.002)


def test_loader_without_and_with_refinement(tmp_project, ev):
    r, s = ev
    base = load_reference_features(tmp_project)
    assert not base["refined"].any()
    add_refinement("TA1", _poly(), evidence_raster=r, screenshot=s, settings=tmp_project)
    g = load_reference_features(tmp_project).set_index("feature_id")
    assert g.loc["TA1", "refined"] and g.loc["TA1"].geometry.geom_type == "Polygon"
    assert g.loc["TA1", "original_geometry"] == Point(*TA1).wkt
    assert g.loc["TA1", "refinement_evidence"] == str(r)
    assert not g.loc["PO1", "refined"]
    o = load_reference_features(tmp_project, refined=False).set_index("feature_id")
    assert o.loc["TA1"].geometry.geom_type == "Point" and not o["refined"].any()
    orig = json.loads((tmp_project.config_dir / "reference_features.geojson").read_text())
    assert orig["features"][4]["geometry"]["type"] == "Point"


def test_latest_wins(tmp_project, ev):
    r, s = ev
    add_refinement("TA1", _poly(), evidence_raster=r, screenshot=s, settings=tmp_project)
    add_refinement("TA1", _poly(0.01), evidence_raster=r, screenshot=s, settings=tmp_project)
    g = load_reference_features(tmp_project).set_index("feature_id")
    assert g.loc["TA1"].geometry.centroid.x == pytest.approx(TA1[0] + 0.01)


def test_record_schema(tmp_project, ev):
    r, s = ev
    f = add_refinement(
        "TA1", _poly(), evidence_raster=r, screenshot=s, layer="x", settings=tmp_project
    )
    p = f["properties"]
    assert p["geometry_type"] == "polygon" and p["supersedes_precision"] == "approximate"
    assert len(p["evidence"]["sha256"]) == 64 and p["refined_by"] == "human"


def test_refusals(tmp_project, ev, tmp_path):
    r, s = ev
    kw = {"settings": tmp_project}
    with pytest.raises(RefinementError, match="raster not found"):
        add_refinement("TA1", _poly(), evidence_raster=tmp_path / "no.tif", screenshot=s, **kw)
    with pytest.raises(RefinementError, match="screenshot not found"):
        add_refinement("TA1", _poly(), evidence_raster=r, screenshot=tmp_path / "no.png", **kw)
    with pytest.raises(RefinementError, match="outside AOI"):
        add_refinement("TA1", Point(0, 0), evidence_raster=r, screenshot=s, **kw)
    with pytest.raises(RefinementError, match="raster bounds"):
        add_refinement("TA1", Point(5.15, 24.03), evidence_raster=r, screenshot=s, **kw)
    with pytest.raises(RefinementError, match="unknown feature_id"):
        add_refinement("ZZ", _poly(), evidence_raster=r, screenshot=s, **kw)
    with pytest.raises(RefinementError, match="refined_by"):
        add_refinement("TA1", _poly(), evidence_raster=r, screenshot=s, refined_by="bot", **kw)
    assert not load_reference_features(tmp_project)["refined"].any()


def test_iaea_move_needs_note(tmp_project, tmp_path):
    r = tmp_path / "g.tif"
    b = (-0.12, 26.25, -0.0, 26.37)
    with rasterio.open(
        r,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(*b, 10, 10),
    ) as d:
        d.write(np.ones((1, 10, 10), "float32"))
    s = tmp_path / "s.png"
    s.write_bytes(b"x")
    far = Point(-0.0572, 26.3117 + 0.02)  # ~2.2 km
    with pytest.raises(RefinementError, match="force-note"):
        add_refinement("GZ1", far, evidence_raster=r, screenshot=s, settings=tmp_project)
    f = add_refinement(
        "GZ1", far, evidence_raster=r, screenshot=s, force_note="crater seen", settings=tmp_project
    )
    assert "crater seen" in f["properties"]["notes"]
    near = Point(-0.0572, 26.3117 + 0.002)  # ~220 m
    add_refinement("GZ1", near, evidence_raster=r, screenshot=s, settings=tmp_project)


def test_import_polygon_and_harness(tmp_project, ev, tmp_path):
    import geopandas as gpd

    r, s = ev
    f = tmp_path / "poly.geojson"
    gpd.GeoDataFrame(geometry=[_poly()], crs="EPSG:4326").to_file(f, driver="GeoJSON")
    rec = import_polygon("TA1", f, evidence_raster=r, screenshot=s, settings=tmp_project)
    assert rec["properties"]["method"] == "manual_digitised"
    ref = load_reference_features(tmp_project)
    foot = reference_footprints("INE", ref, "EPSG:32631").set_index("feature_id")
    assert foot.loc["TA1"].geometry.geom_type == "Polygon"
    bad = tmp_path / "two.geojson"
    gpd.GeoDataFrame(geometry=[_poly(), _poly(0.01)], crs="EPSG:4326").to_file(
        bad, driver="GeoJSON"
    )
    with pytest.raises(RefinementError, match="exactly one"):
        import_polygon("TA1", bad, evidence_raster=r, screenshot=s, settings=tmp_project)


def test_chip(tmp_project, ev, tmp_path):
    r, _ = ev
    out = export_feature_chip("TA1", r, tmp_path / "c" / "ta1.png", 1000, settings=tmp_project)
    assert out.read_bytes()[:4] == b"\x89PNG"
    with pytest.raises(RefinementError, match="no geometry"):
        export_feature_chip("BASE", r, tmp_path / "b.png", settings=tmp_project)
    out = export_feature_chip(
        "BASE", r, tmp_path / "b.png", 1000, center_lonlat=(5.05, 24.03), settings=tmp_project
    )
    assert out.exists()


def test_cli_list(tmp_project):
    from sbf.cli import app

    res = CliRunner().invoke(app, ["ingest", "refine", "list"])
    assert res.exit_code == 0, res.output
    assert "TA1\tapproximate\tno" in res.output and "BASE\tto_locate" in res.output
    assert len(list_refinements(tmp_project)) == 7
