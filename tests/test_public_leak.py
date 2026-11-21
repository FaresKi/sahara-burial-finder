"""SPEC 9.4: nothing but public coordinates under outputs/public (leak guard)."""

import pytest
from typer.testing import CliRunner

from sbf.cli import app
from sbf.core.crs import lonlat_to_xy
from sbf.report.public_check import check_public


def put(s, name, text):
    p = s.outputs_public / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def kinds(s):
    return sorted({(v.file, v.kind) for v in check_public(s)})


def test_clean_and_reference_coordinates_pass(tmp_project):
    s = tmp_project
    put(s, "ok.md", "# metrics\n\nrecall@50 0.500, AP 0.1234, random 0.0123\n")
    put(s, "ref.md", "GZ1 (IAEA): lat 26.3117, lon -0.0572 (public)\n")  # reference feature
    put(s, "aoi.csv", "aoi,lon,lat\nREG,-0.25,26.05\nREG,0.25,26.75\n")  # AOI vertices
    x, y = lonlat_to_xy(-0.0572, 26.3117, "EPSG:32631")
    put(s, "ref_utm.md", f"GZ1 at {x:.1f} {y:.1f}\n")
    assert check_public(s) == []


def test_catalogue_footprint_vertices_pass(tmp_project):
    s = tmp_project
    wkt = "POLYGON ((-0.1612 26.2211, -0.0402 26.2205, -0.0391 26.3290, -0.1603 26.3301, -0.1612 26.2211))"
    put(s, "coverage.csv", f'entity_id,footprint_wkt\nTEST_X,"{wkt}"\n')
    assert check_public(s) == []  # vertices of a public footprint (no cross-vertex pairing)


@pytest.mark.parametrize(
    "text,kind",
    [
        ("candidate at 26.20480, -0.15310 here", "latlon_pair"),
        ("lon,lat\n-0.1531,26.2048\n", "latlon_pair"),
        ("POINT (-0.15310 26.20480)", "latlon_pair"),
    ],
)
def test_non_reference_degrees_fail(tmp_project, text, kind):
    put(tmp_project, "leak.md", text)
    assert ("leak.md", kind) in kinds(tmp_project)


def test_non_reference_utm_and_tile_ids_fail(tmp_project):
    s = tmp_project
    x, y = lonlat_to_xy(-0.1531, 26.2048, "EPSG:32631")
    put(s, "utm.json", f'{{"easting": {x:.1f}, "northing": {y:.1f}}}')
    put(s, "tile.md", "top tile REG-64-00012-00034\n")
    got = kinds(s)
    assert ("utm.json", "utm_pair") in got and ("tile.md", "tile_id") in got


def test_binary_files_ignored_and_values_not_printed(tmp_project):
    s = tmp_project
    (s.outputs_public / "fig.png").write_bytes(b"\x89PNG 26.20480 -0.15310")
    put(s, "leak.md", "26.20480 -0.15310\n")
    r = CliRunner().invoke(app, ["report", "check-public"])
    assert r.exit_code == 6
    assert "26.2048" not in r.output and "0.1531" not in r.output
    assert "fig.png" not in r.output
