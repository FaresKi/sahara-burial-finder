"""Regression tests for integration bugs found by the end-to-end run (ticket 18)."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from sbf.catalog.report import coverage_map
from sbf.core.aoi import load_aois, load_reference_features
from sbf.fuse.pipeline import _merge_summary
from sbf.georef.overlay import gz_in_frame


def test_georef_package_exports_registry_loaders():
    from sbf.georef import load_registry, load_support  # fuse/report CLIs import these

    assert callable(load_registry) and callable(load_support)


def test_coverage_map_accepts_polygon_reference_features(tmp_project, tmp_path):
    ref = load_reference_features(tmp_project)
    ref.loc[ref["feature_id"] == "GZ1", "geometry"] = box(-0.06, 26.30, -0.05, 26.32)
    cov = pd.DataFrame(columns=["entity_id", "dataset", "acquisition_date", "resolution_class",
                                "availability", "footprint_wkt"])  # fmt: skip
    out = coverage_map(cov, ref, load_aois(tmp_project), tmp_path / "m.html")
    assert out.exists()


def test_gz_overlay_accepts_polygon_features():
    class W:
        transform = __import__("rasterio").transform.Affine(10, 0, 0, 0, -10, 0)
        shape = (10, 10)
        nodata = 0

        def window(self, *a):
            return __import__("numpy").ones((1, 1))

    gz = gpd.GeoDataFrame(
        {"feature_id": ["GZ1"]}, geometry=[box(-0.06, 26.30, -0.05, 26.32)], crs=4326
    )
    assert gz_in_frame(W(), gz) == {}  # no crash; outside the tiny frame


def test_fusion_public_summary_merges_aois():
    a = "# Fusion summary: REG\n\nreg body\n"
    b = "# Fusion summary: INE\n\nine body\n"
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.md"
        p.write_text(_merge_summary(p, "REG", a))
        p.write_text(_merge_summary(p, "INE", b))
        p.write_text(_merge_summary(p, "REG", a.replace("reg body", "reg body v2")))
        text = p.read_text()
    assert text.count("# Fusion summary:") == 2 and "ine body" in text and "reg body v2" in text
