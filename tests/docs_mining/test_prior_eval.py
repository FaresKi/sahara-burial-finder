import math
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from test_prior import _row, small_grid, write_review

from sbf.core.crs import lonlat_to_xy
from sbf.detect.scores import read_scores, write_scores
from sbf.docs_mining.prior import Claim, resolve_claims
from sbf.docs_mining.prior_eval import evaluate_prior, leaking_claims, render_report
from sbf.features.tiling import make_grid
from sbf.ingest.modern.grid import grid_bounds

GZ1 = (-0.0572, 26.3117)
GZ4 = (-0.0733, 26.3217)
KS = (10, 50, 100)


def toward_gz1():
    """Distance (km) and compass azimuth (deg) from GZ4 to GZ1, computed from the public points."""
    x4, y4 = lonlat_to_xy(*GZ4, "EPSG:32631")
    x1, y1 = lonlat_to_xy(*GZ1, "EPSG:32631")
    return math.hypot(x1 - x4, y1 - y4) / 1000, math.degrees(math.atan2(x1 - x4, y1 - y4)) % 360


def _claims():
    d, az = toward_gz1()
    return [
        # points straight at GZ1 (leaks), flag not set: must be caught geometrically
        Claim("leak", "Gerboise Verte", d, 0.15, f"{az:.0f}", 8.0, "official", False),
        # describes a reference feature by flag
        Claim("flag", "GZ3", 3.0, 0.5, "N", 20.0, "testimony", True),
        # harmless: 3 km north of GZ3 (no reference feature there)
        Claim("ok", "GZ3", 3.0, 0.5, "N", 20.0, "secondary", False),
        Claim("unk", "Mount Nowhere", 3.0, 0.5, "N", 20.0, "secondary", False),
    ]


@pytest.fixture
def synthetic(tmp_project):
    grid = small_grid(center_lonlat=GZ1, half_m=6000, res=10.0)
    g = make_grid("REG", 128, tmp_project.overlap, tmp_project, bounds=grid_bounds(grid))
    n = len(g)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"tile_id": g["tile_id"], "aoi_id": "REG", "size_m": 128, "detector": "texture",
         "layer": "L", "score": rng.random(n)}
    )  # fmt: skip
    write_scores(df, tmp_project)
    return tmp_project, grid, read_scores("REG", settings=tmp_project)


def test_leakage_rule(tmp_project):
    from sbf.core.aoi import load_reference_features
    from sbf.detect.harness import reference_footprints

    grid = small_grid(center_lonlat=GZ1, half_m=6000)
    rcs = resolve_claims(_claims(), "REG", settings=tmp_project)
    foot = reference_footprints("REG", load_reference_features(tmp_project), "EPSG:32631")
    out = leaking_claims(rcs, grid, foot)
    assert set(out) == {"leak"}  # "flag" is excluded by the flag rule, not geometry
    assert "GZ1" in out["leak"]
    assert "ok" not in out and "unk" not in out


def test_eval_unfiltered_improves_and_filtered_excludes(synthetic):
    s, grid, scores = synthetic
    kw = {"claims": _claims()[:1], "scores": scores, "grid": grid, "ks": KS, "write": False,
          "exclusions": gpd.GeoDataFrame()}  # fmt: skip
    raw = evaluate_prior("REG", settings=s, leakage_control=False, **kw)
    assert raw.excluded == {} and raw.used_ids == ["leak"]
    prod = raw.table[raw.table["variant"] == "product"].iloc[0]
    assert prod["recall_at_10_after"] > prod["recall_at_10_before"]
    assert prod["d_recall_at_10"] > 0 and prod["d_ap"] > 0
    assert set(raw.table["variant"]) == {"product", "alpha"}

    ctl = evaluate_prior("REG", settings=s, **kw)
    assert set(ctl.excluded) == {"leak"} and "GZ1" in ctl.excluded["leak"]
    assert ctl.table.empty and "no usable claims remain" in ctl.note


def test_eval_end_to_end_report_and_cli(synthetic):
    s, grid, scores = synthetic
    write_review(
        s,
        [
            _row("leak", "Gerboise Verte", distance_km=f"{toward_gz1()[0]:.3f}",
                 distance_uncertainty_km=0.15, bearing=f"{toward_gz1()[1]:.0f}",
                 bearing_uncertainty_deg=8),
            _row("flag", "GZ3", distance_km=3, bearing="N", describes_reference_feature="True"),
            _row("ok", "GZ3", distance_km=3, distance_uncertainty_km=0.5, bearing="N",
                 confidence_source="secondary"),
            _row("unk", "Mount Nowhere"),
            _row("rej", "GZ1", accept="n"),
        ],
    )  # fmt: skip
    from sbf.docs_mining.prior import load_accepted_claims

    res = evaluate_prior(
        "REG", settings=s, claims=load_accepted_claims(s), scores=scores, grid=grid, ks=KS
    )
    assert set(res.excluded) == {"leak", "flag"}
    assert res.used_ids == ["ok"] and res.unresolved == {"unknown_anchor": ["unk"]}
    assert not res.table.empty
    text = (s.outputs_public / "prior_eval.md").read_text()
    assert "`leak`" in text and "`flag`" in text and "describes_reference_feature" in text
    assert "synthetic" not in text and not re.search(r"2[0-9]\.\d{3,}", text)
    assert text == render_report("REG", res, KS)
