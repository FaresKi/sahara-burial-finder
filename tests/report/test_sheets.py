import json
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from typer.testing import CliRunner

from sbf.fuse.families import FAMILIES
from sbf.report import cli as report_cli
from sbf.report.confidence import assess_confidence, resolve_blind_test
from sbf.report.sheets import public_stats_md, write_public_stats, write_sheets

CRS = "EPSG:32631"
X0, Y0 = 500000.0, 3000000.0


def conf(**kw):
    base = {
        "label": "ops_1961_1968", "mean_S": 0.95, "families": "hist_change,hist_anomaly",
        "best_rmse_m": 5.0, "n_hist_dates": 3, "prior_contribution": None, "blind_test_passed": True,
    }  # fmt: skip
    base.update(kw)
    return assess_confidence(**base)


def test_confidence_high_and_downgrades():
    assert conf().level == "high"
    assert conf(mean_S=0.85).level == "medium"
    assert conf(best_rmse_m=15).level == "medium"
    assert conf(best_rmse_m=25).level == "low"
    assert conf(best_rmse_m=None).level == "low"
    assert conf(families="hist_anomaly").level == "low"
    assert conf(families="SAR_L,DEM").level == "medium"  # no historical family -> not high
    assert conf(label="other").level == "medium"


def test_confidence_caps():
    c = conf(n_hist_dates=1)
    assert c.level == "medium" and any("single historical date" in j for j in c.justification)
    for lab in ("post_2025_likely_cleanup", "pre_1960_natural_or_unknown"):
        c = conf(label=lab)
        assert c.level == "low" and any(lab in j for j in c.justification)
    c = conf(blind_test_passed=False)
    assert c.level == "low" and any("blind-test" in j for j in c.justification)


def test_confidence_justification_lists_rules():
    j = " | ".join(conf().justification)
    assert "high:" in j and "label=ops_1961_1968" in j


def _make_raster(path, x0=X0, size=100, res=10.0, bands=1):
    rng = np.random.default_rng(0)
    data = rng.random((bands, size, size)).astype("float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=bands, dtype="float32",
        crs=CRS, transform=from_origin(x0, Y0, res, res),
    ) as dst:  # fmt: skip
        dst.write(data)


@pytest.fixture
def world(tmp_project, tmp_path):
    rdir = tmp_path / "rasters"
    rdir.mkdir()
    _make_raster(rdir / "a.tif")
    _make_raster(rdir / "b.tif")
    _make_raster(rdir / "far.tif", x0=900000.0)
    reg = pd.DataFrame(
        {
            "entity_id": ["EB", "EA", "EF"],
            "acquisition_date": ["1965-05-01", "1962-05-01", "1963-01-01"],
            "era": ["pre_1967"] * 3,
            "georef_file": [str(rdir / "b.tif"), str(rdir / "a.tif"), str(rdir / "far.tif")],
            "checkpoint_rmse_m": [8.0, 12.0, 5.0],
            "resolution_m": [10.0] * 3,
        }
    )
    geoms = [box(X0 + 300 + 100 * i, Y0 - 600, X0 + 360 + 100 * i, Y0 - 540) for i in range(4)]
    ranked = gpd.GeoDataFrame(
        {
            "rank": [1, 2, 3, 4],
            "cand_id": [f"REG-c{i}" for i in range(4)],
            "label": ["ops_1961_1968", "other", "post_2025_likely_cleanup", "other"],
            "bracket": ["1962-05-01 .. 1965-05-01"] * 4,
            "mean_S": [0.95, 0.8, 0.9, 0.3],
            "area_m2": [3600.0] * 4,
            "families": ["hist_change,hist_anomaly", "hist_anomaly,DEM", "hist_anomaly", "DEM"],
        },
        geometry=geoms,
        crs=CRS,
    )
    ft = pd.DataFrame({"prior": [0.2] * 4, "hist_anomaly": [0.9] * 4},
                      index=ranked["cand_id"])  # fmt: skip
    return ranked, reg, ft, tmp_project


BLOCK = ("route to", "access via", "drive", "visit the site")


def test_write_sheets(world, monkeypatch):
    monkeypatch.delenv("RESTRICTED_KEY", raising=False)
    ranked, reg, ft, s = world
    paths = write_sheets(ranked, reg, "REG", CRS, top_n=3, blind_test_passed=True,
                         family_table=ft, settings=s)  # fmt: skip
    restricted = s.outputs_restricted.resolve()
    assert all(p.resolve().is_relative_to(restricted) for p in paths)
    sheets = sorted((restricted / "candidate_sheets").glob("REG-c*.html"))
    assert len(sheets) == 3
    assert not list(s.outputs_public.rglob("*"))
    html = sheets[0].read_text()
    assert html.count("data:image/png;base64,") == 2  # a.tif and b.tif; far.tif excluded
    assert html.index("1962-05-01 EA") < html.index("1965-05-01 EB")  # chronological
    assert "checkpoint RMSE 12.00 m" in html and "200 m" in html
    assert "Candidate area for verification — not a confirmed burial site." in html
    assert "Requires ground verification by competent authorities." in html
    assert "1–2 m" in html and "Label-noise caveat" in html
    assert "Prior contribution (documents): 0.20" in html
    assert "Confidence justification" in html and "<strong>high</strong>" in html
    # self-contained: no external references
    assert not re.search(r'(src|href)="(?!data:)', html)
    assert "http" not in html.lower()
    for sh in sheets:
        low = sh.read_text().lower()
        assert not any(b in low for b in BLOCK)
    idx = (restricted / "candidate_sheets" / "index.html").read_text()
    assert "REG-c0.html" in idx and "REG-c3" not in idx
    assert (restricted / "candidate_confidence_REG.csv").exists()


def test_blind_test_failed_caps_low(world, monkeypatch):
    monkeypatch.delenv("RESTRICTED_KEY", raising=False)
    ranked, reg, _ft, s = world
    write_sheets(ranked, reg, "REG", CRS, top_n=1, blind_test_passed=False, settings=s)
    t = pd.read_csv(s.outputs_restricted / "candidate_confidence_REG.csv")
    assert set(t["confidence"]) == {"low"}


def test_public_stats_no_leaks(world, monkeypatch):
    monkeypatch.delenv("RESTRICTED_KEY", raising=False)
    ranked, reg, _ft, s = world
    write_sheets(ranked, reg, "REG", CRS, top_n=2, blind_test_passed=True, settings=s)
    out = write_public_stats(s)
    txt = out.read_text()
    assert out.parent == s.outputs_public.resolve() or out.parent == s.outputs_public
    assert "| REG | medium | other | 0.7-0.9 | 1 |" in txt
    assert "REG-c" not in txt and "EA" not in txt and "EB" not in txt
    assert not re.search(r"\d{5,}|\d+\.\d{3,}", txt)  # no coordinate-like numbers
    assert "not a confirmed burial site" in txt


def test_public_stats_md_empty():
    assert "AOI" in public_stats_md({"REG": pd.DataFrame()})


def test_cli_gate(world):
    r = CliRunner().invoke(report_cli.app, ["sheets", "--aoi", "REG"])
    assert r.exit_code == 2
    assert "sign-off" in r.output
    r = CliRunner().invoke(report_cli.app, ["public-stats"])
    assert r.exit_code == 1  # nothing to summarise yet, but not gated


def test_cli_sheets_end_to_end(world, monkeypatch):
    monkeypatch.delenv("RESTRICTED_KEY", raising=False)
    ranked, reg, _, s = world
    ranked.to_file(s.outputs_restricted / "candidates_REG.gpkg", layer="candidates", driver="GPKG")
    import sys
    import types

    mod = types.ModuleType("sbf.georef")
    mod.load_registry = lambda included_only=True: reg
    monkeypatch.setitem(sys.modules, "sbf.georef", mod)
    r = CliRunner().invoke(
        report_cli.app, ["sheets", "--aoi", "REG", "--top", "2", "--confirm-restricted"]
    )
    assert r.exit_code == 0, r.output
    assert "unencrypted" in r.output
    assert "REG-c" not in r.output
    assert (s.outputs_restricted / "candidate_sheets" / "REG-c0.html").exists()


# ------------------------------------------------------------------ fusion columns / gate
def _with_fusion_cols(ranked, n_present=3):
    r = ranked.copy()
    for f in FAMILIES:
        r[f"fam_{f}"] = np.nan
        r[f"contrib_{f}"] = np.nan
    r["fam_hist_anomaly"], r["contrib_hist_anomaly"] = 0.9, 0.75
    r["fam_prior"], r["contrib_prior"] = 0.4, 0.25
    r["n_present"] = n_present
    return r


def test_sheet_shows_family_table_and_prior_number(world, monkeypatch):
    monkeypatch.delenv("RESTRICTED_KEY", raising=False)
    ranked, reg, _ft, s = world
    write_sheets(_with_fusion_cols(ranked), reg, "REG", CRS, top_n=1, blind_test_passed=True,
                 settings=s)  # fmt: skip
    html = (s.outputs_restricted / "candidate_sheets" / "REG-c0.html").read_text()
    assert "Prior contribution (documents): 0.25" in html
    assert "mean 0.90, contribution 0.75" in html
    assert "<td>SAR_L</td><td>mean n/a, contribution n/a</td>" in html


def test_single_date_cap_uses_n_present(world, monkeypatch):
    monkeypatch.delenv("RESTRICTED_KEY", raising=False)
    ranked, reg, _ft, s = world  # c0 is covered by 2 frames but would be high otherwise
    write_sheets(_with_fusion_cols(ranked, n_present=1), reg, "REG", CRS, top_n=1,
                 blind_test_passed=True, settings=s)  # fmt: skip
    t = pd.read_csv(s.outputs_restricted / "candidate_confidence_REG.csv")
    assert t.set_index("cand_id").loc["REG-c0", "confidence"] == "medium"
    write_sheets(_with_fusion_cols(ranked, n_present=2), reg, "REG", CRS, top_n=1,
                 blind_test_passed=True, settings=s)  # fmt: skip
    t = pd.read_csv(s.outputs_restricted / "candidate_confidence_REG.csv")
    assert t.set_index("cand_id").loc["REG-c0", "confidence"] == "high"


def test_fallback_warning_without_columns(world, monkeypatch, caplog):
    monkeypatch.delenv("RESTRICTED_KEY", raising=False)
    ranked, reg, _ft, s = world
    with caplog.at_level("WARNING", logger="sbf.report.sheets"):
        write_sheets(ranked, reg, "REG", CRS, top_n=1, blind_test_passed=True, settings=s)
    assert any("lacks fam_*/contrib_*/n_present" in m for m in caplog.messages)
    caplog.clear()
    with caplog.at_level("WARNING", logger="sbf.report.sheets"):
        write_sheets(_with_fusion_cols(ranked), reg, "REG", CRS, top_n=1, blind_test_passed=True,
                     settings=s)  # fmt: skip
    assert not any("lacks" in m for m in caplog.messages)


@pytest.mark.parametrize(
    "status,flag,expected,src",
    [
        ("PASS", None, True, "blind_test_gate.json"),
        ("FAIL", None, False, "blind_test_gate.json"),
        ("NOT_RUN", None, False, "blind_test_gate.json"),
        ("FAIL", True, True, "flag"),
        ("PASS", False, False, "flag"),
        (None, None, False, "assuming FAILED"),
    ],
)
def test_resolve_blind_test(tmp_path, status, flag, expected, src):
    if status:
        (tmp_path / "blind_test_gate.json").write_text(json.dumps({"status": status}))
    passed, source = resolve_blind_test(flag, tmp_path)
    assert passed is expected and src in source


def _run_cli(world, monkeypatch, args):
    import sys
    import types

    monkeypatch.delenv("RESTRICTED_KEY", raising=False)
    ranked, reg, _, s = world
    ranked.to_file(s.outputs_restricted / "candidates_REG.gpkg", layer="candidates", driver="GPKG")
    mod = types.ModuleType("sbf.georef")
    mod.load_registry = lambda included_only=True: reg
    monkeypatch.setitem(sys.modules, "sbf.georef", mod)
    r = CliRunner().invoke(
        report_cli.app, ["sheets", "--aoi", "REG", "--top", "1", "--confirm-restricted", *args]
    )
    assert r.exit_code == 0, r.output
    t = pd.read_csv(s.outputs_restricted / "candidate_confidence_REG.csv")
    return r.output, dict(zip(t["cand_id"], t["confidence"], strict=True))


def test_cli_gate_json_and_override(world, monkeypatch):
    s = world[3]
    out, levels = _run_cli(world, monkeypatch, [])  # no file, no flag
    assert "assuming FAILED" in out and set(levels.values()) == {"low"}
    s.outputs_public.mkdir(parents=True, exist_ok=True)
    (s.outputs_public / "blind_test_gate.json").write_text(json.dumps({"status": "PASS"}))
    out, levels = _run_cli(world, monkeypatch, [])
    assert "PASSED" in out and "blind_test_gate.json" in out and levels["REG-c0"] == "high"
    out, levels = _run_cli(world, monkeypatch, ["--blind-test-failed"])
    assert "explicit command-line flag" in out and set(levels.values()) == {"low"}
