import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point
from typer.testing import CliRunner

from sbf.catalog import GoDecision, best_frames, collect_counts, evaluate_go
from sbf.catalog.report import build_report
from sbf.cli import app
from sbf.core.aoi import load_aois, load_reference_features

REF = gpd.GeoDataFrame(
    {"feature_id": ["GZ1", "GZ2", "TA1", "BASE"]},
    geometry=[Point(0, 0), Point(1, 1), Point(2, 2), None],
    crs="EPSG:4326",
)


def frame(eid, feats, date, res="<=3m", cloud=5.0, avail="scanned"):
    return {
        "entity_id": eid,
        "dataset": "test_declass_1",
        "mission": "m",
        "camera": "c",
        "acquisition_date": date,
        "cloud_cover": cloud,
        "availability": avail,
        "browse_url": "",
        "footprint_wkt": "",
        "intersecting_features": feats,
        "resolution_class": res,
        "aoi_id": "REG",
        "raw_file": "f.json",
    }


def cov(*rows):
    return pd.DataFrame(rows)


def test_go_when_both_criteria_met():
    d = evaluate_go(
        cov(frame("A", "GZ1", "1962-01-01"), frame("B", "GZ1", "1970-01-01", res="3-10m")), REF
    )
    assert d.go and d.usable_hires_frames == ["A"] and d.temporal_features == ["GZ1"]


def test_no_go_empty():
    d = evaluate_go(cov(), REF)
    assert (
        d.status == "NO_GO"
        and any("A NOT met" in r for r in d.reasons)
        and any("B NOT met" in r for r in d.reasons)
    )


def test_no_go_resolution_too_coarse():
    d = evaluate_go(
        cov(
            frame("A", "GZ1", "1962-01-01", res="3-10m"),
            frame("B", "GZ1", "1970-01-01", res="3-10m"),
        ),
        REF,
    )
    assert not d.go and not d.usable_hires_frames and d.temporal_features == ["GZ1"]


def test_cloudy_is_no_go_and_unknown_cloud_needs_review():
    cloudy = frame("A", "GZ1", "1962-01-01", cloud=30.0)  # 30 is not < 30
    post = frame("B", "GZ1", "1970-01-01", cloud=None)
    d = evaluate_go(cov(cloudy, post), REF)
    assert d.status == "NEEDS_REVIEW" and d.review_entity_ids == ["B"]  # cloudy A never counts
    assert (
        evaluate_go(cov(cloudy, frame("B", "GZ1", "1970-01-01", res="3-10m")), REF).status
        == "NO_GO"
    )
    d = evaluate_go(
        cov(
            frame("C", "GZ1", "1962-01-01", cloud=None),
            frame("D", "GZ1", "1970-01-01", res="3-10m"),
        ),
        REF,
    )
    assert d.status == "NEEDS_REVIEW" and d.review_entity_ids == ["C"] and not d.go


def test_review_answers_resolve_needs_review():
    c = cov(
        frame("C", "GZ1", "1962-01-01", cloud=None),
        frame("E", "GZ1", "1963-01-01", cloud=None),
        frame("D", "GZ1", "1970-01-01", res="3-10m"),
    )
    d = evaluate_go(c, REF)
    assert d.status == "NEEDS_REVIEW" and d.review_entity_ids == ["C", "E"]
    assert evaluate_go(c, REF, review={"C": True}).status == "GO"
    d = evaluate_go(c, REF, review={"C": False})
    assert d.status == "NEEDS_REVIEW" and d.review_entity_ids == ["E"]
    assert evaluate_go(c, REF, review={"C": False, "E": False}).status == "NO_GO"
    # a human answer never overrides a known cloud value
    assert (
        evaluate_go(
            cov(
                frame("A", "GZ1", "1962-01-01", cloud=90.0),
                frame("D", "GZ1", "1970-01-01", res="3-10m"),
            ),
            REF,
            review={"A": True},
        ).status
        == "NO_GO"
    )


def test_unknown_cloud_without_temporal_pair_is_no_go():
    assert evaluate_go(cov(frame("C", "GZ1", "1962-01-01", cloud=None)), REF).status == "NO_GO"


def test_no_go_only_one_side_of_1967():
    d = evaluate_go(cov(frame("A", "GZ1", "1962-01-01"), frame("B", "GZ1", "1966-12-31")), REF)
    assert not d.go and d.temporal_features == []
    d = evaluate_go(cov(frame("A", "GZ1", "1962-01-01"), frame("B", "GZ1", "1967-06-01")), REF)
    assert not d.go  # 1967 itself is neither pre nor post


def test_temporal_must_be_same_feature_and_a_must_be_go_feature():
    # pre on GZ1, post on TA1 only: B not met
    assert not evaluate_go(
        cov(frame("A", "GZ1", "1962-01-01"), frame("B", "TA1", "1970-01-01")), REF
    ).go
    # hi-res frame only over a non-GO feature (PO1-like id) does not satisfy A
    ref = gpd.GeoDataFrame({"feature_id": ["PO1"]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    assert not evaluate_go(
        cov(frame("A", "PO1", "1962-01-01"), frame("B", "PO1", "1970-01-01")), ref
    ).go


def test_best_frames_prefers_resolution_then_free_then_earliest():
    c = cov(
        frame("coarse", "GZ1", "1960-01-01", res="3-10m"),
        frame("paid", "GZ1", "1961-01-01", avail="needs_order"),
        frame("free_late", "GZ1", "1969-01-01"),
        frame("free_early", "GZ1", "1964-01-01"),
    )
    bf = best_frames(c, REF).set_index("feature_id")
    assert bf.loc["GZ1", "entity_id"] == "free_early" and bf.loc["GZ1", "era"] == "pre-1967"
    assert bf.loc["GZ2", "n_frames"] == 0


def test_counts_never_filled_in():
    geoms = {"REG": Point(0, 0).buffer(1)}
    out = collect_counts(None, geoms, "1959-01-01", "2026-01-01")
    status = {(c.sensor, c.status) for c in out}
    assert status == {("landsat", "not_run"), ("sentinel2", "not_run"), ("spot", "unverified")}
    assert all(c.count is None for c in out)

    class Counter:
        def count_scenes(self, sensor, aoi, start, end):
            if sensor == "sentinel2":
                raise RuntimeError("gee down")
            return 7 if start == "1972-01-01" else -1

    out = {c.sensor: c for c in collect_counts(Counter(), geoms, "1959-01-01", "2026-01-01")}
    assert (
        out["landsat"].count == 7
        and out["sentinel2"].status == "failed"
        and out["sentinel2"].count is None
    )


def _write_cov(settings, rows):
    pub = settings.outputs_public
    pub.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(pub / "coverage.csv", index=False)


def test_build_report_go_and_no_go(tmp_project):
    aois, ref = load_aois(tmp_project), load_reference_features(tmp_project)
    wkt = "POLYGON ((-0.3 26.1, 0.1 26.1, 0.1 26.5, -0.3 26.5, -0.3 26.1))"
    good = [
        {**frame("TEST_FIXTURE_0001", "GZ1", "1962-01-01"), "footprint_wkt": wkt},
        {
            **frame("TEST_FIXTURE_0002", "GZ1", "1970-01-01", avail="needs_order"),
            "footprint_wkt": wkt,
        },
    ]
    _write_cov(tmp_project, good)
    counts = collect_counts(None, {"REG": aois.geometry.iloc[0]}, "1959-01-01", "2026-01-01")
    pub = tmp_project.outputs_public
    (pub / "NO_GO.md").write_text("stale")
    assert build_report(tmp_project, aois, ref, counts).go
    assert not (pub / "NO_GO.md").exists()
    md = (pub / "coverage_summary.md").read_text()
    assert "TEST_FIXTURE_0001" in md and "free (already scanned)" in md and "unverified" in md
    assert "TEST_FIXTURE_0001" in (pub / "coverage_map.html").read_text()
    assert json.loads((pub / "sensor_counts.json").read_text())[0]["status"] == "not_run"

    _write_cov(tmp_project, [good[1]])
    assert not build_report(tmp_project, aois, ref, counts).go
    assert "B NOT met" in (pub / "NO_GO.md").read_text()


def test_cli_report_exit_codes(tmp_project):
    r = CliRunner()
    assert r.invoke(app, ["catalog", "report"]).exit_code == 2  # no coverage.csv
    _write_cov(tmp_project, [frame("TEST_FIXTURE_0002", "GZ1", "1970-01-01")])
    res = r.invoke(app, ["catalog", "report"])
    assert res.exit_code == 1 and (tmp_project.outputs_public / "NO_GO.md").exists()
    wkt = "POLYGON ((-0.3 26.1, 0.1 26.1, 0.1 26.5, -0.3 26.5, -0.3 26.1))"
    _write_cov(
        tmp_project,
        [
            {**frame("TEST_FIXTURE_0001", "GZ1", "1962-01-01"), "footprint_wkt": wkt},
            frame("TEST_FIXTURE_0002", "GZ1", "1970-01-01"),
        ],
    )
    assert r.invoke(app, ["catalog", "report"]).exit_code == 0


def test_cli_report_needs_review_flow(tmp_project):
    r = CliRunner()
    wkt = "POLYGON ((-0.3 26.1, 0.1 26.1, 0.1 26.5, -0.3 26.5, -0.3 26.1))"
    _write_cov(
        tmp_project,
        [
            {**frame("TEST_FIXTURE_0001", "GZ1", "1962-01-01", cloud=None), "footprint_wkt": wkt},
            frame("TEST_FIXTURE_0002", "GZ1", "1970-01-01", res="3-10m"),
        ],
    )
    res = r.invoke(app, ["catalog", "report"])
    pub = tmp_project.outputs_public
    assert res.exit_code == 3 and "TEST_FIXTURE_0001" in res.output
    assert not (pub / "NO_GO.md").exists()
    rev = pd.read_csv(pub / "browse_review.csv", dtype=str).fillna("")
    assert list(rev.columns) == [
        "entity_id",
        "dataset",
        "date",
        "resolution_class",
        "features",
        "browse_url",
        "usable",
    ]
    assert rev["entity_id"].tolist() == ["TEST_FIXTURE_0001"] and rev["usable"].tolist() == [""]
    assert "NEEDS_REVIEW" in (pub / "coverage_summary.md").read_text()
    rev["usable"] = "yes"
    rev.to_csv(pub / "browse_review.csv", index=False)
    assert r.invoke(app, ["catalog", "report"]).exit_code == 0
    rev["usable"] = "no"
    rev.to_csv(pub / "browse_review.csv", index=False)
    assert r.invoke(app, ["catalog", "report"]).exit_code == 1
    assert (pub / "NO_GO.md").exists()


def test_cli_search_requires_credentials(tmp_project, monkeypatch):
    monkeypatch.delenv("USGS_USERNAME", raising=False)
    monkeypatch.delenv("USGS_M2M_TOKEN", raising=False)
    res = CliRunner().invoke(app, ["catalog", "search", "--aoi", "REG"])
    assert res.exit_code == 2
    assert CliRunner().invoke(app, ["catalog", "search", "--aoi", "XXX"]).exit_code == 2


def test_godecision_is_dataclass():
    assert GoDecision().reasons == [] and GoDecision().status == "NO_GO"


@pytest.mark.parametrize("d", ["1962-01-01", None])
def test_era_handles_missing(d):
    from sbf.catalog.go import era

    assert era(d) in {"pre-1967", "undated"}


def test_report_rerun_keeps_suggestion_columns(tmp_project):
    r = CliRunner()
    wkt = "POLYGON ((-0.3 26.1, 0.1 26.1, 0.1 26.5, -0.3 26.5, -0.3 26.1))"
    _write_cov(
        tmp_project,
        [
            {**frame("TEST_FIXTURE_0001", "GZ1", "1962-01-01", cloud=None), "footprint_wkt": wkt},
            frame("TEST_FIXTURE_0002", "GZ1", "1970-01-01", res="3-10m"),
        ],
    )
    assert r.invoke(app, ["catalog", "report"]).exit_code == 3
    p = tmp_project.outputs_public / "browse_review.csv"
    rev = pd.read_csv(p, dtype=str).fillna("")
    rev["suggested_usable"] = "no"
    rev.to_csv(p, index=False)
    assert r.invoke(app, ["catalog", "report"]).exit_code == 3
    again = pd.read_csv(p, dtype=str).fillna("")
    assert again["suggested_usable"].tolist() == ["no"] and again["usable"].tolist() == [""]
