"""run-all orchestration: replay client, gates, dry-run, selection, incrementality (no network)."""

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from sbf.catalog import M2MError
from sbf.catalog.m2m import FixtureClient, RawStore
from sbf.catalog.replay import ReplayClient
from sbf.cli import app
from sbf.pipeline import build_stages


def run(*args):
    return CliRunner().invoke(app, ["run-all", *args])


def test_replay_client_serves_saved_responses_and_refuses_unknown(tmp_path):
    store = RawStore(tmp_path)
    env = {"data": [1, 2], "errorCode": None, "errorMessage": None}
    FixtureClient({"dataset-search": [env]}, store).call("dataset-search", {"datasetName": "x"})
    rc = ReplayClient(tmp_path)
    assert rc.call("dataset-search", {"datasetName": "x"}) == env
    assert rc.last_raw_file and rc.last_raw_file.startswith("dataset-search_")
    with pytest.raises(M2MError, match="no saved raw response"):
        rc.call("dataset-search", {"datasetName": "other"})
    assert rc.has_endpoint("dataset-search") and not rc.has_endpoint("scene-search")


def test_stage_names_unique_and_documented():
    names = [s.name for s in build_stages()]
    assert len(names) == len(set(names))
    assert (
        names[0] == "catalog-replay"
        and names[1] == "catalog-report"
        and names[-1] == "check-public"
    )
    assert {s.name for s in build_stages() if s.restricted} == {"fuse", "candidate-sheets"}


def test_dry_run_prints_plan_and_writes_nothing(tmp_project):
    r = run("--dry-run")
    assert r.exit_code == 0, r.output
    for st in build_stages():
        assert st.name in r.output
    assert "missing input" in r.output
    assert not (tmp_project.data_interim / "run_manifest.json").exists()


def test_unknown_stage_and_half_restricted_flags_are_usage_errors(tmp_project):
    assert run("--only", "nope").exit_code == 2
    assert run("--include-restricted").exit_code == 2
    assert run("--confirm-restricted").exit_code == 2


def test_missing_inputs_are_skipped_honestly_and_report_says_so(tmp_project):
    r = run()
    assert r.exit_code == 0, r.output
    man = json.loads((tmp_project.data_interim / "run_manifest.json").read_text())["stages"]
    assert man["catalog-replay"]["status"] == "skipped"
    assert "missing input" in man["catalog-replay"]["detail"]
    rep = (tmp_project.outputs_public / "report.md").read_text()
    assert "**positive**" not in rep  # nothing is claimed without artefacts
    assert "### Q1: not yet answered" in rep and "### Q4: not yet answered" in rep
    assert "catalog-replay" in rep  # skip reasons are listed in the report
    assert not list(tmp_project.outputs_restricted.glob("*"))  # restricted off by default


def _coverage(s, rows):
    cols = ["entity_id", "dataset", "mission", "camera", "acquisition_date", "cloud_cover",
            "availability", "browse_url", "footprint_wkt", "intersecting_features",
            "resolution_class", "aoi_id", "raw_file"]  # fmt: skip
    pd.DataFrame(rows, columns=cols).to_csv(s.outputs_public / "coverage.csv", index=False)


FP = "POLYGON ((-0.2 26.2, 0.0 26.2, 0.0 26.4, -0.2 26.4, -0.2 26.2))"


def test_no_go_stops_pipeline_with_exit_4_and_bypass_stamps_outputs(tmp_project):
    s = tmp_project
    _coverage(
        s,
        [
            [
                "TEST_A",
                "d",
                "KH-4",
                "c",
                "1962-01-01",
                5.0,
                "scanned",
                "",
                FP,
                "GZ1",
                "3-10m",
                "REG",
                "",
            ]
        ],
    )
    r = run("--from", "catalog-report")
    assert r.exit_code == 4, r.output
    assert "NO_GO" in r.output
    assert not (s.outputs_public / "report.md").exists()  # nothing downstream ran
    assert (s.outputs_public / "NO_GO.md").exists()

    r = run("--from", "catalog-report", "--continue-past-gates")
    assert r.exit_code == 0, r.output
    assert "NOT VALID: gates bypassed" in r.output
    assert (s.outputs_public / "NOT_VALID.txt").exists()
    assert "NOT VALID: gates bypassed" in (s.outputs_public / "report.md").read_text()

    r = run("--from", "catalog-report")  # a normal run clears the stamp and stops again
    assert r.exit_code == 4
    assert not (s.outputs_public / "NOT_VALID.txt").exists()


def test_needs_review_stops_with_exit_3(tmp_project):
    s = tmp_project
    rows = [
        [
            "TEST_A",
            "d",
            "KH-4B",
            "c",
            "1962-01-01",
            None,
            "scanned",
            "",
            FP,
            "GZ1",
            "<=3m",
            "REG",
            "",
        ],
        [
            "TEST_B",
            "d",
            "KH-9",
            "c",
            "1973-01-01",
            5.0,
            "scanned",
            "",
            FP,
            "GZ1",
            "3-10m",
            "REG",
            "",
        ],
    ]
    _coverage(s, rows)
    r = run("--only", "catalog-report")
    assert r.exit_code == 3, r.output
    assert "NEEDS_REVIEW" in r.output


def test_only_from_and_incremental(tmp_project):
    s = tmp_project
    assert run("--only", "writeup").exit_code == 0
    r = run("--only", "writeup")
    assert "[writeup] up-to-date" in r.output
    r = run("--only", "writeup", "--force")
    assert "[writeup] running" in r.output
    (s.outputs_public / "coverage.csv").write_text("changed")  # an input of writeup changed
    assert "[writeup] running" in run("--only", "writeup").output
