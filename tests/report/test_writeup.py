"""Phase 7 write-up: answers come from artefacts only; restricted briefing is gated."""

import json

import pandas as pd
from typer.testing import CliRunner

from sbf.cli import app
from sbf.report.writeup import (
    NEGATIVE,
    POSITIVE,
    UNANSWERED,
    Evidence,
    answer_q1,
    answer_q3,
    answer_q4,
    answer_q5,
    render_report,
    write_briefing,
)


def test_all_unanswered_without_artefacts(tmp_project):
    ev = Evidence.load(tmp_project)
    assert [a(ev).status for a in (answer_q1, answer_q3, answer_q4, answer_q5)] == [UNANSWERED] * 4
    rep = render_report(tmp_project, ev)
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        assert f"### {q}: not yet answered" in rep
    for section in ("## Method", "## Data coverage", "## Georeferencing accuracy",
                    "## Blind-test metrics", "## Limits", "## Negative and null results",
                    "## Recommendations for authorised field-verification partners"):  # fmt: skip
        assert section in rep
    low = rep.lower()
    assert "invisible targets" in low and "resolution floor" in low and "label noise" in low
    assert "drone magnetometry" in low and "ground-penetrating radar" in low
    for banned in ("directions to", "drive to", "access road", "travel to", "waypoint"):
        assert banned not in low


def test_negative_blind_test_and_go_are_reported_as_negative(tmp_project):
    s = tmp_project
    (s.outputs_public / "blind_test_gate.json").write_text(
        json.dumps({"status": "FAIL", "best_rows": [], "warnings": []})
    )
    (s.outputs_public / "go_decision.json").write_text(
        json.dumps({"status": "NO_GO", "reasons": ["A NOT met"]})
    )
    pd.DataFrame(
        {"entity_id": ["X"], "mission": ["KH-4"], "resolution_class": ["3-10m"],
         "availability": ["scanned"], "acquisition_date": ["1962-01-01"],
         "intersecting_features": [""], "footprint_wkt": [""]}
    ).to_csv(s.outputs_public / "coverage.csv", index=False)  # fmt: skip
    ev = Evidence.load(s)
    assert answer_q4(ev).status == NEGATIVE and "valid negative result" in answer_q4(ev).text
    assert answer_q1(ev).status == NEGATIVE
    rep = render_report(s, ev)
    assert "Q4 is negative" in rep


def test_positive_q3_from_registry(tmp_project):
    s = tmp_project
    p = s.data_processed / "historical"
    p.mkdir(parents=True)
    from sbf.georef.registry import REGISTRY_COLUMNS

    row = dict.fromkeys(REGISTRY_COLUMNS)
    row.update(entity_id="TEST_X", status="ok", method="poly2", n_gcps=40, cv_rmse_m=7.5,
               included_for_change_detection=True, acquisition_date=pd.Timestamp("1962-01-01"))  # fmt: skip
    pd.DataFrame([row]).to_parquet(p / "registry.parquet")
    a = answer_q3(Evidence.load(s))
    assert a.status == POSITIVE and "7.5" in a.text and "meets" in a.text


def test_q5_from_prior_summary(tmp_project):
    s = tmp_project
    (s.outputs_public / "prior_eval_summary.json").write_text(
        json.dumps({"rows": [{"d_ap": -0.01, "d_recall_at_50": 0.0}], "n_used": 3, "n_excluded": 1})
    )
    assert answer_q5(Evidence.load(s)).status == NEGATIVE


def test_briefing_needs_ranking_and_cli_requires_both_flags(tmp_project):
    s = tmp_project
    assert write_briefing(s) is None
    r = CliRunner().invoke(app, ["report", "writeup", "--include-restricted"])
    assert r.exit_code == 2
    rk = s.outputs_restricted / "ranking.csv"
    pd.DataFrame(
        {"aoi_id": ["REG"], "rank": [1], "cand_id": ["REG-C0001"], "label": ["ops_1961_1968"],
         "mean_S": [0.9], "area_m2": [4096.0]}
    ).to_csv(rk, index=False)  # fmt: skip
    r = CliRunner().invoke(
        app, ["report", "writeup", "--include-restricted", "--confirm-restricted"]
    )
    assert r.exit_code == 0, r.output
    b = (s.outputs_restricted / "briefing.md").read_text()
    assert "REG-C0001" in b and "ground verification by competent authorities" in b
    assert (s.outputs_public / "report.md").exists()
    assert "REG-C0001" not in (s.outputs_public / "report.md").read_text()
