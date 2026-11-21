"""SPEC section 8/9: end-to-end `sbf run-all` on a fully synthetic raw dataset (no network)."""

import json

import pandas as pd
import pytest
from synth_world import ENTITIES, setup_project
from typer.testing import CliRunner

from sbf.cli import app


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    import os
    import shutil

    from synth_world import REPO

    from sbf.core.config import load_settings

    root = tmp_path_factory.mktemp("e2e")
    shutil.copytree(REPO / "config", root / "config")
    for d in ("data/raw", "data/interim", "data/processed", "outputs/public", "outputs/restricted"):
        (root / d).mkdir(parents=True)
    old = {k: os.environ.get(k) for k in ("SBF_ROOT", "HF_HOME", "HF_HUB_CACHE", "RESTRICTED_KEY")}
    os.environ["SBF_ROOT"] = str(root)
    os.environ["HF_HOME"] = str(root / "no_hf_cache")  # no DINOv2 weights: embedding stage skips
    os.environ.pop("HF_HUB_CACHE", None)
    os.environ.pop("RESTRICTED_KEY", None)
    s = load_settings(root)
    truth = setup_project(s)
    yield s, truth
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def invoke(*args):
    return CliRunner().invoke(app, list(args))


@pytest.fixture(scope="module")
def run1(world):
    s, _ = world
    r = invoke("run-all", "--same-era", "--include-restricted", "--confirm-restricted")
    return s, r


def test_run_all_exit_and_stage_accounting(run1):
    s, r = run1
    assert r.exit_code == 0, r.output
    man = json.loads((s.data_interim / "run_manifest.json").read_text())
    stages = man["stages"]
    from sbf.pipeline import build_stages

    for st in build_stages():
        rec = stages[st.name]  # every stage is accounted for
        assert rec["status"] in {"ok", "skipped"}, (st.name, rec)
        if rec["status"] == "skipped":
            assert rec["detail"], st.name  # skipped with a reason
    assert (
        "no accepted rows" in stages["docs-prior"]["detail"]
        or "review.csv" in stages["docs-prior"]["detail"]
    )
    assert (
        "DINOv2" in stages["detect-embedding"]["detail"]
        or "torch" in stages["detect-embedding"]["detail"]
    )


def test_catalogue_replayed_and_go(run1):
    s, _ = run1
    cov = pd.read_csv(s.outputs_public / "coverage.csv")
    assert set(cov["entity_id"]) >= set(ENTITIES.values())
    assert json.loads((s.outputs_public / "go_decision.json").read_text())["status"] == "GO"


def test_georef_includes_both_frames(run1):
    s, _ = run1
    from sbf.georef.registry import load_registry

    reg = load_registry(included_only=True, settings=s)
    assert set(reg["entity_id"]) == set(ENTITIES.values()), reg[
        ["entity_id", "status", "error", "cv_rmse_m"]
    ]
    assert (reg["cv_rmse_m"] <= 20).all()


def test_change_detection_pair_tables(run1):
    s, _ = run1
    tables = sorted(p.name for p in (s.data_processed / "scores" / "REG").glob("change_diff__*"))
    assert any("pair_" in t for t in tables), tables
    assert any("1962" in t and "1965" in t for t in tables), tables  # same-era pair


def test_blind_test_gate_and_report(run1):
    s, _ = run1
    gate = json.loads((s.outputs_public / "blind_test_gate.json").read_text())
    assert gate["status"] == "PASS", gate
    rep = (s.outputs_public / "report.md").read_text()
    for q in ("Q1", "Q2", "Q3", "Q4"):
        assert f"### {q}: positive" in rep, q
    assert "### Q5: not yet answered" in rep
    assert "Limits" in rep and "Invisible targets" in rep and "drone magnetometry" in rep
    for ref in __import__("re").findall(r"!\[[^\]]*\]\(([^)]+)\)", rep):
        assert (s.outputs_public / ref).exists(), ref  # every referenced figure exists


def test_restricted_only_under_restricted(run1):
    s, _ = run1
    r = s.outputs_restricted
    assert (r / "ranking.csv").exists() and (r / "briefing.md").exists()
    assert "verification" in (r / "briefing.md").read_text()
    restricted_names = {p.name for p in r.rglob("*") if p.is_file()}
    assert {"candidate_confidence_REG.csv", "candidates_REG.gpkg"} <= restricted_names
    for p in s.outputs_public.rglob("*"):
        assert p.name not in restricted_names, p  # nothing restricted leaks into public
        assert p.suffix != ".gpkg" and "candidate_sheets" not in p.parts, p


def test_check_public_passes_and_catches_leaks(run1, world):
    s, _ = run1
    _, truth = world
    assert invoke("report", "check-public").exit_code == 0
    from pyproj import Transformer

    gx, gy = truth.decoy_ground
    lon, lat = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True).transform(gx, gy)
    leak = s.outputs_public / "leak_probe.md"
    leak.write_text(f"candidate at {lat:.5f}, {lon:.5f}\n")
    try:
        r = invoke("report", "check-public")
        assert r.exit_code == 6
        assert f"{lat:.5f}" not in r.output  # values are never echoed
    finally:
        leak.unlink()
    assert invoke("report", "check-public").exit_code == 0


def test_second_run_is_incremental(run1):
    _s, _ = run1
    r = invoke("run-all", "--same-era", "--include-restricted", "--confirm-restricted")
    assert r.exit_code == 0, r.output
    assert "[georef-batch] up-to-date" in r.output
    assert "[detect-texture] up-to-date" in r.output
