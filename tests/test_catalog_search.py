import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from sbf.catalog import (
    FixtureClient,
    M2MClient,
    M2MError,
    RawStore,
    discover_datasets,
    resolution_class,
    run_search,
)
from sbf.catalog.search import COVERAGE_COLUMNS, _meta, scene_to_row
from sbf.core.aoi import load_aois, load_reference_features

FX = Path(__file__).parent / "fixtures" / "catalog"


def load(name):
    return json.loads((FX / name).read_text())


def responder(endpoint, payload):
    if endpoint == "dataset-search":
        return load("dataset-search.json")
    env = load(f"scene-search_{payload['datasetName']}.json")
    gj = payload["sceneFilter"]["spatialFilter"]["geoJson"]
    xs = [p[0] for p in gj["coordinates"][0]]
    ys = [p[1] for p in gj["coordinates"][0]]
    keep = []
    for r in env["data"]["results"]:
        c = r["spatialCoverage"]["coordinates"][0]
        if (
            min(p[0] for p in c) <= max(xs)
            and max(p[0] for p in c) >= min(xs)
            and min(p[1] for p in c) <= max(ys)
            and max(p[1] for p in c) >= min(ys)
        ):
            keep.append(r)
    env["data"]["results"] = keep
    return env


@pytest.mark.parametrize(
    "text,cls",
    [
        ("KH-4B", "<=3m"),
        ("kh-4a", "<=3m"),
        ("KH-4", "3-10m"),
        ("CORONA", "3-10m"),
        ("KH-7", "<=3m"),
        ("KH-9 mapping", "3-10m"),
        ("KH-9 panoramic", "<=3m"),
        ("ARGON", ">10m"),
        ("LANYARD", "<=3m"),
        ("KH-9", "unknown"),
        ("something else", "unknown"),
        (None, "unknown"),
    ],
)
def test_resolution_class(text, cls):
    assert resolution_class(text) == cls


def test_discover_keeps_only_declass_123():
    names = [
        d["dataset"]
        for d in discover_datasets(FixtureClient({"dataset-search": [load("dataset-search.json")]}))
    ]
    assert names == ["test_declass_1", "test_declass_2", "test_declass_3"]


def test_run_search_end_to_end(tmp_project):
    store = RawStore(tmp_project.data_raw / "catalog")
    client = FixtureClient(responder, store)
    res = run_search(
        client, load_aois(tmp_project), load_reference_features(tmp_project), ["REG", "INE"]
    )
    cov = res.coverage.set_index("entity_id")
    assert list(res.coverage.columns) == COVERAGE_COLUMNS
    assert set(cov.index) == {f"TEST_FIXTURE_000{i}" for i in (1, 2, 3, 4, 5)}
    assert not res.failures
    r1 = cov.loc["TEST_FIXTURE_0001"]
    assert (r1.dataset, r1.mission, r1.availability, r1.resolution_class) == (
        "test_declass_1",
        "KH-4B",
        "scanned",
        "<=3m",
    )
    assert r1.acquisition_date == "1962-05-10" and r1.cloud_cover == 5
    assert set(r1.intersecting_features.split(";")) == {"GZ1", "GZ2", "GZ3", "GZ4"}
    assert cov.loc["TEST_FIXTURE_0002"].availability == "needs_order"
    assert pd.isna(cov.loc["TEST_FIXTURE_0003"].cloud_cover)
    assert cov.loc["TEST_FIXTURE_0003"].resolution_class == "3-10m"
    assert cov.loc["TEST_FIXTURE_0004"].resolution_class == ">10m"
    assert cov.loc["TEST_FIXTURE_0004"].intersecting_features == "TA1"
    # traceability: every row names a saved raw file that contains its entity
    for eid, row in cov.iterrows():
        raw = json.loads((store.dir / row.raw_file).read_text())
        assert eid in [r["entityId"] for r in raw["data"]["results"]]


def test_failures_recorded_not_filled(tmp_project):
    def flaky(endpoint, payload):
        if endpoint == "scene-search" and payload["datasetName"] == "test_declass_2":
            return load("error_envelope.json")
        return responder(endpoint, payload)

    store = RawStore(tmp_project.data_raw / "catalog")
    res = run_search(
        FixtureClient(flaky, store),
        load_aois(tmp_project),
        load_reference_features(tmp_project),
        ["REG"],
    )
    assert "TEST_FIXTURE_0002" not in set(res.coverage.entity_id)
    assert len(res.failures) == 1 and "AUTH_INVALID" in res.failures[0].message
    assert res.failures[0].raw_file and (store.dir / res.failures[0].raw_file).exists()


def test_pagination(tmp_project):
    scenes = load("scene-search_test_declass_1.json")["data"]["results"]

    def paged(endpoint, payload):
        if endpoint == "dataset-search":
            return {"data": [{"datasetAlias": "test_declass_1"}], "errorCode": None}
        n = payload["startingNumber"]
        if n == 1:
            return {"data": {"results": scenes[:1], "nextRecord": 2}, "errorCode": None}
        return {"data": {"results": scenes[1:], "nextRecord": 0}, "errorCode": None}

    res = run_search(
        FixtureClient(paged),
        load_aois(tmp_project),
        load_reference_features(tmp_project),
        ["REG", "INE"],
    )
    assert set(res.coverage.entity_id) == {"TEST_FIXTURE_0001", "TEST_FIXTURE_0004"}


def test_missing_fields_stay_missing():
    row = scene_to_row(
        {"entityId": "TEST_FIXTURE_X"}, dataset="test_declass_1", aoi_id="REG", raw_file="f.json"
    )
    assert row["acquisition_date"] is None and row["cloud_cover"] is None
    assert row["availability"] == "unknown" and row["footprint_wkt"] == ""
    assert row["resolution_class"] == "unknown" and row["mission"] is None
    assert _meta({"metadata": []}, ["Mission"]) is None


def test_raw_store_verbatim_and_login_not_saved(tmp_path):
    store = RawStore(tmp_path)
    env = {"requestId": 1, "data": "SECRET", "errorCode": None}
    assert store.save("login-token", {"username": "u", "token": "t"}, env) is None
    assert not list(tmp_path.glob("login-token*"))
    name = store.save("dataset-search", {"datasetName": "x"}, {"a": 1})
    assert json.loads((tmp_path / name).read_text()) == {"a": 1}
    assert store.save("dataset-search", {"datasetName": "x"}, {"a": 1}) == name  # deterministic


class FakeResp:
    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, json, headers))
        if url.endswith("login-token"):
            return FakeResp({"data": "APIKEY", "errorCode": None})
        return FakeResp({"data": [], "errorCode": None})


def test_m2m_client_login_and_headers(tmp_path, monkeypatch):
    monkeypatch.setenv("USGS_USERNAME", "u")
    monkeypatch.setenv("USGS_M2M_TOKEN", "t")
    sess = FakeSession()
    c = M2MClient(RawStore(tmp_path), session=sess)
    c.call("dataset-search", {"datasetName": "declass"})
    assert sess.calls[0][0].endswith("/stable/login-token") and sess.calls[0][1] == {
        "username": "u",
        "token": "t",
    }
    assert sess.calls[1][2] == {"X-Auth-Token": "APIKEY"}
    assert c.last_raw_file.startswith("dataset-search_")
    assert "APIKEY" not in "".join(p.read_text() for p in tmp_path.iterdir())


def test_m2m_requires_env_and_wraps_transport_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("USGS_USERNAME", raising=False)
    monkeypatch.delenv("USGS_M2M_TOKEN", raising=False)
    with pytest.raises(M2MError):
        M2MClient.credentials()

    class Boom:
        def post(self, *a, **k):
            raise requests.ConnectionError("nope")

    monkeypatch.setenv("USGS_USERNAME", "u")
    monkeypatch.setenv("USGS_M2M_TOKEN", "t")
    with pytest.raises(M2MError, match="transport failure"):
        M2MClient(RawStore(tmp_path), session=Boom()).call("dataset-search", {})


def test_declass_series_from_live_titles_and_aliases():
    # Real dataset-search titles/aliases observed 2026-11-21.
    from sbf.catalog.search import _SERIES, DECLASS_PATTERN

    cases = {
        "corona2 Declass 1 = Corona, Lanyard, & Argon Missions": "1",
        "declassii Declass 2 = KH-7 and KH-9 Global Camera Photos": "2",
        "declassiii Declass 3 = USGS Subset of Hexagon Missions": "3",
        "declassiii": "3",
    }
    for text, want in cases.items():
        g = DECLASS_PATTERN.search(text).group(1).lower()
        assert _SERIES.get(g, g) == want, text


def test_resolution_class_from_live_camera_resolution_values():
    # Values observed in the first live scene-search (2026-11-21).
    from sbf.catalog.resolution import resolution_class

    assert resolution_class("Stereo High", "Aft", "1107-2", "corona2") == "<=3m"
    assert resolution_class("Stereo Medium", "Aft", "1049-1", "corona2") == "<=3m"
    assert resolution_class("Vertical Low", "Cartographic", "9058A", "corona2") == ">10m"
    assert resolution_class("2 to 4 feet", "F", "1214-3", "declassiii") == "<=3m"
    assert (
        resolution_class(None, "KH-7 High Resolution Surveillance", "4001", "declassii") == "<=3m"
    )
    assert resolution_class(None, "KH-9 Lower Resolution Mapping", "1205", "declassii") == "3-10m"


def test_kh4_mural_mission_overrides_stereo_medium_label():
    from sbf.catalog.resolution import resolution_class

    assert resolution_class("Stereo Medium", "Aft", "9062", "corona2") == "3-10m"
    assert resolution_class("Stereo Medium", "Aft", "1042-2", "corona2") == "<=3m"
    assert resolution_class("Vertical Low", "Cartographic", "9062A", "corona2") == ">10m"
