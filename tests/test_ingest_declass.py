import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from sbf.catalog import FixtureClient, M2MError, refine_availability
from sbf.catalog.download import choose_product, download_options
from sbf.core.provenance import read_prov
from sbf.ingest.declass import (
    COST_PER_FRAME_USD,
    TransferError,
    build_order_list,
    download_browse,
    download_file,
    download_free_frames,
    extract_archive,
    select_free_frames,
    write_order_list,
)
from sbf.ingest.declass import cli as declass_cli
from sbf.ingest.declass.cli import declass_app

FX = Path(__file__).parent / "fixtures" / "ingest_declass"


def load(name):
    return json.loads((FX / name).read_text())


def frame(eid, feats, date, res="<=3m", avail="scanned", url="https://example.invalid/b.jpg"):
    return {
        "entity_id": eid,
        "dataset": "test_declass_1",
        "mission": "m",
        "camera": "c",
        "acquisition_date": date,
        "cloud_cover": None,
        "availability": avail,
        "browse_url": url,
        "footprint_wkt": "",
        "intersecting_features": feats,
        "resolution_class": res,
        "aoi_id": "REG",
        "raw_file": "f.json",
    }


class Resp:
    def __init__(self, status, body=b"", headers=None, fail_after=None):
        self.status_code, self.body, self.headers = status, body, headers or {}
        self.fail_after = fail_after

    def iter_content(self, n):
        for i in range(0, len(self.body), 7):  # small chunks
            if self.fail_after is not None and i >= self.fail_after:
                raise ConnectionError("dropped")
            yield self.body[i : i + 7]

    def close(self):
        pass


class FakeSession:
    """Serves {url: bytes}; honours Range; `fail_once` {url: n} drops the stream after n bytes."""

    def __init__(self, files, fail_once=None, ignore_range=False):
        self.files, self.fail_once, self.ignore_range = files, dict(fail_once or {}), ignore_range
        self.calls = []

    def get(self, url, headers=None, stream=True, timeout=None):
        self.calls.append((url, dict(headers or {})))
        if url not in self.files:
            return Resp(404)
        data = self.files[url]
        rng = (headers or {}).get("Range")
        start = int(rng.split("=")[1].rstrip("-")) if rng and not self.ignore_range else 0
        if start >= len(data) and rng:
            return Resp(416)
        body = data[start:]
        fa = self.fail_once.pop(url, None)
        h = {"Content-Length": str(len(body))}
        if start:
            h["Content-Range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
            return Resp(206, body, h, fa)
        return Resp(200, body, h, fa)


def tgz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for n, b in files.items():
            ti = tarfile.TarInfo(n)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    return buf.getvalue()


# ------------------------------------------------------------------ transfer


def test_download_resume_after_interruption(tmp_path):
    data = bytes(range(256)) * 4
    url = "https://example.invalid/x.bin"
    s = FakeSession({url: data}, fail_once={url: 140})
    dest = tmp_path / "e" / "x.bin"
    with pytest.raises(TransferError, match="resume"):
        download_file(s, url, dest, expected_size=len(data))
    part = dest.with_name("x.bin.part")
    assert part.exists() and not dest.exists() and 0 < part.stat().st_size < len(data)
    got = part.stat().st_size
    info = download_file(s, url, dest, expected_size=len(data))
    assert dest.read_bytes() == data and not part.exists() and info["verified"] == "size"
    assert s.calls[-1][1]["Range"] == f"bytes={got}-"


def test_download_server_ignores_range_restarts(tmp_path):
    data = b"abcdefghij" * 10
    url = "https://example.invalid/y"
    dest = tmp_path / "y.bin"
    dest.with_name("y.bin.part").write_bytes(b"junk")
    download_file(FakeSession({url: data}, ignore_range=True), url, dest, expected_size=len(data))
    assert dest.read_bytes() == data


def test_download_checksum_and_size_verification(tmp_path):
    data = b"hello world" * 5
    url = "https://example.invalid/z"
    s = FakeSession({url: data})
    ok = hashlib.sha256(data).hexdigest()
    assert (
        download_file(s, url, tmp_path / "a", checksum=ok, expected_size=len(data))["verified"]
        == "checksum"
    )
    with pytest.raises(TransferError, match="checksum mismatch"):
        download_file(s, url, tmp_path / "b", checksum="0" * 64)
    assert not (tmp_path / "b").exists() and not (tmp_path / "b.part").exists()
    with pytest.raises(TransferError, match="size mismatch"):
        download_file(s, url, tmp_path / "c", expected_size=len(data) - 1)
    assert not (tmp_path / "c").exists()


def test_download_http_error_and_bad_scheme(tmp_path):
    with pytest.raises(TransferError, match="HTTP 404"):
        download_file(FakeSession({}), "https://example.invalid/none", tmp_path / "n")
    with pytest.raises(TransferError, match="scheme"):
        download_file(FakeSession({}), "file:///etc/passwd", tmp_path / "n")
    assert not (tmp_path / "n").exists()


def test_extract_tgz_and_zip_ok(tmp_path):
    t = tmp_path / "a.tgz"
    t.write_bytes(tgz({"sub/a.tif": b"AAA", "b.txt": b"B"}))
    assert sorted(extract_archive(t, tmp_path / "o1")) == ["b.txt", "sub/a.tif"]
    assert (tmp_path / "o1" / "sub" / "a.tif").read_bytes() == b"AAA"
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("d/x.tif", b"X")
    assert extract_archive(z, tmp_path / "o2") == ["d/x.tif"]


@pytest.mark.parametrize("evil", ["../escape.txt", "/abs.txt", "ok/../../escape.txt"])
def test_extract_rejects_path_traversal(tmp_path, evil):
    t = tmp_path / "bad.tgz"
    t.write_bytes(tgz({"fine.txt": b"1", evil: b"2"}))
    out = tmp_path / "out"
    with pytest.raises(TransferError, match="unsafe"):
        extract_archive(t, out)
    assert not (tmp_path / "escape.txt").exists()
    assert not (out / "fine.txt").exists()  # validated before anything is written
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(evil, b"2")
    with pytest.raises(TransferError, match="unsafe"):
        extract_archive(z, tmp_path / "out2")


def test_extract_rejects_symlink(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        ti = tarfile.TarInfo("link")
        ti.type, ti.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tf.addfile(ti)
    t = tmp_path / "s.tar"
    t.write_bytes(buf.getvalue())
    with pytest.raises(TransferError, match="not a plain file"):
        extract_archive(t, tmp_path / "o")


# ------------------------------------------------------------------ catalog download helpers


def test_options_choose_highest_available_non_browse():
    client = FixtureClient({"download-options": [load("download-options.json")]})
    prods = download_options(client, "test_declass_1", ["TEST_FIXTURE_0001", "TEST_FIXTURE_0004"])
    mine = [p for p in prods if p.entity_id == "TEST_FIXTURE_0001"]
    assert choose_product(mine).product_id == "P_FULL"  # not the (bigger) browse product
    assert choose_product([p for p in prods if p.entity_id == "TEST_FIXTURE_0004"]) is None
    assert client.calls[0][1] == {
        "datasetName": "test_declass_1",
        "entityIds": ["TEST_FIXTURE_0001", "TEST_FIXTURE_0004"],
    }


def test_refine_availability_overrides_guess_and_keeps_fallback():
    cov = pd.DataFrame(
        [
            frame("TEST_FIXTURE_0001", "GZ1", "1962-01-01", avail="needs_order"),  # guess wrong
            frame("TEST_FIXTURE_0004", "GZ1", "1963-01-01", avail="scanned"),  # guess wrong
            frame("TEST_FIXTURE_0009", "GZ1", "1963-01-01", avail="unknown"),  # not in options
        ]
    )
    client = FixtureClient({"download-options": [load("download-options.json")]})
    new, fails = refine_availability(client, cov)
    assert not fails
    got = new.set_index("entity_id")
    assert got.loc["TEST_FIXTURE_0001", "availability"] == "scanned"
    assert got.loc["TEST_FIXTURE_0004", "availability"] == "needs_order"
    assert got.loc["TEST_FIXTURE_0009", "availability"] == "unknown"
    assert got.loc["TEST_FIXTURE_0009", "availability_source"] == "scene-search"
    assert got.loc["TEST_FIXTURE_0001", "availability_source"] == "download-options"


def test_refine_availability_records_failure():
    def boom(endpoint, payload):
        return {"data": None, "errorCode": "RATE_LIMIT", "errorMessage": "slow down"}

    cov = pd.DataFrame([frame("E1", "GZ1", "1962-01-01", avail="needs_order")])
    new, fails = refine_availability(FixtureClient(boom), cov)
    assert fails[0]["endpoint"] == "download-options" and new["availability"][0] == "needs_order"


# ------------------------------------------------------------------ browse


def test_browse_downloads_all_rows_and_records_failures(tmp_path):
    cov = pd.DataFrame(
        [
            frame("TEST_FIXTURE_0001", "GZ1", "1962-01-01", url="https://example.invalid/a.png"),
            frame("TEST_FIXTURE_0004", "", "1963-01-01", avail="needs_order"),
            frame("TEST_FIXTURE_0005", "", "1963-01-01", url="https://example.invalid/missing"),
            frame("TEST_FIXTURE_0006", "", "1963-01-01", url=""),
            frame("../evil", "", "1963-01-01"),
        ]
    )
    s = FakeSession(
        {"https://example.invalid/a.png": b"PNG", "https://example.invalid/b.jpg": b"JPG"}
    )
    rep = download_browse(cov, tmp_path, s)
    assert sorted(rep.done) == ["TEST_FIXTURE_0001", "TEST_FIXTURE_0004"]
    assert (tmp_path / "TEST_FIXTURE_0001" / "browse.png").read_bytes() == b"PNG"
    assert (tmp_path / "TEST_FIXTURE_0004" / "browse.jpg").exists()
    assert (
        read_prov(tmp_path / "TEST_FIXTURE_0001" / "browse.png")["source_id"] == "TEST_FIXTURE_0001"
    )
    assert {f.entity_id for f in rep.failures} == {
        "TEST_FIXTURE_0005",
        "TEST_FIXTURE_0006",
        "../evil",
    }
    assert not (tmp_path / "TEST_FIXTURE_0005" / "browse.jpg").exists()
    assert download_browse(cov, tmp_path, s).skipped  # idempotent


# ------------------------------------------------------------------ free frames


def _dl_client(preparing_first=False):
    seen = {"retrieve": 0}

    def responder(endpoint, payload):
        if endpoint == "download-options":
            return load("download-options.json")
        if endpoint == "download-request":
            env = load("download-request.json")
            if preparing_first:
                env["data"]["preparingDownloads"] = env["data"].pop("availableDownloads")
                env["data"]["availableDownloads"] = []
            return env
        if endpoint == "download-retrieve":
            seen["retrieve"] += 1
            env = load("download-request.json")
            return {"data": {"available": env["data"]["availableDownloads"]}, "errorCode": None}
        raise AssertionError(endpoint)

    return FixtureClient(responder), seen


def test_select_free_frames():
    cov = pd.DataFrame(
        [
            frame("A", "GZ1;GZ2", "1962-01-01"),
            frame("B", "GZ2", "1962-01-01"),
            frame("C", "GZ1", "1962-01-01", avail="needs_order"),
            frame("D", "", "1962-01-01"),
        ]
    )
    assert list(select_free_frames(cov)["entity_id"]) == ["A", "B"]
    assert list(select_free_frames(cov, ["GZ1"])["entity_id"]) == ["A"]


def test_download_free_frames_end_to_end(tmp_path):
    archive = tgz({"scan/TEST_FIXTURE_0001_a.tif": b"IMG-A", "scan/b.tif": b"IMG-B"})
    pad = archive + b"\0" * 0
    opts = load("download-options.json")
    opts["data"][1]["filesize"] = len(pad)
    client, _ = _dl_client(preparing_first=True)
    client.responses = lambda e, p: (
        opts if e == "download-options" else _dl_client(True)[0].responses(e, p)
    )
    url = "https://example.invalid/dl/TEST_FIXTURE_0001.tgz"
    cov = pd.DataFrame([frame("TEST_FIXTURE_0001", "GZ1", "1962-05-10")])
    waits = []
    rep = download_free_frames(
        client, FakeSession({url: pad}), cov, tmp_path, sleep=waits.append, poll_seconds=0
    )
    assert rep.done == ["TEST_FIXTURE_0001"] and not rep.failures
    d = tmp_path / "TEST_FIXTURE_0001"
    f = d / "TEST_FIXTURE_0001.tgz"
    assert f.read_bytes() == pad and (d / "scan" / "b.tif").read_bytes() == b"IMG-B"
    prov = read_prov(f)
    assert prov["source"] == "test_declass_1" and prov["source_id"] == "TEST_FIXTURE_0001"
    assert prov["acquisition_date"] == "1962-05-10"
    assert prov["processing_steps"][0] == "m2m download Full resolution tgz"
    assert any("extracted" in s for s in prov["processing_steps"]) and len(waits) == 1
    # requested the full product, not the browse one
    req = next(p for e, p in client.calls if e == "download-request")
    assert req["downloads"] == [{"entityId": "TEST_FIXTURE_0001", "productId": "P_FULL"}]
    assert not any(e in {"order", "order-submit"} for e, _ in client.calls)
    rep2 = download_free_frames(client, FakeSession({url: pad}), cov, tmp_path, sleep=waits.append)
    assert rep2.skipped == ["TEST_FIXTURE_0001"]


def test_download_free_frames_records_failures_without_faking(tmp_path):
    cov = pd.DataFrame(
        [
            frame("TEST_FIXTURE_0001", "GZ1", "1962-05-10"),
            frame("TEST_FIXTURE_0004", "GZ1", "1963-06-01"),  # no available product
        ]
    )
    client, _ = _dl_client()
    rep = download_free_frames(client, FakeSession({}), cov, tmp_path)  # 404 on download
    assert {f.entity_id for f in rep.failures} == {"TEST_FIXTURE_0001", "TEST_FIXTURE_0004"}
    assert not rep.done
    assert not list(tmp_path.rglob("*.tgz")) and not list(tmp_path.rglob("*.prov.json"))


def test_download_free_frames_options_error(tmp_path):
    def boom(endpoint, payload):
        return {"data": None, "errorCode": "AUTH", "errorMessage": "no"}

    cov = pd.DataFrame([frame("E1", "GZ1", "1962-01-01")])
    rep = download_free_frames(FixtureClient(boom), FakeSession({}), cov, tmp_path)
    assert rep.failures[0].stage == "download-options"
    with pytest.raises(M2MError):
        FixtureClient(boom).call("download-options", {})


# ------------------------------------------------------------------ order list


def test_order_list_ranking_and_cost():
    cov = pd.DataFrame(
        [
            frame("coarse_pre", "GZ1", "1962-01-01", res="3-10m", avail="needs_order"),
            frame("hires_post", "GZ1", "1975-01-01", avail="needs_order"),
            frame("hires_pre_multi", "GZ1;GZ2", "1964-01-01", avail="needs_order"),
            frame("hires_bracket", "GZ1", "1967-06-01", avail="needs_order"),
            frame("free", "GZ1", "1964-01-01", avail="scanned"),
            frame("untagged", "", "1964-01-01", avail="needs_order"),
            frame("undated_arg", "GZ1", None, res=">10m", avail="needs_order"),
        ]
    )
    df = build_order_list(cov)
    assert list(df["entity_id"]) == [
        "hires_pre_multi",  # 3 + 3 + 2 = 8
        "coarse_pre",  # 2 + 3 + 1 = 6 (tie: earlier date first)
        "hires_bracket",  # 3 + 2 + 1 = 6
        "hires_post",  # 3 + 1 + 1 = 5
        "undated_arg",  # 0 + 0 + 1
    ]
    assert list(df["score"]) == [8, 6, 6, 5, 1]
    assert list(df["cumulative_cost_usd"]) == [30, 60, 90, 120, 150]
    assert (df["estimated_cost_usd"] == COST_PER_FRAME_USD).all() and COST_PER_FRAME_USD == 30
    assert df["rank"].tolist() == [1, 2, 3, 4, 5]


def test_order_list_empty(tmp_path):
    df = build_order_list(pd.DataFrame([frame("x", "GZ1", "1962-01-01")]))
    assert df.empty and write_order_list(df, tmp_path / "o.csv") == 0


# ------------------------------------------------------------------ CLI


def _write_cov(settings, rows):
    pd.DataFrame(rows).to_csv(settings.outputs_public / "coverage.csv", index=False)


def test_cli_order_list_never_orders(tmp_project):
    _write_cov(
        tmp_project,
        [
            frame("A", "GZ1", "1962-01-01", avail="needs_order"),
            frame("B", "GZ2", "1970-01-01", avail="needs_order"),
        ],
    )
    res = CliRunner().invoke(declass_app, ["order-list"])
    assert res.exit_code == 0 and "$60" in res.output and "NO ORDERS PLACED" in res.output
    out = pd.read_csv(tmp_project.outputs_public / "order_list.csv")
    assert out["entity_id"].tolist() == ["A", "B"] and out["cumulative_cost_usd"].tolist() == [
        30,
        60,
    ]


def test_cli_requires_coverage_and_credentials(tmp_project, monkeypatch):
    r = CliRunner()
    assert r.invoke(declass_app, ["order-list"]).exit_code == 2
    _write_cov(tmp_project, [frame("A", "GZ1", "1962-01-01")])
    monkeypatch.delenv("USGS_USERNAME", raising=False)
    monkeypatch.delenv("USGS_M2M_TOKEN", raising=False)
    assert r.invoke(declass_app, ["download"]).exit_code == 2
    assert r.invoke(declass_app, ["availability"]).exit_code == 2
    assert r.invoke(declass_app, ["download", "--feature", "NOPE", "--dry-run"]).exit_code == 2


def test_cli_download_dry_run_no_network(tmp_project, monkeypatch):
    monkeypatch.setattr(declass_cli, "_session", lambda: pytest.fail("no network on dry run"))
    _write_cov(
        tmp_project,
        [
            frame("A", "GZ1", "1962-01-01"),
            frame("B", "GZ2", "1962-01-01"),
            frame("C", "GZ1", "1962-01-01", avail="needs_order"),
        ],
    )
    res = CliRunner().invoke(declass_app, ["download", "--dry-run", "--feature", "GZ1"])
    assert res.exit_code == 0 and "1 free frame" in res.output and "A\t" in res.output
    assert not (tmp_project.data_raw / "declass").exists()


def test_cli_browse_and_download_with_fakes(tmp_project, monkeypatch):
    url = "https://example.invalid/dl/TEST_FIXTURE_0001.tgz"
    data = tgz({"x.tif": b"X"})
    opts = load("download-options.json")
    opts["data"][1]["filesize"] = len(data)
    client = FixtureClient(
        lambda e, p: opts if e == "download-options" else load("download-request.json")
    )
    sess = FakeSession({url: data, "https://example.invalid/b.jpg": b"J"})
    monkeypatch.setattr(declass_cli, "_session", lambda: sess)
    monkeypatch.setattr(declass_cli, "_client", lambda s: client)
    _write_cov(tmp_project, [frame("TEST_FIXTURE_0001", "GZ1", "1962-05-10")])
    r = CliRunner()
    assert r.invoke(declass_app, ["browse"]).exit_code == 0
    assert (tmp_project.data_raw / "declass" / "TEST_FIXTURE_0001" / "browse.jpg").exists()
    res = r.invoke(declass_app, ["download", "--feature", "GZ1"])
    assert res.exit_code == 0, res.output
    d = tmp_project.data_raw / "declass" / "TEST_FIXTURE_0001"
    assert (d / "x.tif").read_bytes() == b"X" and (d / "TEST_FIXTURE_0001.tgz.prov.json").exists()
    assert r.invoke(declass_app, ["availability"]).exit_code == 0
    cov = pd.read_csv(tmp_project.outputs_public / "coverage.csv")
    assert cov["availability_source"][0] == "download-options"
    sess.files.pop(url)
    (d / "TEST_FIXTURE_0001.tgz").unlink()
    (d / "TEST_FIXTURE_0001.tgz.prov.json").unlink()
    res = r.invoke(declass_app, ["download"])
    assert res.exit_code == 1
    assert json.loads((tmp_project.data_interim / "declass" / "download_failures.json").read_text())
