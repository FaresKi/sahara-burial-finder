import csv
import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from sbf.docs_mining.cli import app
from sbf.docs_mining.extract import (
    ClaudeExtractor,
    ExtractionError,
    FakeExtractor,
    estimate_cost,
    run_extraction,
)
from sbf.docs_mining.pdf import chunk_pages, ingest_docs, load_chunks, make_doc_id, slugify
from sbf.docs_mining.review import compute_precision, export_review, write_precision_report
from sbf.docs_mining.schema import (
    ClaimDraft,
    SpatialClaim,
    describes_reference_feature,
)

QUOTE = "Equipment was buried 10 km north-east of Anchor A."


def draft(quote=QUOTE, anchor="Anchor A", **kw):
    base = {
        "quote": quote,
        "object_buried": "unknown",
        "anchor": anchor,
        "distance_km": 10.0,
        "distance_uncertainty_km": None,
        "bearing": "NE",
        "bearing_uncertainty_deg": None,
        "time_period": None,
        "confidence_source": "secondary",
        "notes": None,
    }
    base.update(kw)
    return ClaimDraft(**base)


@pytest.fixture
def pdf_project(tmp_project):
    import pymupdf

    d = tmp_project.data_raw / "docs"
    d.mkdir()
    doc = pymupdf.open()
    for text in ["Synthetic page one. Nothing here.", f"Test document. {QUOTE}"]:
        doc.new_page().insert_text((72, 72), text)
    doc.save(d / "Test Doc (Synthetic).pdf")
    return tmp_project


def test_quote_validator():
    with pytest.raises(ValueError):
        draft(quote=" ".join(["w"] * 26))
    draft(quote=" ".join(["w"] * 25))
    with pytest.raises(ValueError):
        SpatialClaim(
            extraction_id="x",
            doc_id="d",
            page=1,
            quote="a " * 30,
            object_buried="unknown",
            anchor="a",
            confidence_source="official",
            model="m",
            prompt_version="v",
        )


def test_enums_strict():
    with pytest.raises(ValueError):
        draft(object_buried="tanks")
    with pytest.raises(ValueError):
        draft(confidence_source="rumour")


def test_reference_flag():
    for a in ["Gerboise Bleue GZ", "Pollen site", "E2 portal", "Béryl", "GZ3", "Point zéro"]:
        assert describes_reference_feature(a), a
    assert not describes_reference_feature("Reggane plateau")


def test_slug_and_chunking():
    assert slugify("Rapport ICAN/Böll (FR).pdf") == "rapport-ican-boll-fr-pdf"
    pages = [(1, " ".join(f"w{i}" for i in range(100))), (2, ""), (3, "short")]
    ch = chunk_pages("d", pages, chunk_words=40, overlap_words=10)
    assert [c.page for c in ch] == [1, 1, 1, 3]
    assert ch[0].text.split()[-10:] == ch[1].text.split()[:10]
    assert ch[2].text.split()[-1] == "w99"
    assert len({c.chunk_hash for c in ch}) == 4


def test_ingest(pdf_project):
    recs = ingest_docs(pdf_project)
    assert len(recs) == 1 and recs[0].n_pages == 2
    assert recs[0].doc_id == make_doc_id(pdf_project.data_raw / "docs" / "Test Doc (Synthetic).pdf")
    assert recs[0].doc_id.startswith("test-doc-synthetic-")
    man = json.loads((pdf_project.data_interim / "docs" / "manifest.json").read_text())
    assert man[0]["sha256"] and man[0]["filename"].endswith(".pdf")
    assert {c.page for c in load_chunks(pdf_project)} == {1, 2}


def _setup_extraction(pdf_project, extractor):
    ingest_docs(pdf_project)
    return run_extraction(extractor, pdf_project)


def test_extraction_pipeline_and_cache(pdf_project):
    ingest_docs(pdf_project)
    ch2 = next(c for c in load_chunks(pdf_project) if c.page == 2)
    ex = FakeExtractor({ch2.chunk_hash: [draft(), draft(quote="This was never in the text.")]})
    s = run_extraction(ex, pdf_project)
    assert (s.claims, s.rejected_quotes, s.failed, s.cached) == (1, 1, 0, 0)
    rec = json.loads((pdf_project.data_interim / "docs" / "extractions.jsonl").read_text())
    assert rec["page"] == 2 and rec["model"] == "fake-model" and rec["prompt_version"] == "v1"
    assert rec["describes_reference_feature"] is False
    fails = (pdf_project.data_interim / "docs" / "failures.jsonl").read_text()
    assert "not verbatim" in fails
    # second run hits cache: no extractor calls
    ex2 = FakeExtractor()
    s2 = run_extraction(ex2, pdf_project)
    assert s2.cached == 2 and ex2.calls == []
    # different prompt version -> cache miss
    s3 = run_extraction(FakeExtractor(prompt_version="v2"), pdf_project)
    assert s3.cached == 0


def test_whitespace_normalised_quote_ok(pdf_project):
    ingest_docs(pdf_project)
    ex = FakeExtractor(
        default=[draft(quote="Equipment was  buried 10 km\nnorth-east of Anchor A.")]
    )
    s = run_extraction(ex, pdf_project, limit=None)
    assert s.claims == 1 and s.rejected_quotes == 1  # page 1 chunk rejects, page 2 accepts


def test_failure_recorded_not_guessed(pdf_project):
    ingest_docs(pdf_project)
    first = load_chunks(pdf_project)[0]
    s = run_extraction(FakeExtractor(fail_on={first.chunk_hash}), pdf_project)
    assert s.failed == 1 and s.claims == 0
    assert "fake failure" in (pdf_project.data_interim / "docs" / "failures.jsonl").read_text()


class _Client:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.texts.pop(0))])


def test_claude_extractor_retry_and_schema(pdf_project):
    ingest_docs(pdf_project)
    chunk = load_chunks(pdf_project)[1]
    good = json.dumps({"claims": [draft().model_dump()]})
    c = _Client(['{"claims": [{"quote": 3}]}', good])
    out = ClaudeExtractor(model="m", client=c).extract(chunk)
    assert len(out) == 1 and len(c.calls) == 2
    assert c.calls[0]["output_config"]["format"]["type"] == "json_schema"
    assert c.calls[0]["model"] == "m"
    bad = _Client(["nope"] * 3)
    with pytest.raises(ExtractionError):
        ClaudeExtractor(model="m", client=bad).extract(chunk)


def test_claude_extractor_needs_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ExtractionError, match="ANTHROPIC_API_KEY"):
        ClaudeExtractor().client  # noqa: B018


def test_model_env(monkeypatch):
    monkeypatch.setenv("SBF_EXTRACT_MODEL", "other-model")
    assert ClaudeExtractor().model == "other-model"
    monkeypatch.delenv("SBF_EXTRACT_MODEL")
    assert ClaudeExtractor().model == "claude-sonnet-5-5"


def test_review_roundtrip_and_precision(pdf_project):
    ingest_docs(pdf_project)
    ex = FakeExtractor(
        default=[draft(), draft(anchor="Pollen", quote="Equipment was buried 10 km")]
    )
    run_extraction(ex, pdf_project)
    path = export_review(pdf_project)
    rows = list(csv.DictReader(path.open()))
    assert "accept" in rows[0] and "reviewer_note" in rows[0] and rows[0]["accept"] == ""
    p = compute_precision(path)
    assert p.reviewed == 0 and p.precision is None and p.reference_feature_records >= 1
    # human fills in decisions; re-export must preserve them
    rows[0]["accept"] = "y"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    export_review(pdf_project)
    assert next(iter(csv.DictReader(path.open())))["accept"] == "y"
    out, p = write_precision_report(pdf_project)
    assert "insufficient" in out.read_text()


def _fake_review(path, n_acc, n_rej):
    from sbf.docs_mining.review import REVIEW_COLUMNS

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for i in range(n_acc + n_rej):
            w.writerow(
                {
                    "extraction_id": str(i),
                    "quote": "SECRET QUOTE",
                    "confidence_source": "official",
                    "accept": "yes" if i < n_acc else "n",
                    "describes_reference_feature": "False",
                }
            )


def test_precision_threshold_and_no_quotes(tmp_project):
    d = tmp_project.data_interim / "docs"
    d.mkdir(parents=True)
    _fake_review(d / "review.csv", 20, 5)
    out, p = write_precision_report(tmp_project)
    assert not p.sufficient and "insufficient" in out.read_text()
    _fake_review(d / "review.csv", 27, 9)
    out, p = write_precision_report(tmp_project)
    assert p.sufficient and p.precision == pytest.approx(0.75)
    text = out.read_text()
    assert "75.0%" in text and "SECRET" not in text
    assert out.parent == tmp_project.outputs_public


def test_estimate_and_cli(pdf_project):
    runner = CliRunner()
    r = runner.invoke(app, ["ingest"])
    assert r.exit_code == 0 and "2 pages" in r.output
    r = runner.invoke(app, ["extract", "--dry-run"])
    assert r.exit_code == 0 and "DRY RUN: 2 chunks" in r.output
    assert estimate_cost(load_chunks(pdf_project)).cost_usd > 0
    assert not (pdf_project.data_interim / "docs" / "extractions.jsonl").exists()
