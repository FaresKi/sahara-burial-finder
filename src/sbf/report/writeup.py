# ruff: noqa: ISC004
"""Phase 7 write-up: `outputs/public/report.md` and (gated) `outputs/restricted/briefing.md`.

Everything is generated from artefacts already on disk; nothing is written from memory. A question
whose artefact is missing is "not yet answered" (SPEC section 0.2/0.5: no speculation). The public
report never contains coordinates of non-reference features; `sbf report check-public` enforces it.

Answer vocabulary: ``positive`` | ``negative`` | ``not yet answered``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sbf.core.config import Settings, load_settings

POSITIVE, NEGATIVE, UNANSWERED = "positive", "negative", "not yet answered"

LIMITS = [
    "**Invisible targets.** Anything inside the tunnels, or buried deeper than about 1-2 m, may "
    "have no surface or L-band signature. This holds for every result below.",
    "**Resolution floor.** 1960s CORONA-class imagery (about 2-3 m) can show trenches and vehicle "
    "parks, not individual drums.",
    "**Label noise.** Reference geometries are approximate (some are marked approximate or "
    "to-be-located); archive-derived labels have been shown to cut accuracy substantially "
    "(about 98% to about 65% in Chornobyl work). Metrics are indicative only.",
    "**Desert film saturation and dune migration.** Bright sand clips in scans; dunes move tens "
    "of metres per decade, so dunes are masked and never used as control points.",
    "**Panoramic distortion.** Strong across-track scale change; frame-wide polynomials fail and "
    "strip-wise fits are used. Checkpoint error only measures accuracy where control points are; "
    "the spatial cross-validation RMSE is the headline number.",
    "**Clutter.** Army presence since 1966, scavenging and recent cleanup works create many real "
    "disturbances that are not burials; dating a disturbance is essential.",
    "**Small samples.** A handful of reference features means recall moves in large steps.",
]

RECOMMENDATIONS = [
    "Any candidate area is a hypothesis from desk analysis. It requires ground verification by "
    "competent, authorised Algerian teams; this project does not plan, route or advise any site "
    "visit (the sites are restricted military zones and may be contaminated).",
    "Suitable non-invasive verification methods for authorised teams to consider: drone "
    "magnetometry (flight height about 2-3 m above ground), gamma-ray screening and "
    "ground-penetrating radar, performed under the radiation-protection procedures of the "
    "responsible authority.",
    "Share restricted candidate material only with the rehabilitation agency and partner "
    "organisations through the restricted briefing, never through public channels.",
]

VERIFICATION_STATEMENTS = [
    "Every entry is a candidate area for verification, not a finding of waste.",
    "Each candidate requires ground verification by competent authorities.",
    "Confidence levels reflect desk evidence and the Phase 4 validation score only.",
    "This analysis cannot see targets inside tunnels or buried deeper than about 1-2 m.",
    "No fieldwork, access or site-visit instructions are given or implied.",
]


@dataclass
class Answer:
    qid: str
    question: str
    status: str
    text: str
    artefacts: list[str] = field(default_factory=list)


def _read_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _demote(md: str, by: int = 2) -> str:
    """Push markdown headings down so an embedded artefact nests under the report sections."""
    return re.sub(r"^(#+) ", lambda m: "#" * (len(m.group(1)) + by) + " ", md, flags=re.MULTILINE)


def _rel(s: Settings, p: Path) -> str:
    return str(p.relative_to(s.root)) if p.is_relative_to(s.root) else str(p)


@dataclass
class Evidence:
    """Everything the report reads, loaded once."""

    s: Settings
    go: dict | None
    coverage: pd.DataFrame | None
    registry: pd.DataFrame | None
    gate: dict | None
    prior_summary: dict | None
    texts: dict[str, str]

    @classmethod
    def load(cls, s: Settings) -> Evidence:
        pub = s.outputs_public
        cov = None
        if (pub / "coverage.csv").exists():
            from sbf.catalog.report import read_coverage

            cov = read_coverage(pub / "coverage.csv")
        reg = None
        rp = s.data_processed / "historical" / "registry.parquet"
        if rp.exists():
            from sbf.georef.registry import load_registry

            reg = load_registry(included_only=False, settings=s)
        texts = {}
        for name in (
            "coverage_summary.md",
            "georef_summary.md",
            "blind_test.md",
            "docs_extraction_precision.md",
            "prior_eval.md",
            "candidates_public_stats.md",
            "fusion_weights.md",
            "fusion_summary.md",
        ):
            if (pub / name).exists():
                texts[name] = (pub / name).read_text()
        return cls(
            s,
            _read_json(pub / "go_decision.json"),
            cov,
            reg,
            _read_json(pub / "blind_test_gate.json"),
            _read_json(pub / "prior_eval_summary.json"),
            texts,
        )


# ------------------------------------------------------------------ answers


def answer_q1(ev: Evidence) -> Answer:
    q = "Does declassified imagery cover the AOIs, at which dates and resolutions?"
    arts = ["outputs/public/coverage.csv", "outputs/public/coverage_summary.md"]
    if ev.coverage is None or ev.go is None:
        return Answer(
            "Q1", q, UNANSWERED, "No coverage catalogue or go decision was produced.", arts
        )
    cov = ev.coverage
    dates = pd.to_datetime(cov["acquisition_date"], errors="coerce").dropna()
    span = f"{dates.min().date()} to {dates.max().date()}" if len(dates) else "undated"
    facts = (
        f"{len(cov)} catalogue scenes over the AOIs (acquisition dates {span}); resolution "
        f"classes: {_counts(cov['resolution_class'])}."
    )
    status = ev.go.get("status")
    why = " ".join(ev.go.get("reasons", []))
    if status == "GO":
        return Answer("Q1", q, POSITIVE, f"{facts} Phase 1 go rule met. {why}", arts)
    if status == "NEEDS_REVIEW":
        return Answer(
            "Q1",
            q,
            UNANSWERED,
            f"{facts} The go rule depends on frames with unknown cloud cover that need a human "
            f"browse check. {why}",
            arts,
        )
    return Answer("Q1", q, NEGATIVE, f"{facts} Phase 1 go rule NOT met. {why}", arts)


def _counts(series: pd.Series) -> str:
    vc = series.fillna("unknown").value_counts()
    return ", ".join(f"{k}: {v}" for k, v in vc.items()) or "none"


def answer_q2(ev: Evidence) -> Answer:
    q = "Are the known features visible in historical and modern imagery?"
    arts = ["outputs/public/georef/", "outputs/public/blind_test.md"]
    det = ev.s.outputs_public / "detectability.csv"
    if det.exists():
        d = pd.read_csv(det)
        vis = d["verdict"].astype(str).str.lower()
        n_vis = int(vis.isin({"visible", "ambiguous"}).sum())
        status = POSITIVE if n_vis else NEGATIVE
        return Answer(
            "Q2",
            q,
            status,
            f"Human detectability table: {_counts(d['verdict'])} ({len(d)} feature/layer rows).",
            ["outputs/public/detectability.csv"],
        )
    if ev.gate and ev.gate.get("status") == "PASS":
        best = (ev.gate.get("best_rows") or [{}])[0]
        return Answer(
            "Q2",
            q,
            POSITIVE,
            "Indicative, from detector evidence only: reference-feature tiles are separable from "
            f"background in a historical layer (`{best.get('detector')}` / `{best.get('layer')}`). "
            "Visual detectability per layer still needs the human check "
            "(`notebooks/02_modern_detectability.ipynb`).",
            arts,
        )
    return Answer(
        "Q2",
        q,
        UNANSWERED,
        "A failed or missing detector test is not evidence of invisibility. The visual "
        "detectability table (human judgement) has not been recorded.",
        arts,
    )


def _reg_stats(reg: pd.DataFrame) -> dict:
    ok = reg[reg["status"].eq("ok") & reg["cv_rmse_m"].notna()]
    inc = reg[reg["included_for_change_detection"]]
    return {
        "n": len(reg),
        "n_ok": len(ok),
        "n_inc": len(inc),
        "best": float(inc["cv_rmse_m"].min()) if len(inc) else None,
        "median": float(inc["cv_rmse_m"].median()) if len(inc) else None,
        "worst": float(inc["cv_rmse_m"].max()) if len(inc) else None,
    }


def answer_q3(ev: Evidence) -> Answer:
    q = "Can the historical frames be georeferenced to < 20 m RMSE (target < 10 m)?"
    arts = ["outputs/public/georef_summary.md", "data/processed/historical/registry.parquet"]
    if ev.registry is None or ev.registry.empty:
        return Answer("Q3", q, UNANSWERED, "No georeferencing registry was produced.", arts)
    st = _reg_stats(ev.registry)
    if st["n_inc"]:
        tgt = "meets" if st["best"] < 10 else "does not meet"
        text = (
            f"{st['n_inc']} of {st['n']} frames pass the quality gate (spatial-CV RMSE <= 20 m and "
            f"adequate GCP coverage). Headline spatial-CV RMSE of included frames: best "
            f"{st['best']:.1f} m, median {st['median']:.1f} m, worst {st['worst']:.1f} m; the best "
            f"frame {tgt} the 10 m target."
        )
        return Answer("Q3", q, POSITIVE, text, arts)
    return Answer(
        "Q3",
        q,
        NEGATIVE,
        f"None of {st['n']} frames passed the gate (CV RMSE > 20 m, too few well-spread control "
        "points, or no model could be fitted).",
        arts,
    )


def answer_q4(ev: Evidence) -> Answer:
    q = "Does unsupervised anomaly/change detection recover the known features (blind test)?"
    arts = ["outputs/public/blind_test.md", "outputs/public/blind_test_gate.json"]
    g = ev.gate
    if not g or g.get("status") == "NOT_RUN":
        return Answer(
            "Q4", q, UNANSWERED, "The blind test was not run (no historical scores).", arts
        )
    best = (g.get("best_rows") or [None])[0]
    desc = ""
    if best:
        m = best["metrics"]
        desc = (
            f" Best historical row: `{best['detector']}` / `{best['layer']}` ({best['aoi']}, "
            f"{best['size_m']} m): recall@50 {_f(m.get('recall_at_50'))} (random "
            f"{_f(m.get('random_recall_at_50'))}), AP {_f(m.get('ap'))} (random "
            f"{_f(m.get('random_ap'))})."
        )
    warn = f" Warnings: {'; '.join(g['warnings'])}." if g.get("warnings") else ""
    if g["status"] == "PASS":
        return Answer("Q4", q, POSITIVE, "Blind-test gate PASSED." + desc + warn, arts)
    return Answer(
        "Q4",
        q,
        NEGATIVE,
        "Blind-test gate FAILED; this is a valid negative result (SPEC section 2)." + desc + warn,
        arts,
    )


def answer_q5(ev: Evidence) -> Answer:
    q = "Can LLM document mining produce a useful spatial prior?"
    arts = ["outputs/public/docs_extraction_precision.md", "outputs/public/prior_eval.md"]
    prec = ev.texts.get("docs_extraction_precision.md", "")
    prec_note = ""
    m = re.search(r"insufficient", prec, re.IGNORECASE)
    if prec:
        prec_note = (
            " Extraction precision: insufficient reviewed records."
            if m
            else " Extraction precision is reported in docs_extraction_precision.md."
        )
    summ = ev.prior_summary
    if not summ or not summ.get("rows"):
        return Answer(
            "Q5",
            q,
            UNANSWERED,
            "No evaluated spatial prior: it needs human-reviewed extractions "
            "(`accept` = y) and detector scores." + prec_note,
            arts,
        )
    rows = summ["rows"]
    d_ap = [r["d_ap"] for r in rows if r.get("d_ap") is not None]
    d_rec = [r["d_recall_at_50"] for r in rows if r.get("d_recall_at_50") is not None]
    improved = any(v > 0 for v in d_ap) or any(v > 0 for v in d_rec)
    mean_ap = sum(d_ap) / len(d_ap) if d_ap else 0.0
    status = POSITIVE if improved and mean_ap > 0 else NEGATIVE
    text = (
        f"{len(rows)} detector/layer/variant rows evaluated with leakage control; mean change in "
        f"AP {mean_ap:+.3f}; best change in recall@50 {max(d_rec, default=0.0):+.3f}; "
        f"{summ.get('n_used', 0)} claims used, {summ.get('n_excluded', 0)} excluded for leakage."
        + prec_note
    )
    return Answer("Q5", q, status, text, arts)


def answer_q6(ev: Evidence) -> Answer:
    q = "What candidate areas emerge, ranked, with confidence? (restricted)"
    stats = "candidates_public_stats.md" in ev.texts
    if stats:
        return Answer(
            "Q6",
            q,
            UNANSWERED if not ev.gate or ev.gate.get("status") != "PASS" else POSITIVE,
            "Candidate areas exist only in restricted outputs; the public report carries "
            "aggregate counts only (see below). Locations are not published.",
            ["outputs/public/candidates_public_stats.md"],
        )
    return Answer(
        "Q6",
        q,
        UNANSWERED,
        "No candidate set was produced (fusion needs a passed blind test and human sign-off for "
        "restricted outputs).",
        [],
    )


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


# ------------------------------------------------------------------ report


def _sensor_table(cov: pd.DataFrame) -> str:
    cov = cov.copy()
    cov["mission"] = cov["mission"].fillna("unknown")
    g = (
        cov.groupby(["mission", "resolution_class", "availability"], dropna=False)
        .size()
        .reset_index(name="scenes")
    )
    lines = ["| mission | resolution class | availability | scenes |", "|---|---|---|---|"]
    lines += [
        f"| {r.mission} | {r.resolution_class} | {r.availability} | {r.scenes} |"
        for r in g.itertuples()
    ]
    return "\n".join(lines)


def _georef_table(reg: pd.DataFrame) -> str:
    lines = [
        "| entity | date | method | GCPs | spatial-CV RMSE (m) | included |",
        "|---|---|---|---|---|---|",
    ]
    for r in reg.itertuples():
        d = "" if pd.isna(r.acquisition_date) else str(pd.to_datetime(r.acquisition_date).date())
        cv = "" if pd.isna(r.cv_rmse_m) else f"{r.cv_rmse_m:.1f}"
        inc = "yes" if r.included_for_change_detection else "no"
        lines.append(f"| {r.entity_id} | {d} | {r.method} | {r.n_gcps} | {cv} | {inc} |")
    return "\n".join(lines)


def _figures(s: Settings) -> list[str]:
    pub = s.outputs_public
    out = []
    for pat in ("figures/*.png", "georef/*.png"):
        out += [str(p.relative_to(pub)) for p in sorted(pub.glob(pat))]
    return out


def render_report(
    s: Settings, ev: Evidence | None = None, *, stamp: str | None = None, run_notes=None
) -> str:
    ev = ev or Evidence.load(s)
    answers = [answer_q1(ev), answer_q2(ev), answer_q3(ev), answer_q4(ev), answer_q5(ev)]
    q6 = answer_q6(ev)
    L: list[str] = []
    if stamp:
        L += [f"> **{stamp}**", ""]
    L += [
        "# Sahara Burial Finder: experiment report",
        "",
        f"Generated {datetime.now(UTC).date().isoformat()} from the artefacts in "
        "`outputs/public/` by `sbf report writeup`. Desk analysis only. The deliverable is "
        "evidence about whether the method works, not a public map. Findings are phrased as "
        "candidate areas for verification; none is a finding of waste.",
        "",
        "## Answers to the research questions",
        "",
        "| ID | Question | Answer | Backed by |",
        "|---|---|---|---|",
    ]
    for a in [*answers, q6]:
        L.append(f"| {a.qid} | {a.question} | **{a.status}** | {', '.join(a.artefacts) or '-'} |")
    L.append("")
    for a in [*answers, q6]:
        L += [f"### {a.qid}: {a.status}", "", f"*{a.question}*", "", a.text, ""]

    L += [
        "## Method",
        "",
        "1. **Catalogue** (Phase 1): USGS Declass 1/2/3 scene searches over the AOIs, replayed "
        "from saved API responses; intersected with public reference features; go/no-go rule.",
        "2. **Modern baseline** (Phase 2): Sentinel-2, Sentinel-1, PALSAR and Copernicus DEM "
        "co-registered on a common 10 m grid.",
        "3. **Georeferencing** (Phase 3): feature matching against the modern baseline on stable "
        "terrain, RANSAC, polynomial or thin-plate-spline models (per strip for panoramic "
        "cameras); spatial cross-validation and GCP-coverage gate.",
        "4. **Detection** (Phase 4): texture baseline, DINOv2 embedding anomaly (when weights are "
        "available), SAR and DEM anomalies, change detection on dated pairs. Detectors never see "
        "reference labels; only the scoring harness does.",
        "5. **Blind test**: feature-level recall@K and tile-level average precision against the "
        "exact random expectation; pass/fail gate defined in `sbf.report.blind_test`.",
        "6. **Document mining** (Phase 5): LLM extraction of spatial claims, human review, "
        "spatial prior, evaluated with leakage control.",
        "7. **Fusion** (Phase 6): weighted fusion and candidate polygons; restricted outputs only.",
        "",
        "## Data coverage",
        "",
    ]
    if ev.coverage is not None and len(ev.coverage):
        L += [_sensor_table(ev.coverage), ""]
    else:
        L += ["No coverage table was produced.", ""]
    if "coverage_summary.md" in ev.texts:
        L += [_demote(ev.texts["coverage_summary.md"]), ""]

    L += ["## Georeferencing accuracy", ""]
    if ev.registry is not None and len(ev.registry):
        st = _reg_stats(ev.registry)
        if st["n_inc"]:
            L += [
                f"**Headline (spatial-CV RMSE, included frames):** best {st['best']:.1f} m, "
                f"median {st['median']:.1f} m, worst {st['worst']:.1f} m "
                f"({st['n_inc']} of {st['n']} frames included).",
                "",
            ]
        else:
            L += [f"**No frame passed the gate** ({st['n']} processed).", ""]
        L += [_georef_table(ev.registry), ""]
    else:
        L += ["Georeferencing was not run.", ""]

    L += ["## Blind-test metrics", ""]
    if "blind_test.md" in ev.texts:
        L += [_demote(ev.texts["blind_test.md"]), ""]
    else:
        L += ["The blind test was not run.", ""]
    figs = _figures(s)
    if figs:
        L += ["### Figures (known reference features only)", ""]
        L += [f"![{Path(f).stem}]({f})" for f in figs]
        L.append("")

    L += ["## Document mining", ""]
    for name in ("docs_extraction_precision.md", "prior_eval.md"):
        if name in ev.texts:
            L += [_demote(ev.texts[name]), ""]
    if not any(n in ev.texts for n in ("docs_extraction_precision.md", "prior_eval.md")):
        L += ["Not run: no human-reviewed extractions were available.", ""]

    fus = [
        k for k in ev.texts if k.startswith("fusion_summary") or k == "candidates_public_stats.md"
    ]
    if fus:
        L += ["## Candidate statistics (aggregate only)", ""]
        for k in fus:
            L += [_demote(ev.texts[k]), ""]

    L += ["## Limits", ""] + [f"- {x}" for x in LIMITS] + [""]

    L += ["## Negative and null results", ""]
    neg = [a for a in answers if a.status == NEGATIVE]
    una = [a for a in answers if a.status == UNANSWERED]
    if neg:
        L += [f"- {a.qid} is negative: {a.text}" for a in neg]
    if una:
        L += [f"- {a.qid} is not yet answered: {a.text}" for a in una]
    if not neg and not una:
        L += ["- No negative or unanswered question among Q1-Q5."]
    if run_notes:
        L += ["", "### Pipeline stages not run or skipped", ""] + [f"- {n}" for n in run_notes]
    L += [""]

    L += ["## Recommendations for authorised field-verification partners", ""]
    L += [f"- {x}" for x in RECOMMENDATIONS] + [""]
    return "\n".join(L)


def write_public_report(
    settings: Settings | None = None, *, stamp: str | None = None, run_notes=None
) -> Path:
    s = settings or load_settings()
    s.outputs_public.mkdir(parents=True, exist_ok=True)
    out = s.outputs_public / "report.md"
    out.write_text(render_report(s, stamp=stamp, run_notes=run_notes))
    return out


# ------------------------------------------------------------------ restricted briefing


def render_briefing(s: Settings, top: int = 20) -> str | None:
    """Candidate summaries from restricted ranking.csv (+ confidence tables); None if no ranking."""
    from sbf.core.restricted import restricted_path

    ranking = restricted_path("ranking.csv", settings=s)
    if not ranking.exists():
        return None
    r = pd.read_csv(ranking)
    gate = _read_json(s.outputs_public / "blind_test_gate.json") or {}
    L = [
        "# Restricted briefing: candidate areas for verification",
        "",
        "RESTRICTED. For the rehabilitation agency and named partners only.",
        "",
        f"Phase 4 blind-test gate: {gate.get('status', 'not recorded')}.",
        "",
        "## Statements that apply to every entry",
        "",
    ]
    L += [f"- {x}" for x in VERIFICATION_STATEMENTS] + [""]
    for aoi, grp in r.groupby("aoi_id"):
        conf = restricted_path(f"candidate_confidence_{aoi}.csv", settings=s)
        grp = grp.sort_values("rank").head(top)
        if conf.exists():
            c = pd.read_csv(conf)
            keep = [x for x in ("cand_id", "confidence", "justification") if x in c.columns]
            if "cand_id" in keep:
                grp = grp.merge(c[keep], on="cand_id", how="left")
        L += [f"## {aoi}: top {len(grp)} candidates", ""]
        cols = [
            c
            for c in (
                "rank", "cand_id", "label", "bracket", "mean_S", "area_m2", "n_families",
                "confidence",
            )
            if c in grp.columns
        ]  # fmt: skip
        L += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for row in grp.itertuples(index=False):
            rd = row._asdict()
            L.append("| " + " | ".join(_cell(rd.get(c)) for c in cols) + " |")
        L.append("")
        if "justification" in grp.columns:
            for row in grp.itertuples():
                if isinstance(row.justification, str):
                    L.append(f"- `{row.cand_id}`: {row.justification}")
            L.append("")
    L += (
        [
            "Review sheets with imagery crops are in `candidate_sheets/` (restricted).",
            "",
            "## Recommendations for authorised partners",
            "",
        ]
        + [f"- {x}" for x in RECOMMENDATIONS]
        + [""]
    )
    return "\n".join(L)


def _cell(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def write_briefing(settings: Settings | None = None, top: int = 20) -> Path | None:
    """Write outputs/restricted/briefing.md (via restricted_path; encrypted when keyed)."""
    from sbf.core.restricted import encrypt_if_key, restricted_path

    s = settings or load_settings()
    text = render_briefing(s, top)
    if text is None:
        return None
    p = restricted_path("briefing.md", settings=s)
    p.write_text(text)
    return encrypt_if_key(p)
