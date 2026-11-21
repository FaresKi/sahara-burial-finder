"""Stage list of `sbf run-all` (ticket 18). Each stage wraps existing package code or CLI.

Order: catalog-replay -> catalog-report (GATE) -> modern-stack -> georef-batch -> detect-texture ->
detect-embedding -> detect-sar-dem -> detect-change -> blind-test -> docs-precision -> docs-prior ->
docs-prior-eval -> fuse* -> candidate-sheets* -> writeup -> check-public   (* restricted, off by
default). Reproducible, offline, public-only rebuild from `data/raw/` (SPEC section 9.3).
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pandas as pd

from sbf.pipeline.core import (
    EXIT_LEAK,
    EXIT_NEEDS_REVIEW,
    EXIT_NO_GO,
    NOT_VALID,
    Ctx,
    Outcome,
    Stage,
)

TEXTURE_HIST_SIZE = 64
TEXTURE_MODERN_SIZE = 128
SAR_LAYERS = ("palsar_hv", "palsar_hh", "s1_vv_mean", "s1_vh_mean")
EMBED_SIZE_HIST = 64
EMBED_SIZE_MODERN = 128


# ------------------------------------------------------------------ helpers


def run_cli(args: list[str]) -> tuple[int, str]:
    """Invoke the real `sbf` CLI in-process; returns (exit code, combined output)."""
    from typer.testing import CliRunner

    from sbf.cli import app

    res = CliRunner().invoke(app, args)
    out = res.output or ""
    if res.exception and not isinstance(res.exception, SystemExit):
        out += f"\n{type(res.exception).__name__}: {res.exception}"
    return res.exit_code, out


def _resolve(ctx: Ctx, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ctx.root / p


def all_aois(ctx: Ctx) -> list[str]:
    from sbf.core.aoi import load_aois

    return list(load_aois(ctx.settings)["aoi_id"])


def raw_modern_aois(ctx: Ctx) -> list[str]:
    return [
        a
        for a in all_aois(ctx)
        if (ctx.settings.data_raw / "modern" / a).is_dir()
        and any((ctx.settings.data_raw / "modern" / a).glob("*.tif"))
    ]


def stack_aois(ctx: Ctx) -> list[str]:
    return [
        a
        for a in all_aois(ctx)
        if (ctx.settings.data_processed / "modern" / a / "stack.json").exists()
    ]


def score_aois(ctx: Ctx) -> list[str]:
    d = ctx.settings.data_processed / "scores"
    return (
        sorted(p.name for p in d.iterdir() if p.is_dir() and any(p.glob("*.parquet")))
        if d.is_dir()
        else []
    )


def registry(ctx: Ctx, included_only: bool = True) -> pd.DataFrame | None:
    from sbf.georef.registry import load_registry

    try:
        return load_registry(included_only, ctx.settings)
    except FileNotFoundError:
        return None


def hist_layers(ctx: Ctx, aoi: str) -> list[tuple[str, Path]]:
    """(layer name, georeferenced raster) of every included frame of an AOI."""
    from sbf.detect.change import historical_layer_name

    reg = registry(ctx)
    if reg is None or reg.empty:
        return []
    out = []
    for row in reg[reg["aoi"].astype(str) == aoi].to_dict("records"):
        f = row.get("georef_file")
        if f and _resolve(ctx, f).exists():
            out.append((historical_layer_name(row), _resolve(ctx, f)))
    return out


def modern_layer(ctx: Ctx, aoi: str, name: str) -> tuple[Path, int] | None:
    mf = ctx.settings.data_processed / "modern" / aoi / "stack.json"
    if not mf.exists():
        return None
    info = json.loads(mf.read_text())["layers"].get(name)
    if not info:
        return None
    names = [b["name"] for b in info.get("bands", [])]
    band = names.index("B4") + 1 if "B4" in names else 1
    return mf.parent / info["file"], band


def embedding_available() -> str | None:
    """None when DINOv2 can run offline, else the reason it cannot."""
    for mod in ("torch", "transformers"):
        if importlib.util.find_spec(mod) is None:
            return f"python package {mod!r} not installed (uv sync --extra ml)"
    hub = Path(
        os.environ.get("HF_HUB_CACHE")
        or Path(os.environ.get("HF_HOME") or "~/.cache/huggingface").expanduser() / "hub"
    )
    if not (hub / "models--facebook--dinov2-small").exists():
        return "DINOv2 weights not cached locally (human step: first `sbf detect embedding embed`)"
    return None


def blind_status(ctx: Ctx) -> str | None:
    p = ctx.settings.outputs_public / "blind_test_gate.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("status")
    except json.JSONDecodeError:
        return None


def accepted_claims(ctx: Ctx) -> list:
    if not (ctx.settings.data_interim / "docs" / "review.csv").exists():
        return []
    from sbf.docs_mining.prior import load_accepted_claims

    return load_accepted_claims(ctx.settings)


# ------------------------------------------------------------------ 1-2 catalogue


def _need_raw_catalog(ctx: Ctx) -> str | None:
    p = ctx.settings.data_raw / "catalog" / "manifest.jsonl"
    return None if p.exists() else "data/raw/catalog/manifest.jsonl"


def run_catalog_replay(ctx: Ctx) -> Outcome:
    from sbf.catalog.download import refine_availability
    from sbf.catalog.replay import ReplayClient
    from sbf.catalog.search import run_search, write_search_outputs
    from sbf.core.aoi import load_aois, load_reference_features

    s = ctx.settings
    client = ReplayClient(s.data_raw / "catalog")
    aois = load_aois(s)
    searched = client.searched_aois(aois)
    if not searched:
        return Outcome.skipped(
            "missing input: no recorded scene-search response matches a configured AOI"
        )
    start, end = client.recorded_range()
    kw = {k: v for k, v in (("start", start), ("end", end)) if v}
    res = run_search(client, aois, load_reference_features(s), searched, **kw)
    notes = []
    if client.has_endpoint("download-options") and len(res.coverage):
        res.coverage, fails = refine_availability(client, res.coverage)
        if fails:
            notes.append(f"{len(fails)} availability replay failure(s)")
    write_search_outputs(
        res, coverage_csv=s.outputs_public / "coverage.csv", interim_dir=s.data_interim / "catalog"
    )
    if res.failures:
        notes.append(f"{len(res.failures)} replay failure(s) in data/interim/catalog/failures.json")
    return Outcome.ok(
        f"{len(res.datasets)} datasets, {len(res.coverage)} scenes, AOIs {searched}"
        + (f"; {'; '.join(notes)}" if notes else "")
    )


def run_catalog_report(ctx: Ctx) -> Outcome:
    from sbf.catalog.counts import collect_counts
    from sbf.catalog.report import build_report
    from sbf.core.aoi import load_aois, load_reference_features

    s = ctx.settings
    aois = load_aois(s)
    counts = collect_counts(
        None, {r.aoi_id: r.geometry for r in aois.itertuples()}, "1959-01-01", "2026-12-31"
    )
    d = build_report(s, aois, load_reference_features(s), counts)
    status = d.status
    (s.outputs_public / "go_decision.json").write_text(
        json.dumps(
            {
                "status": status,
                "reasons": d.reasons,
                "usable_hires_frames": d.usable_hires_frames,
                "temporal_features": d.temporal_features,
                "review_entity_ids": d.review_entity_ids,
            },
            indent=2,
        )
    )
    if status == "GO":
        return Outcome.ok("GO (pending human review of coverage_summary.md)")
    if ctx.bypass_gates:
        return Outcome.ok(f"{status} (gate bypassed: {NOT_VALID})")
    if status == "NEEDS_REVIEW":
        return Outcome(
            "gate_stop",
            f"NEEDS_REVIEW: fill `usable` in outputs/public/browse_review.csv for "
            f"{len(d.review_entity_ids)} frame(s), then re-run",
            EXIT_NEEDS_REVIEW,
        )
    return Outcome(
        "gate_stop", "NO_GO: see outputs/public/NO_GO.md; stop for human review", EXIT_NO_GO
    )


# ------------------------------------------------------------------ 3-4 modern + georef


def _need_raw_modern(ctx: Ctx) -> str | None:
    return None if raw_modern_aois(ctx) else "data/raw/modern/<AOI>/*.tif"


def run_modern_stack(ctx: Ctx) -> Outcome:
    from sbf.ingest.modern import build_stack

    notes = []
    for aoi in raw_modern_aois(ctx):
        m = build_stack(aoi, settings=ctx.settings)
        notes.append(f"{aoi}: {len(m['layers'])} layers, {len(m['missing'])} missing")
    return Outcome.ok("; ".join(notes))


def _need_frames(ctx: Ctx) -> str | None:
    from sbf.georef.registry import downloaded_entities

    if not downloaded_entities(ctx.settings):
        return "data/raw/declass/<entity>/*.tif (no downloaded frames)"
    if not (ctx.settings.outputs_public / "coverage.csv").exists():
        return "outputs/public/coverage.csv"
    if not stack_aois(ctx):
        return "data/processed/modern/<AOI>/stack.json"
    return None


def run_georef_batch(ctx: Ctx) -> Outcome:
    from sbf.georef.registry import run_batch

    df = run_batch(ctx.settings)
    inc = int(df["included_for_change_detection"].sum())
    bad = int((df["status"] != "ok").sum())
    return Outcome.ok(f"{len(df)} frames, {inc} included, {bad} failed/unsolved")


# ------------------------------------------------------------------ 5-8 detectors


def _need_registry_or_stack(ctx: Ctx) -> str | None:
    if registry(ctx) is None and not stack_aois(ctx):
        return "data/processed/historical/registry.parquet or modern stack"
    return None


def run_texture(ctx: Ctx) -> Outcome:
    from sbf.detect.scores import random_scores, write_scores
    from sbf.detect.texture_detector import run_texture_detector

    s = ctx.settings
    done, failed = 0, []
    for aoi in all_aois(ctx):
        jobs = [(n, p, 1, TEXTURE_HIST_SIZE, "clahe") for n, p in hist_layers(ctx, aoi)]
        m = modern_layer(ctx, aoi, "s2_median")
        if m:
            jobs.append(("s2_median", m[0], m[1], TEXTURE_MODERN_SIZE, "none"))
        for layer, raster, band, size, pre in jobs:
            try:
                df = run_texture_detector(
                    aoi, layer, raster, size, "iforest", band=band, preprocess=pre, settings=s
                )
                write_scores(random_scores(df["tile_id"].tolist(), layer, seed=0), s)
                done += 1
            except Exception as exc:  # noqa: BLE001 - one layer failing must not hide the others
                failed.append(f"{aoi}/{layer}: {type(exc).__name__}: {exc}")
    if not done and failed:
        return Outcome.failed("; ".join(failed))
    if not done:
        return Outcome.skipped("missing input: no georeferenced frames and no modern s2_median")
    return Outcome.ok(f"{done} layer(s) scored" + (f"; FAILED {failed}" if failed else ""))


def _need_embedding(ctx: Ctx) -> str | None:
    why = embedding_available()
    if why:
        return why
    return _need_registry_or_stack(ctx)


def run_embedding(ctx: Ctx) -> Outcome:
    n = 0
    for aoi in all_aois(ctx):
        jobs = [(nm, p, EMBED_SIZE_HIST) for nm, p in hist_layers(ctx, aoi)]
        m = modern_layer(ctx, aoi, "s2_median")
        if m:
            jobs.append(("s2_median", m[0], EMBED_SIZE_MODERN))
        for layer, raster, size in jobs:
            for sub in (
                [
                    "embed",
                    "--aoi",
                    aoi,
                    "--layer",
                    layer,
                    "--raster",
                    str(raster),
                    "--size",
                    str(size),
                ],
                ["score", "--aoi", aoi, "--layer", layer, "--size", str(size)],
            ):
                code, out = run_cli(["detect", "embedding", *sub])
                if code != 0:
                    return Outcome.failed(f"detect embedding {sub[0]} {aoi}/{layer}: {out[-300:]}")
            n += 1
    return Outcome.ok(f"{n} layer(s) embedded and scored")


def _need_modern_stack(ctx: Ctx) -> str | None:
    return None if stack_aois(ctx) else "data/processed/modern/<AOI>/stack.json"


def run_sar_dem(ctx: Ctx) -> Outcome:
    n, notes = 0, []
    for aoi in stack_aois(ctx):
        man = json.loads((ctx.settings.data_processed / "modern" / aoi / "stack.json").read_text())
        have = man["layers"]
        for layer in SAR_LAYERS:
            if layer not in have:
                notes.append(f"{aoi}/{layer} absent")
                continue
            code, out = run_cli(
                ["detect", "sar-dem", "sar", "--aoi", aoi, "--layer", layer, "--size", "128"]
            )
            if code != 0:
                return Outcome.failed(f"sar {aoi}/{layer}: {out[-300:]}")
            n += 1
        if "dem" in have:
            code, out = run_cli(["detect", "sar-dem", "dem", "--aoi", aoi, "--size", "128"])
            if code != 0:
                return Outcome.failed(f"dem {aoi}: {out[-300:]}")
            n += 1
    if not n:
        return Outcome.skipped("missing input: no SAR/DEM layers in the modern stack")
    return Outcome.ok(f"{n} table(s)" + (f"; {len(notes)} layer(s) absent" if notes else ""))


def _need_registry(ctx: Ctx) -> str | None:
    reg = registry(ctx)
    if reg is None:
        return "data/processed/historical/registry.parquet"
    if reg.empty:
        return "no frame included for change detection (registry has 0 included rows)"
    return None


def run_change_stage(ctx: Ctx) -> Outcome:
    embed_ok = embedding_available() is None
    n, notes = 0, []
    for aoi in stack_aois(ctx):
        flags = ["--same-era"] if ctx.same_era else []
        code, out = run_cli(["detect", "change", "pairs", "--aoi", aoi, *flags])
        if code != 0:
            return Outcome.failed(f"change pairs {aoi}: {out[-300:]}")
        if out.strip().splitlines()[-1].strip() == "0 pairs":
            notes.append(f"{aoi}: no frame pairs overlap")
            continue
        sub = "run" if embed_ok else "diff"
        code, out = run_cli(["detect", "change", sub, "--aoi", aoi, *flags])
        if code != 0:
            return Outcome.failed(f"change {sub} {aoi}: {out[-300:]}")
        n += 1
    if not embed_ok:
        notes.append("change_embed skipped: " + (embedding_available() or ""))
    if not n:
        return Outcome.skipped("missing input: no overlapping frame pairs. " + "; ".join(notes))
    return Outcome.ok(f"{n} AOI(s) scored; " + "; ".join(notes))


# ------------------------------------------------------------------ 9 blind test


def _need_scores(ctx: Ctx) -> str | None:
    return None if score_aois(ctx) else "data/processed/scores/<AOI>/*.parquet"


def run_blind(ctx: Ctx) -> Outcome:
    from sbf.report.blind_test import run_blind_test

    note = ""
    try:
        res = run_blind_test(settings=ctx.settings, figures=True)
    except Exception as exc:  # noqa: BLE001 - figures are optional; the gate must still be written
        note = f" (figures failed: {type(exc).__name__}: {exc})"
        res = run_blind_test(settings=ctx.settings, figures=False)
    if res.status == "NOT_RUN":
        return Outcome.ok("gate NOT_RUN: no historical score tables" + note)
    if res.status == "FAIL":
        return Outcome.ok("gate FAIL: negative result recorded; fusion will not run" + note)
    return Outcome.ok("gate PASS" + note)


# ------------------------------------------------------------------ 10-12 documents


def _need_review(ctx: Ctx) -> str | None:
    p = ctx.settings.data_interim / "docs" / "review.csv"
    return None if p.exists() else "data/interim/docs/review.csv (human extraction review)"


def run_docs_precision(ctx: Ctx) -> Outcome:
    from sbf.docs_mining.review import write_precision_report

    _, p = write_precision_report(ctx.settings)
    return Outcome.ok(
        "insufficient reviewed records" if not p.sufficient else f"precision {p.precision:.1%}"
    )


def _need_accepted(ctx: Ctx) -> str | None:
    miss = _need_review(ctx)
    if miss:
        return miss
    return None if accepted_claims(ctx) else "no accepted rows in review.csv (human gate)"


def run_docs_prior(ctx: Ctx) -> Outcome:
    from sbf.docs_mining.prior import build_prior_products

    done, refused = 0, []
    for aoi in stack_aois(ctx) or all_aois(ctx):
        try:
            build_prior_products(aoi, min_accepted=1, settings=ctx.settings)
            done += 1
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            refused.append(f"{aoi}: {exc}")
    if not done:
        return Outcome.skipped("refused: " + "; ".join(refused))
    return Outcome.ok(f"{done} prior(s)" + (f"; refused {refused}" if refused else ""))


def _need_prior_eval(ctx: Ctx) -> str | None:
    return _need_accepted(ctx) or _need_scores(ctx)


def run_docs_prior_eval(ctx: Ctx) -> Outcome:
    from sbf.docs_mining.prior_eval import evaluate_prior, render_report

    pub = ctx.settings.outputs_public
    parts, rows, n_used, n_excl = [], [], 0, 0
    for aoi in score_aois(ctx):
        res = evaluate_prior(aoi, settings=ctx.settings, write=False)
        parts.append(render_report(aoi, res))
        n_used += len(res.used_ids)
        n_excl += len(res.excluded)
        for r in res.table.to_dict("records"):
            rows.append(
                {
                    "aoi": aoi,
                    "detector": r["detector"],
                    "layer": r["layer"],
                    "variant": r["variant"],
                    "d_recall_at_50": _num(r.get("d_recall_at_50")),
                    "d_ap": _num(r.get("d_ap")),
                }
            )
    (pub / "prior_eval.md").write_text("\n\n".join(parts))
    (pub / "prior_eval_summary.json").write_text(
        json.dumps({"rows": rows, "n_used": n_used, "n_excluded": n_excl}, indent=2)
    )
    return Outcome.ok(f"{len(rows)} evaluated rows; {n_used} claims used, {n_excl} excluded")


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


# ------------------------------------------------------------------ 13-14 restricted


def _need_fusion(ctx: Ctx) -> str | None:
    return _need_scores(ctx) or (None if registry(ctx) is not None else "registry.parquet")


def run_fuse(ctx: Ctx) -> Outcome:
    st = blind_status(ctx)
    if st != "PASS" and not ctx.bypass_gates:
        return Outcome.skipped(
            f"blind-test gate is {st or 'not recorded'}, not PASS: fusion is not run (SPEC 9.2)"
        )
    n = 0
    for aoi in score_aois(ctx):
        code, out = run_cli(["fuse", "run", "--aoi", aoi, "--confirm-restricted"])
        if code != 0:
            return Outcome.failed(f"fuse {aoi}: {out[-300:]}")
        n += 1
    return Outcome.ok(f"{n} AOI(s) fused (restricted outputs; counts only here)")


def _need_candidates(ctx: Ctx) -> str | None:
    return (
        None
        if any(ctx.settings.outputs_restricted.glob("candidates_*.gpkg"))
        else "outputs/restricted/candidates_<AOI>.gpkg"
    )


def run_sheets(ctx: Ctx) -> Outcome:
    n = 0
    for gpkg in sorted(ctx.settings.outputs_restricted.glob("candidates_*.gpkg")):
        aoi = gpkg.stem.removeprefix("candidates_")
        code, out = run_cli(["report", "sheets", "--aoi", aoi, "--confirm-restricted"])
        if code != 0:
            return Outcome.failed(f"sheets {aoi}: {out[-300:]}")
        n += 1
    code, out = run_cli(["report", "public-stats"])
    if code != 0:
        return Outcome.failed(f"public-stats: {out[-300:]}")
    return Outcome.ok(f"{n} AOI(s): sheets + public aggregate stats")


# ------------------------------------------------------------------ 15-16 write-up + leak check


def run_writeup(ctx: Ctx) -> Outcome:
    from sbf.pipeline.core import read_manifest
    from sbf.report.writeup import write_briefing, write_public_report

    man = read_manifest(ctx)
    notes = [
        f"`{n}`: {r['detail']}"
        for n, r in man.get("stages", {}).items()
        if r.get("status") == "skipped" and n != "writeup"
    ]
    stamp = NOT_VALID if ctx.bypass_gates else None
    path = write_public_report(ctx.settings, stamp=stamp, run_notes=notes)
    msg = f"{path.name}"
    if ctx.include_restricted:
        b = write_briefing(ctx.settings)
        msg += "; briefing.md (restricted)" if b else "; no briefing (no ranking.csv)"
    return Outcome.ok(msg)


def run_check_public(ctx: Ctx) -> Outcome:
    from sbf.report.public_check import check_public

    bad = check_public(ctx.settings)
    if bad:
        return Outcome.failed(
            f"{len(bad)} possible leak(s) in outputs/public: "
            + "; ".join(str(v) for v in bad[:10]),
            EXIT_LEAK,
        )
    return Outcome.ok("outputs/public clean")


def _public_inputs(ctx: Ctx) -> list[str]:
    return [
        "outputs/public/*.md",
        "outputs/public/*.json",
        "outputs/public/*.csv",
        "outputs/public/figures",
        "outputs/public/georef",
        "data/processed/historical/registry.parquet",
    ]


def build_stages() -> list[Stage]:
    return [
        Stage(
            "catalog-replay",
            "rebuild coverage.csv from saved raw M2M responses (never calls the API)",
            run_catalog_replay,
            inputs=lambda c: [
                "data/raw/catalog/*",
                "config/aois.geojson",
                "config/reference_features.geojson",
                "config/reference_refinements.geojson",
            ],
            outputs=lambda c: ["outputs/public/coverage.csv"],
            needs=_need_raw_catalog,
            after_go_gate=False,
        ),
        Stage(
            "catalog-report",
            "coverage map + summary + Phase 1 go/no-go decision",
            run_catalog_report,
            inputs=lambda c: [
                "outputs/public/coverage.csv",
                "outputs/public/browse_review.csv",
                "config/reference_features.geojson",
            ],
            outputs=lambda c: [
                "outputs/public/go_decision.json",
                "outputs/public/coverage_summary.md",
            ],
            needs=lambda c: (
                None
                if (c.settings.outputs_public / "coverage.csv").exists()
                else "outputs/public/coverage.csv"
            ),
            gate="Phase 1 go/no-go (NO_GO / NEEDS_REVIEW stops the pipeline)",
            after_go_gate=False,
        ),
        Stage(
            "modern-stack",
            "build the modern 10 m stack per AOI from data/raw/modern",
            run_modern_stack,
            inputs=lambda c: ["data/raw/modern"],
            outputs=lambda c: ["data/processed/modern/*/stack.json"],
            needs=_need_raw_modern,
        ),
        Stage(
            "georef-batch",
            "georeference all downloaded frames; registry + georef_summary.md",
            run_georef_batch,
            inputs=lambda c: [
                "data/raw/declass",
                "outputs/public/coverage.csv",
                "data/processed/modern",
            ],
            outputs=lambda c: [
                "data/processed/historical/registry.parquet",
                "outputs/public/georef_summary.md",
            ],
            needs=_need_frames,
        ),
        Stage(
            "detect-texture",
            "texture baseline + random baseline on every included frame and modern s2_median",
            run_texture,
            inputs=lambda c: [
                "data/processed/historical/registry.parquet",
                "data/processed/historical/*/*_georef.tif",
                "data/processed/modern",
            ],
            outputs=lambda c: ["data/processed/scores/*/texture__*.parquet"],
            needs=_need_registry_or_stack,
        ),
        Stage(
            "detect-embedding",
            "DINOv2 embedding anomaly (skipped when torch/weights are unavailable)",
            run_embedding,
            inputs=lambda c: [
                "data/processed/historical/registry.parquet",
                "data/processed/historical/*/*_georef.tif",
                "data/processed/modern",
            ],
            outputs=lambda c: ["data/processed/scores/*/embed_*.parquet"],
            needs=_need_embedding,
        ),
        Stage(
            "detect-sar-dem",
            "SAR backscatter and DEM micro-relief anomalies",
            run_sar_dem,
            inputs=lambda c: ["data/processed/modern"],
            outputs=lambda c: ["data/processed/scores/*/dem_relief__*.parquet"],
            needs=_need_modern_stack,
        ),
        Stage(
            "detect-change",
            "change detection on dated frame pairs (diff; embedding when available)",
            run_change_stage,
            inputs=lambda c: [
                "data/processed/historical/registry.parquet",
                "data/processed/historical/*/*_georef.tif",
                "data/processed/modern",
            ],
            outputs=lambda c: ["data/processed/scores/*/change_diff__*.parquet"],
            needs=_need_registry,
        ),
        Stage(
            "blind-test",
            "blind test, gate json, figures",
            run_blind,
            inputs=lambda c: [
                "data/processed/scores",
                "config/reference_features.geojson",
                "config/reference_refinements.geojson",
                "config/exclusions.geojson",
            ],
            outputs=lambda c: [
                "outputs/public/blind_test_gate.json",
                "outputs/public/blind_test.md",
            ],
            needs=_need_scores,
            gate="Phase 4 blind test (FAIL is recorded as a negative result; fusion then does not run)",
        ),
        Stage(
            "docs-precision",
            "extraction precision from the human-reviewed CSV",
            run_docs_precision,
            inputs=lambda c: ["data/interim/docs/review.csv"],
            outputs=lambda c: ["outputs/public/docs_extraction_precision.md"],
            needs=_need_review,
            gate="Phase 5 extraction review (human)",
        ),
        Stage(
            "docs-prior",
            "spatial prior from accepted extractions",
            run_docs_prior,
            inputs=lambda c: [
                "data/interim/docs/review.csv",
                "config/anchors.geojson",
                "config/reference_features.geojson",
            ],
            outputs=lambda c: ["data/processed/prior/prior_*.tif"],
            needs=_need_accepted,
            gate="Phase 5 extraction review (human)",
        ),
        Stage(
            "docs-prior-eval",
            "prior x detector evaluation with leakage control",
            run_docs_prior_eval,
            inputs=lambda c: [
                "data/interim/docs/review.csv",
                "data/processed/scores",
                "config/anchors.geojson",
            ],
            outputs=lambda c: [
                "outputs/public/prior_eval.md",
                "outputs/public/prior_eval_summary.json",
            ],
            needs=_need_prior_eval,
        ),
        Stage(
            "fuse",
            "fusion and candidate polygons (restricted outputs)",
            run_fuse,
            inputs=lambda c: [
                "data/processed/scores",
                "data/processed/historical/registry.parquet",
                "outputs/public/blind_test_gate.json",
            ],
            outputs=lambda c: ["outputs/restricted/ranking.csv"],
            needs=_need_fusion,
            gate="restricted outputs: --include-restricted --confirm-restricted and blind-test PASS",
            restricted=True,
        ),
        Stage(
            "candidate-sheets",
            "restricted review sheets + public aggregate stats",
            run_sheets,
            inputs=lambda c: [
                "outputs/restricted/ranking.csv",
                "outputs/public/blind_test_gate.json",
            ],
            outputs=lambda c: ["outputs/public/candidates_public_stats.md"],
            needs=_need_candidates,
            gate="restricted outputs: --include-restricted --confirm-restricted",
            restricted=True,
        ),
        Stage(
            "writeup",
            "outputs/public/report.md (and restricted briefing.md when enabled)",
            run_writeup,
            inputs=_public_inputs,
            outputs=lambda c: ["outputs/public/report.md"],
        ),
        Stage(
            "check-public",
            "leak guard: no non-public coordinates under outputs/public",
            run_check_public,
            needs=lambda c: None if c.settings.outputs_public.exists() else "outputs/public",
            always_run=True,
            after_go_gate=False,
        ),
    ]
