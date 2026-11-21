"""`sbf docs`: Phase 5 document ingestion, extraction, review, precision."""

from __future__ import annotations

import typer

app = typer.Typer(help="Phase 5: LLM document mining to spatial prior.", no_args_is_help=True)


@app.command()
def ingest() -> None:
    """Register PDFs in data/raw/docs/, write manifest.json and chunks.jsonl."""
    from sbf.docs_mining.pdf import ingest_docs

    recs = ingest_docs()
    for r in recs:
        typer.echo(f"{r.doc_id}: {r.n_pages} pages, {r.n_chunks} chunks")
    if not recs:
        typer.echo("No PDFs found in data/raw/docs/ (see notes/PHASE_5_NOTES.md).")


@app.command()
def extract(
    limit: int | None = typer.Option(None, help="Only process the first N chunks."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate cost; make no API call."),
) -> None:
    """Extract spatial claims with the Claude API (spends money: human gate)."""
    from sbf.docs_mining.extract import ClaudeExtractor, estimate_cost, run_extraction
    from sbf.docs_mining.pdf import load_chunks

    chunks = load_chunks()
    if limit is not None:
        chunks = chunks[:limit]
    if dry_run:
        e = estimate_cost(chunks)
        typer.echo(
            f"DRY RUN: {e.n_chunks} chunks, ~{e.input_tokens} input + ~{e.output_tokens} output "
            f"tokens, model {e.model}, est. ${e.cost_usd:.2f} (placeholder prices; no API call)"
        )
        return
    s = run_extraction(ClaudeExtractor(), limit=limit)
    typer.echo(
        f"chunks={s.chunks} cached={s.cached} failed={s.failed} "
        f"claims={s.claims} rejected_quotes={s.rejected_quotes}"
    )


@app.command("export-review")
def export_review_cmd() -> None:
    """Write data/interim/docs/review.csv for human accept/reject."""
    from sbf.docs_mining.review import export_review

    typer.echo(str(export_review()))


@app.command()
def precision() -> None:
    """Compute extraction precision from the reviewed CSV; write the public report."""
    from sbf.docs_mining.review import write_precision_report

    out, p = write_precision_report()
    shown = "insufficient" if not p.sufficient else f"{p.precision:.1%}"
    typer.echo(f"precision: {shown} ({p.accepted}/{p.reviewed} reviewed) -> {out}")


@app.command()
def anchors() -> None:
    """List anchor resolution status of all accepted claims (no coordinates, no quotes)."""
    from sbf.core.config import load_settings
    from sbf.docs_mining.gazetteer import load_gazetteer
    from sbf.docs_mining.prior import load_accepted_claims, parse_bearing

    s = load_settings()
    gz = load_gazetteer(s)
    claims = load_accepted_claims(s)
    if not claims:
        typer.echo("No accepted rows in review.csv (human gate).")
        return
    counts: dict[str, int] = {}
    for c in claims:
        a = gz.resolve(c.anchor)
        if a is None:
            status = "UNRESOLVED (unknown anchor)"
        elif not a.resolved:
            status = f"UNRESOLVED ({a.anchor_id}: needs_human_location)"
        else:
            status = f"resolved -> {a.anchor_id} [{a.aoi_id}]"
        brg = (
            "bearing ok"
            if parse_bearing(c.bearing) is not None
            else ("bearing unparsed" if c.bearing else "no bearing")
        )
        counts[status] = counts.get(status, 0) + 1
        typer.echo(f"{c.extraction_id}  {status}  anchor={c.anchor!r}  {brg}")
    typer.echo("")
    for k, v in sorted(counts.items()):
        typer.echo(f"{v:4d}  {k}")


@app.command()
def prior(
    aoi: str = typer.Option(..., "--aoi", help="AOI id, e.g. REG."),
    min_accepted: int = typer.Option(1, "--min-accepted", help="Refuse below this many."),
) -> None:
    """Build prior_{AOI}.tif and the per-tile `prior` score tables (needs accepted rows)."""
    from sbf.docs_mining.prior import build_prior_products

    try:
        r = build_prior_products(aoi, min_accepted=min_accepted)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        typer.echo(f"refused: {e}")
        raise typer.Exit(1) from None
    typer.echo(
        f"prior written: {r.raster} ({len(r.used_ids)}/{r.n_accepted} claims used, "
        f"{len(r.score_tables)} score table(s))"
    )
    for status, ids in sorted(r.unresolved.items()):
        typer.echo(f"not used ({status}): {len(ids)}")


@app.command("prior-eval")
def prior_eval(aoi: str = typer.Option(..., "--aoi", help="AOI id, e.g. REG.")) -> None:
    """Evaluate prior x detector scores with leakage control; write outputs/public/prior_eval.md."""
    from sbf.core.config import load_settings
    from sbf.docs_mining.prior_eval import evaluate_prior

    s = load_settings()
    res = evaluate_prior(aoi, settings=s)
    typer.echo(
        f"claims: {res.n_accepted} accepted, {len(res.used_ids)} used, "
        f"{len(res.excluded)} excluded -> {s.outputs_public / 'prior_eval.md'}"
    )
    if res.note:
        typer.echo(res.note)
