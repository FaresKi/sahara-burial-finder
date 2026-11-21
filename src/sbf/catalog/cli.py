"""CLI sub-app for sbf.catalog: `sbf catalog search` and `sbf catalog report`."""

from __future__ import annotations

from typing import Annotated

import typer

from sbf.catalog.counts import collect_counts
from sbf.catalog.m2m import M2MClient, M2MError, RawStore
from sbf.catalog.report import build_report
from sbf.catalog.search import END_DEFAULT, START_DEFAULT, run_search, write_search_outputs
from sbf.core.aoi import load_aois, load_reference_features
from sbf.core.config import load_settings

app = typer.Typer(
    help="Phase 1: EarthExplorer catalogue search and coverage report.", no_args_is_help=True
)


@app.command()
def search(
    aoi: Annotated[list[str], typer.Option("--aoi", help="AOI id (REG, INE); repeatable")],
    start: Annotated[str, typer.Option(help="Acquisition start YYYY-MM-DD")] = START_DEFAULT,
    end: Annotated[str, typer.Option(help="Acquisition end YYYY-MM-DD")] = END_DEFAULT,
) -> None:
    """Query all Declass 1/2/3 datasets via M2M; write outputs/public/coverage.csv.

    Needs USGS_USERNAME and USGS_M2M_TOKEN (human gate: M2M access request).
    """
    settings = load_settings()
    aois = load_aois(settings)
    unknown = sorted(set(aoi) - set(aois["aoi_id"]))
    if unknown:
        typer.echo(f"unknown AOI(s): {unknown}; known: {sorted(aois['aoi_id'])}", err=True)
        raise typer.Exit(2)
    try:
        M2MClient.credentials()
    except M2MError as exc:
        typer.echo(f"{exc}. See notes/PHASE_1_NOTES.md.", err=True)
        raise typer.Exit(2) from exc
    client = M2MClient(RawStore(settings.data_raw / "catalog"))
    res = run_search(client, aois, load_reference_features(settings), aoi, start, end)
    write_search_outputs(
        res,
        coverage_csv=settings.outputs_public / "coverage.csv",
        interim_dir=settings.data_interim / "catalog",
    )
    typer.echo(
        f"datasets: {len(res.datasets)}  scenes: {len(res.coverage)}  failures: {len(res.failures)}"
    )
    if res.failures:
        typer.echo("API failures recorded in data/interim/catalog/failures.json", err=True)
        raise typer.Exit(1)


@app.command()
def report(
    start: Annotated[str, typer.Option(help="Count range start")] = START_DEFAULT,
    end: Annotated[str, typer.Option(help="Count range end")] = "2026-12-31",
) -> None:
    """Build coverage map + summary from coverage.csv and apply the three-state go rule.

    Exit codes: GO 0, NO_GO 1, NEEDS_REVIEW 3 (unknown-cloud frames need a human browse check;
    fill `usable` in outputs/public/browse_review.csv and re-run), bad input 2.

    No SceneCounter is wired yet (GEE layer is separate), so Landsat/S2 counts show `not_run`
    and SPOT shows `unverified`; nothing is filled in.
    """
    settings = load_settings()
    aois = load_aois(settings)
    if not (settings.outputs_public / "coverage.csv").exists():
        typer.echo("outputs/public/coverage.csv missing: run `sbf catalog search` first", err=True)
        raise typer.Exit(2)
    counts = collect_counts(None, {r.aoi_id: r.geometry for r in aois.itertuples()}, start, end)
    decision = build_report(settings, aois, load_reference_features(settings), counts)
    if decision.go:
        typer.echo("GO (pending human review): see outputs/public/coverage_summary.md")
        return
    if decision.needs_review:
        typer.echo(
            f"NEEDS_REVIEW: {len(decision.review_entity_ids)} frame(s) with unknown cloud cover: "
            f"{', '.join(decision.review_entity_ids)}\n"
            "Check browse images (`sbf ingest declass browse`), fill the `usable` column (yes/no) in "
            "outputs/public/browse_review.csv, then re-run `sbf catalog report`.",
            err=True,
        )
        raise typer.Exit(3)
    typer.echo("NO_GO: see outputs/public/NO_GO.md. Stop for human review.", err=True)
    raise typer.Exit(1)


@app.command()
def status() -> None:
    """Show whether M2M credentials are set and whether coverage.csv exists (no network)."""
    import os

    settings = load_settings()
    have = bool(os.environ.get("USGS_USERNAME") and os.environ.get("USGS_M2M_TOKEN"))
    typer.echo(f"M2M credentials in env: {'yes' if have else 'no'}")
    cov = settings.outputs_public / "coverage.csv"
    typer.echo(f"coverage.csv: {'present' if cov.exists() else 'missing'}")
