"""`sbf ingest declass ...` (mounted by the architect): browse, availability, download, order-list.

Nothing here places an order. The order list is a human purchase gate (SPEC section 0.1).
"""

from __future__ import annotations

import json
from typing import Annotated

import typer

declass_app = typer.Typer(
    help="Declass acquisition: browse images, free frames, order list (never orders).",
    no_args_is_help=True,
)


def _session():  # indirection so tests can inject a fake HTTP session
    import requests

    return requests.Session()


def _client(settings):
    from sbf.catalog.m2m import M2MClient, M2MError, RawStore

    try:
        M2MClient.credentials()
    except M2MError as exc:
        typer.echo(f"{exc}. Human gate: see notes/PHASE_1_NOTES.md.", err=True)
        raise typer.Exit(2) from exc
    return M2MClient(RawStore(settings.data_raw / "catalog"))


def _coverage(settings):
    from sbf.catalog.report import read_coverage

    path = settings.outputs_public / "coverage.csv"
    if not path.exists():
        typer.echo("outputs/public/coverage.csv missing: run `sbf catalog search` first", err=True)
        raise typer.Exit(2)
    return read_coverage(path)


def _save_failures(settings, name: str, failures: list[dict]) -> None:
    out = settings.data_interim / "declass"
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(failures, indent=2))


@declass_app.command()
def browse() -> None:
    """Download browse images for ALL coverage rows to data/raw/declass/{entity_id}/browse.<ext>.

    Use them for the browse review behind `outputs/public/browse_review.csv`.
    """
    from sbf.core.config import load_settings
    from sbf.ingest.declass.acquire import download_browse

    settings = load_settings()
    rep = download_browse(_coverage(settings), settings.data_raw / "declass", _session())
    typer.echo(f"browse: {len(rep.done)} downloaded, {len(rep.skipped)} already present")
    _save_failures(settings, "browse_failures.json", [f.as_dict() for f in rep.failures])
    if rep.failures:
        typer.echo(
            f"{len(rep.failures)} failure(s) in data/interim/declass/browse_failures.json", err=True
        )
        raise typer.Exit(1)


@declass_app.command()
def availability() -> None:
    """Refine coverage.csv availability with M2M `download-options` (needs credentials)."""
    from sbf.catalog.download import refine_availability
    from sbf.core.config import load_settings

    settings = load_settings()
    cov = _coverage(settings)
    new, failures = refine_availability(_client(settings), cov)
    new.to_csv(settings.outputs_public / "coverage.csv", index=False)
    _save_failures(settings, "availability_failures.json", failures)
    n = int((new["availability_source"] == "download-options").sum())
    typer.echo(f"availability set from download-options for {n}/{len(new)} frames")
    if failures:
        typer.echo(f"{len(failures)} download-options failure(s) recorded", err=True)
        raise typer.Exit(1)


@declass_app.command()
def download(
    feature: Annotated[
        list[str] | None, typer.Option("--feature", help="Reference feature id; repeatable")
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="List frames; no network")] = False,
) -> None:
    """Download free (scanned) frames intersecting reference features, highest-res product.

    Resumable (.part + HTTP Range), size/checksum verified when the API gives one, with a
    `.prov.json` per file; archives are extracted safely. Needs M2M credentials unless --dry-run.
    """
    from sbf.core.aoi import load_reference_features
    from sbf.core.config import load_settings
    from sbf.ingest.declass.acquire import download_free_frames, select_free_frames

    settings = load_settings()
    known = set(load_reference_features(settings)["feature_id"])
    bad = sorted(set(feature or []) - known)
    if bad:
        typer.echo(f"unknown feature(s) {bad}; known: {sorted(known)}", err=True)
        raise typer.Exit(2)
    frames = select_free_frames(_coverage(settings), feature)
    if dry_run:
        for r in frames.itertuples():
            typer.echo(
                f"{r.entity_id}\t{r.dataset}\t{r.acquisition_date}\t{r.intersecting_features}"
            )
        typer.echo(f"dry run: {len(frames)} free frame(s) would be downloaded; nothing fetched")
        return
    rep = download_free_frames(_client(settings), _session(), frames, settings.data_raw / "declass")
    typer.echo(
        f"download: {len(rep.done)} done, {len(rep.skipped)} skipped, {len(rep.failures)} failed"
    )
    _save_failures(settings, "download_failures.json", [f.as_dict() for f in rep.failures])
    if rep.failures:
        for f in rep.failures:
            typer.echo(f"  {f.entity_id} [{f.stage}]: {f.message}", err=True)
        raise typer.Exit(1)


@declass_app.command("order-list")
def order_list() -> None:
    """Write outputs/public/order_list.csv: unscanned frames ranked by value, with cost.

    This NEVER places an order. Purchasing ($30/frame) is a human decision: review the list,
    then order manually through EarthExplorer.
    """
    from sbf.core.config import load_settings
    from sbf.ingest.declass.orders import COST_PER_FRAME_USD, build_order_list, write_order_list

    settings = load_settings()
    df = build_order_list(_coverage(settings))
    out = settings.outputs_public / "order_list.csv"
    total = write_order_list(df, out)
    typer.echo(f"{len(df)} unscanned frame(s) ranked -> {out}")
    typer.echo(f"TOTAL estimated cost: ${total:,.0f} (at ${COST_PER_FRAME_USD:.0f}/frame, SPEC)")
    typer.echo("NO ORDERS PLACED. Purchasing is a human gate: review and approve before ordering.")
