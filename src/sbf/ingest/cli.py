"""CLI sub-app for sbf.ingest. Sibling sub-apps (e.g. `usgs`) can be mounted next to `modern`."""

from __future__ import annotations

import os
from datetime import date

import typer

_PRODUCT_OPT = typer.Option(None, "--product", help="Limit to product key(s)")
app = typer.Typer(help="Phase 2: downloads, GEE exports, provenance.", no_args_is_help=True)
modern_app = typer.Typer(
    help="Modern baseline stack (GEE export + local build).", no_args_is_help=True
)
app.add_typer(modern_app, name="modern")


def _mount_declass() -> None:
    from sbf.ingest.declass.cli import declass_app

    app.add_typer(declass_app, name="declass", help="Declassified frames: browse, download, orders.")


_mount_declass()


def _mount_refine() -> None:
    from sbf.ingest.refine import refine_app

    app.add_typer(refine_app, name="refine", help="Refine reference features from imagery.")


_mount_refine()


@app.command()
def status() -> None:
    """Placeholder command: not implemented yet."""
    typer.echo("sbf ingest: not implemented yet")


def _check_aoi(aoi: str) -> None:
    from sbf.core.config import load_settings

    known = sorted(load_settings().aois)
    if aoi not in known:
        typer.echo(f"Unknown AOI {aoi!r}; choose one of {known}", err=True)
        raise typer.Exit(2)


@modern_app.command("export")
def modern_export(
    aoi: str = typer.Option(..., "--aoi", help="AOI id (REG or INE)"),
    products: list[str] | None = _PRODUCT_OPT,
    end_date: str = typer.Option(None, "--end-date", help="YYYY-MM-DD end of the 2-year window"),
    verify_only: bool = typer.Option(False, "--verify-only", help="Only check dataset IDs"),
) -> None:
    """Export raw GEE products for an AOI (needs GEE_PROJECT and `earthengine authenticate`)."""
    from sbf.core.config import load_settings
    from sbf.ingest.modern import EarthEngineExporter, GeeNotConfiguredError, modern_grid
    from sbf.ingest.modern.layers import PRODUCTS
    from sbf.ingest.modern.process import raw_dir

    _check_aoi(aoi)
    if not os.environ.get("GEE_PROJECT"):
        typer.echo(
            "GEE_PROJECT is not set. Live export is a human gate: run `earthengine authenticate`, "
            "export GEE_PROJECT=<your-cloud-project>, then retry. See notes/PHASE_2_NOTES.md.",
            err=True,
        )
        raise typer.Exit(2)
    bad = set(products or []) - set(PRODUCTS)
    if bad:
        typer.echo(f"Unknown product(s) {sorted(bad)}; known: {sorted(PRODUCTS)}", err=True)
        raise typer.Exit(2)
    s = load_settings()
    grid = modern_grid(aoi, s)
    exp = EarthEngineExporter(end_date=date.fromisoformat(end_date) if end_date else None)
    try:
        if verify_only:
            for c in exp.verify_datasets(grid, products or None):
                typer.echo(
                    f"{c.product}: {'OK ' + str(c.dataset_id) if c.ok else 'FAILED ' + c.detail}"
                )
            return
        report = exp.export_aoi(aoi, raw_dir(aoi, s), grid=grid, products=products or None)
    except GeeNotConfiguredError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e
    for r in report.results:
        typer.echo(f"{r.product}: {r.status}" + (f" ({r.error})" if r.error else ""))
    if report.failed:
        raise typer.Exit(1)


@modern_app.command("build-stack")
def modern_build_stack(
    aoi: str = typer.Option(..., "--aoi", help="AOI id (REG or INE)"),
) -> None:
    """Co-register raw exports onto the common 10 m grid and write COGs + stack.json."""
    from sbf.ingest.modern import build_stack

    _check_aoi(aoi)
    m = build_stack(aoi)
    typer.echo(f"{aoi}: {len(m['layers'])} layers written, {len(m['missing'])} missing")
    for name, why in m["missing"].items():
        typer.echo(f"  missing {name}: {why}")
