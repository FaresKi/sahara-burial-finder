"""`sbf georef ...` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Phase 3: georeferencing of historical frames.", no_args_is_help=True)


@app.callback()
def _group() -> None:
    """Georeferencing commands (more are added by later tickets)."""


@app.command("frame")
def frame(
    entity_id: Annotated[str, typer.Option("--entity-id", help="USGS entity ID of the frame")],
    raster: Annotated[
        Path | None, typer.Option(help="raw frame GeoTIFF (default: auto-find)")
    ] = None,
    aoi: Annotated[str | None, typer.Option(help="REG or INE (default: from coverage.csv)")] = None,
    method: Annotated[str, typer.Option(help="auto|poly1|poly2|poly3|tps")] = "auto",
    reference: Annotated[
        Path | None,
        typer.Option(help="licensed VHR GeoTIFF to match against instead of the S2 stack"),
    ] = None,
    reference_layer: Annotated[str, typer.Option(help="pan|B2|B3|B4|B8|hillshade")] = "pan",
    seed: Annotated[int, typer.Option(help="seed for the checkpoint split and RANSAC")] = 0,
    strips: Annotated[
        bool | None,
        typer.Option("--strips/--no-strips", help="per-strip solve (default: on for panoramic)"),
    ] = None,
    n_strips: Annotated[int, typer.Option(help="number of strips for panoramic frames")] = 8,
    strip_overlap: Annotated[float, typer.Option(help="strip overlap, fraction of a strip")] = 0.1,
    strip_axis: Annotated[str, typer.Option(help="auto|x|y: film axis that is split")] = "auto",
) -> None:
    """Auto-georeference one frame (multi-part scans assembled; panoramic -> strips)."""
    from sbf.georef.fit import METHODS
    from sbf.georef.pipeline import run_frame

    if method != "auto" and method not in METHODS:
        raise typer.BadParameter(f"method must be auto or one of {METHODS}")
    try:
        rep = run_frame(
            entity_id,
            raster=raster,
            aoi=aoi,
            method=method,
            reference_path=reference,
            reference_mode=reference_layer,
            seed=seed,
            strips=strips,
            n_strips=n_strips,
            strip_overlap=strip_overlap,
            strip_axis=strip_axis,
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(
        f"{entity_id}: method={rep.method} gcps={rep.n_gcps} "
        f"cv_rmse_m={rep.cv_rmse_m} checkpoint_rmse_m={rep.checkpoint_rmse_m} included={rep.included_for_change_detection} "
        f"warnings={rep.warnings}"
    )
    if rep.method == "none":
        raise typer.Exit(2)


@app.command("qgis-export")
def qgis_export(
    entity_id: Annotated[str, typer.Option("--entity-id", help="USGS entity ID of the frame")],
    aoi: Annotated[str | None, typer.Option(help="REG or INE")] = None,
) -> None:
    """Write a QGIS Georeferencer .points file (seeded with automatic GCPs) + README."""
    from sbf.georef.qgis import export_qgis

    try:
        r = export_qgis(entity_id, aoi=aoi)
    except (FileNotFoundError, KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"seed points ({r['n_seed']}): {r['points']}\nREADME: {r['readme']}")


@app.command("qgis-import")
def qgis_import(
    entity_id: Annotated[str, typer.Option("--entity-id", help="USGS entity ID of the frame")],
    points: Annotated[Path, typer.Option("--points", help="QGIS .points file saved by the human")],
    method: Annotated[str, typer.Option(help="auto|poly1|poly2|poly3|tps")] = "auto",
    seed: Annotated[int, typer.Option(help="seed for the checkpoint split")] = 0,
    aoi: Annotated[str | None, typer.Option(help="REG or INE")] = None,
) -> None:
    """Read manual GCPs back; refit + warp + report with method manual_gcps."""
    from sbf.georef.qgis import import_qgis

    try:
        rep = import_qgis(entity_id, points, method=method, seed=seed, aoi=aoi)
    except (FileNotFoundError, KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(
        f"{entity_id}: manual_gcps fit={rep.method} gcps={rep.n_gcps} "
        f"cv_rmse_m={rep.cv_rmse_m} checkpoint_rmse_m={rep.checkpoint_rmse_m} included={rep.included_for_change_detection}"
    )
    if rep.method == "none":
        raise typer.Exit(2)


@app.command("batch")
def batch(
    aoi: Annotated[str | None, typer.Option(help="only frames of this AOI (REG|INE)")] = None,
    only_missing: Annotated[
        bool, typer.Option("--only-missing", help="skip frames that already have a report")
    ] = False,
    method: Annotated[str, typer.Option(help="auto|poly1|poly2|poly3|tps")] = "auto",
    seed: Annotated[int, typer.Option(help="seed for the checkpoint split and RANSAC")] = 0,
) -> None:
    """Georeference every downloaded frame; write registry.parquet and georef_summary.md."""
    from sbf.georef.registry import registry_path, run_batch, summary_path

    df = run_batch(aoi=aoi, only_missing=only_missing, method=method, seed=seed)
    inc = int(df["included_for_change_detection"].sum())
    bad = int((df["status"] != "ok").sum())
    typer.echo(f"{len(df)} frames: {inc} included, {bad} failed/unsolved")
    typer.echo(f"registry: {registry_path()}\nsummary: {summary_path()}")
