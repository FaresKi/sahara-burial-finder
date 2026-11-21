"""`sbf ingest refine` commands."""

# ruff: noqa: B008

from __future__ import annotations

from pathlib import Path

import typer

refine_app = typer.Typer(
    help="Refine reference features from imagery (human-in-the-loop; never invents coordinates).",
    no_args_is_help=True,
)


@refine_app.command("chip")
def chip(
    feature: str = typer.Option(..., "--feature"),
    raster: Path = typer.Option(..., "--raster"),
    out: Path = typer.Option(None, "--out"),
    half_size_m: float = typer.Option(1500.0, "--half-size-m"),
    center_lon: float = typer.Option(None, "--center-lon", help="Human-supplied (BASE)"),
    center_lat: float = typer.Option(None, "--center-lat", help="Human-supplied (BASE)"),
) -> None:
    """Render an inspection chip around a feature for digitising."""
    from sbf.core.config import load_settings
    from sbf.ingest.refine import RefinementError, export_feature_chip

    if (center_lon is None) != (center_lat is None):
        typer.echo("--center-lon and --center-lat go together", err=True)
        raise typer.Exit(2)
    out = out or load_settings().root / "outputs" / "public" / f"refine_chip_{feature}.png"
    try:
        p = export_feature_chip(
            feature,
            raster,
            out,
            half_size_m,
            (center_lon, center_lat) if center_lon is not None else None,
        )
    except RefinementError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e
    typer.echo(f"wrote {p}")


@refine_app.command("import")
def import_(
    feature: str = typer.Option(..., "--feature"),
    file: Path = typer.Option(..., "--file", help="Human-digitised polygon (GeoJSON/GPKG)"),
    evidence_raster: Path = typer.Option(..., "--evidence-raster"),
    screenshot: Path = typer.Option(..., "--screenshot"),
    layer: str = typer.Option(None, "--layer"),
    entity_id: str = typer.Option(None, "--entity-id"),
    refined_by: str = typer.Option("human", "--refined-by", help="human|agent (be truthful)"),
    notes: str = typer.Option("", "--notes"),
    force_note: str = typer.Option(None, "--force-note"),
    max_move_m: float = typer.Option(500.0, "--max-move-m"),
) -> None:
    """Record a human-digitised polygon with its evidence."""
    from sbf.ingest.refine import RefinementError, import_polygon

    try:
        f = import_polygon(
            feature,
            file,
            evidence_raster=evidence_raster,
            screenshot=screenshot,
            layer=layer,
            entity_id=entity_id,
            refined_by=refined_by,
            notes=notes,
            force_note=force_note,
            max_move_m=max_move_m,
        )
    except (RefinementError, ValueError) as e:
        typer.echo(f"refused: {e}", err=True)
        raise typer.Exit(2) from e
    typer.echo(f"refined {feature} ({f['properties']['geometry_type']}) by {refined_by}")


@refine_app.command("list")
def list_() -> None:
    """List reference features with precision, refinement status and evidence."""
    from sbf.ingest.refine import list_refinements

    typer.echo("feature_id\tprecision\trefined\tgeometry\tlon\tlat\tevidence")
    for r in list_refinements():
        lon = "" if r["lon"] is None else f"{r['lon']:.5f}"
        lat = "" if r["lat"] is None else f"{r['lat']:.5f}"
        typer.echo(
            f"{r['feature_id']}\t{r['precision']}\t{'yes' if r['refined'] else 'no'}\t"
            f"{r['geometry_type'] or '-'}\t{lon}\t{lat}\t{r['evidence'] or '-'}"
        )
