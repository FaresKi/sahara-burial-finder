"""CLI for the SAR / DEM anomaly detectors. Counts and tile IDs only (D8)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

sar_dem_app = typer.Typer(
    help="SAR and DEM anomaly detectors (sar_z, dem_relief).", no_args_is_help=True
)


@sar_dem_app.command("sar")
def sar(
    aoi: Annotated[str, typer.Option(help="AOI id")],
    layer: Annotated[str, typer.Option(help="e.g. palsar_hv, s1_vv_mean, s1_vh_std")],
    raster: Annotated[Path | None, typer.Option(help="Raster; default from stack.json")] = None,
    size: Annotated[int, typer.Option(help="Tile size in metres")] = 128,
    inner_m: Annotated[float, typer.Option(help="Ring inner radius (m)")] = 30.0,
    outer_m: Annotated[float, typer.Option(help="Ring outer radius (m)")] = 120.0,
    agg: Annotated[str, typer.Option(help="pNN | max | mean | median")] = "p95",
    robust: Annotated[bool, typer.Option(help="median/MAD instead of mean/std")] = False,
    tail: Annotated[str, typer.Option(help="both | high | low")] = "both",
    units: Annotated[
        str | None, typer.Option(help="db | linear (default: manifest, else db)")
    ] = None,
) -> None:
    """Local backscatter z-score detector: write the sar_z score table."""
    from sbf.detect.sar_dem import run_sar_z

    df = run_sar_z(aoi, layer, raster, size, inner_m, outer_m, robust, agg, tail, units)
    typer.echo(f"sar_z/{layer} {aoi} size {size}: scored {len(df)} tiles")


@sar_dem_app.command("dem")
def dem(
    aoi: Annotated[str, typer.Option(help="AOI id")],
    raster: Annotated[Path | None, typer.Option(help="Raster; default from stack.json")] = None,
    layer: Annotated[str, typer.Option(help="dem | dem_roughness")] = "dem",
    size: Annotated[int, typer.Option(help="Tile size in metres")] = 128,
    sigma_m: Annotated[float, typer.Option(help="Smoothing scale (m)")] = 50.0,
    method: Annotated[str, typer.Option(help="gaussian | median")] = "gaussian",
    agg: Annotated[str, typer.Option(help="pNN | max | mean | median")] = "p95",
) -> None:
    """DEM micro-relief (high-pass residual) detector: write the dem_relief score table."""
    from sbf.detect.sar_dem import GLO30_NOTE, run_dem_relief

    typer.echo(f"note: {GLO30_NOTE}", err=True)
    df = run_dem_relief(aoi, raster, layer, size, sigma_m, method, agg)
    typer.echo(f"dem_relief/{layer} {aoi} size {size}: scored {len(df)} tiles")
