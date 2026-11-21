"""CLI sub-app for sbf.detect. Prints metrics, tile IDs and counts only (D8)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from sbf.core.config import load_settings
from sbf.detect.change_cli import change_app
from sbf.detect.embedding_cli import embed_app
from sbf.detect.sar_dem_cli import sar_dem_app

app = typer.Typer(help="Phase 4: anomaly and change detection, scoring.", no_args_is_help=True)
app.add_typer(embed_app, name="embedding", help="DINOv2 embedding anomaly detector.")
app.add_typer(change_app, name="change", help="Change detection between dated frames.")
app.add_typer(sar_dem_app, name="sar-dem", help="SAR backscatter and DEM micro-relief anomalies.")


@app.command()
def texture(
    aoi: Annotated[str, typer.Option(help="AOI id")],
    layer: Annotated[str, typer.Option(help="Layer name, e.g. s2_median")],
    raster: Annotated[Path, typer.Option(exists=True, help="North-up raster in the AOI CRS")],
    size: Annotated[int, typer.Option(help="Tile size in metres")] = 64,
    method: Annotated[str, typer.Option(help="iforest | knn | maha")] = "iforest",
    preprocess: Annotated[str, typer.Option(help="none | clahe | lcn")] = "none",
) -> None:
    """Texture-baseline detector: write the score table for one layer raster."""
    from sbf.detect.texture_detector import run_texture_detector

    df = run_texture_detector(aoi, layer, raster, size, method, preprocess=preprocess)
    typer.echo(f"texture/{layer} {aoi} size {size}: scored {len(df)} tiles")


@app.command()
def random(
    aoi: Annotated[str, typer.Option(help="AOI id")],
    layer: Annotated[str, typer.Option(help="Existing texture layer whose tiles are re-scored")],
    size: Annotated[int, typer.Option(help="Tile size in metres")] = 64,
    seed: Annotated[int, typer.Option(help="RNG seed")] = 0,
) -> None:
    """Seeded random baseline over the tiles of an existing texture score table."""
    from sbf.detect.scores import random_scores, read_scores, write_scores

    base = read_scores(aoi, "texture", layer)
    base = base[base["size_m"] == size]
    if base.empty:
        raise typer.BadParameter("no texture scores for that aoi/layer/size; run `detect texture`")
    write_scores(random_scores(base["tile_id"].tolist(), layer, seed))
    typer.echo(f"random/{layer} {aoi} size {size}: scored {len(base)} tiles (seed {seed})")


@app.command()
def score(
    aoi: Annotated[str, typer.Option(help="AOI id")],
    detector: Annotated[str | None, typer.Option(help="Restrict to one detector")] = None,
    layer: Annotated[str | None, typer.Option(help="Restrict to one layer")] = None,
    markdown: Annotated[Path | None, typer.Option(help="Also write the table here")] = None,
) -> None:
    """Blind-test stored score tables against reference features; print the metrics table."""
    from sbf.detect.harness import evaluate, to_markdown
    from sbf.detect.scores import read_scores

    scores = read_scores(aoi, detector, layer)
    if scores.empty:
        raise typer.BadParameter("no score tables found; run a detector first")
    table = to_markdown(evaluate(scores, settings=load_settings()))
    typer.echo(table)
    if markdown:
        markdown.write_text(table)


@app.command("draft-exclusions")
def draft_exclusions(
    overwrite: Annotated[bool, typer.Option(help="Replace an existing draft")] = False,
) -> None:
    """Write config/exclusions.geojson as a draft needing human review."""
    from sbf.detect.exclusions import write_draft

    write_draft(overwrite=overwrite)
    typer.echo("wrote draft exclusions (status: draft_needs_human_review); human review required")
