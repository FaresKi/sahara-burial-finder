"""CLI sub-app for the embedding detector (mounted by the architect as ``sbf detect embed ...``).

Prints counts only (D8). Real DINOv2 weights are downloaded from the Hugging Face hub on first use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

embed_app = typer.Typer(help="Embedding anomaly detector (DINOv2).", no_args_is_help=True)


@embed_app.command("embed")
def embed(
    aoi: Annotated[str, typer.Option(help="AOI id")],
    layer: Annotated[str, typer.Option(help="Layer name")],
    raster: Annotated[Path, typer.Option(exists=True, help="North-up raster in the AOI CRS")],
    size: Annotated[
        int, typer.Option(help="Tile size in metres (128 modern, 64 historical)")
    ] = 128,
    encoder: Annotated[
        str, typer.Option(help="dinov2-small | dinov2-base | HF model id")
    ] = "dinov2-small",
    preprocess: Annotated[str, typer.Option(help="none | clahe | lcn")] = "clahe",
    max_tiles: Annotated[int | None, typer.Option(help="Debug: cap new tiles encoded")] = None,
    batch_size: Annotated[int, typer.Option(help="Encoder batch size")] = 32,
    min_valid_frac: Annotated[float, typer.Option(help="Min valid pixel fraction")] = 0.5,
) -> None:
    """Compute and cache tile embeddings (resumable)."""
    from sbf.detect.embedding import compute_embeddings
    from sbf.features.embeddings import DinoV2Encoder

    enc = DinoV2Encoder(encoder, preprocess=preprocess, batch_size=batch_size)
    r = compute_embeddings(
        aoi,
        layer,
        raster,
        enc,
        size,
        min_valid_frac=min_valid_frac,
        max_tiles=max_tiles,
        batch_size=batch_size,
    )
    typer.echo(
        f"embed/{layer} {aoi} size {size} [{enc.name}]: encoded {r['encoded']} new, "
        f"{r['cached_total']} cached, {r['grid']} tiles in footprint"
    )


@embed_app.command("score")
def score(
    aoi: Annotated[str, typer.Option(help="AOI id")],
    layer: Annotated[str, typer.Option(help="Layer name")],
    size: Annotated[int, typer.Option(help="Tile size in metres")] = 128,
    encoder: Annotated[
        str, typer.Option(help="Cache name: <model>_<preprocess>")
    ] = "dinov2-small_clahe",
    methods: Annotated[
        str, typer.Option(help="Comma list of iforest,knn,maha")
    ] = "iforest,knn,maha",
    pca: Annotated[int, typer.Option(help="PCA components before fitting (0 = off)")] = 64,
    seed: Annotated[int, typer.Option(help="RNG seed")] = 0,
) -> None:
    """Fit background models on cached embeddings and write embed_* score tables."""
    from sbf.detect.embedding import run_embedding_scores

    try:
        df = run_embedding_scores(aoi, layer, size, encoder, methods.split(","), pca or None, seed)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(
        f"embed scores/{layer} {aoi} size {size}: {df['tile_id'].nunique()} tiles, "
        f"detectors {sorted(df['detector'].unique())}"
    )
