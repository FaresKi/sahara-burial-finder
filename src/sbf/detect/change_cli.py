"""CLI for change detection (mounted as ``sbf detect change ...``). Names and counts only (D8)."""

from __future__ import annotations

from typing import Annotated

import typer

change_app = typer.Typer(help="Change detection between dated frames.", no_args_is_help=True)


def _pairs(aoi: str, pair: str | None, same_era: bool, min_overlap_km2: float):
    from sbf.core.config import load_settings
    from sbf.detect.change import PairPolicy, load_modern_layers, select_pairs
    from sbf.georef.registry import load_registry

    s = load_settings()
    try:
        modern = load_modern_layers(aoi, s)
    except FileNotFoundError:
        modern = {}
    try:
        reg = load_registry(True, s)
    except FileNotFoundError as e:
        raise typer.BadParameter(str(e)) from None
    policy = PairPolicy(min_overlap_m2=min_overlap_km2 * 1e6, same_era=same_era)
    pairs = select_pairs(reg, modern, policy, s, aoi=aoi)
    if pair:
        pairs = [p for p in pairs if p.name == pair]
        if not pairs:
            raise typer.BadParameter(f"pair {pair!r} not among the selected pairs")
    return pairs, s


_AOI = Annotated[str, typer.Option(help="AOI id")]
_PAIR = Annotated[str | None, typer.Option(help="Only this pair layer name")]
_SAME = Annotated[bool, typer.Option("--same-era", help="Also pair frames of the same era")]
_MIN = Annotated[float, typer.Option(help="Minimum footprint overlap (km2)")]


@change_app.command("pairs")
def pairs_cmd(aoi: _AOI, same_era: _SAME = False, min_overlap_km2: _MIN = 0.25) -> None:
    """List the selected pairs (layer names, eras, overlap area; no coordinates)."""
    pairs, _ = _pairs(aoi, None, same_era, min_overlap_km2)
    for p in pairs:
        typer.echo(f"{p.name}\t{p.a.era}->{p.b.era}\toverlap {p.overlap_m2 / 1e6:.2f} km2")
    typer.echo(f"{len(pairs)} pairs")


@change_app.command("diff")
def diff(
    aoi: _AOI,
    pair: _PAIR = None,
    size: Annotated[int, typer.Option(help="Tile size in metres")] = 64,
    same_era: _SAME = False,
    min_overlap_km2: _MIN = 0.25,
) -> None:
    """Image-difference change scores (shift-tolerant SSIM + abs diff)."""
    from sbf.detect.change import run_change, stack_unstable_provider

    pairs, s = _pairs(aoi, pair, same_era, min_overlap_km2)
    res = run_change(
        aoi,
        pairs,
        do_embed=False,
        diff_size=size,
        settings=s,
        unstable_provider=stack_unstable_provider(aoi, s),
    )
    for k, v in res.items():
        typer.echo(f"change_diff/{k} {aoi} size {size}: scored {v['diff']} tiles")


@change_app.command("embed")
def embed(
    aoi: _AOI,
    pair: _PAIR = None,
    size: Annotated[int, typer.Option(help="Tile size in metres")] = 128,
    encoder: Annotated[
        str, typer.Option(help="dinov2-small | dinov2-base | HF id")
    ] = "dinov2-small",
    preprocess: Annotated[str, typer.Option(help="none | clahe | lcn")] = "clahe",
    same_era: _SAME = False,
    min_overlap_km2: _MIN = 0.25,
) -> None:
    """Embedding cosine-distance change scores (downloads DINOv2 on first use)."""
    from sbf.detect.change import run_change, stack_unstable_provider
    from sbf.features.embeddings import DinoV2Encoder

    pairs, s = _pairs(aoi, pair, same_era, min_overlap_km2)
    enc = DinoV2Encoder(encoder, preprocess=preprocess)
    res = run_change(
        aoi,
        pairs,
        do_diff=False,
        embed_size=size,
        encoder=enc,
        settings=s,
        unstable_provider=stack_unstable_provider(aoi, s),
    )
    for k, v in res.items():
        typer.echo(f"change_embed/{k} {aoi} size {size}: scored {v['embed']} tiles")


@change_app.command("run")
def run(
    aoi: _AOI,
    diff_size: Annotated[int, typer.Option(help="Diff tile size (m)")] = 64,
    embed_size: Annotated[int, typer.Option(help="Embed tile size (m)")] = 128,
    encoder: Annotated[
        str, typer.Option(help="dinov2-small | dinov2-base | HF id")
    ] = "dinov2-small",
    same_era: _SAME = False,
    min_overlap_km2: _MIN = 0.25,
) -> None:
    """All selected pairs: change_diff and change_embed."""
    from sbf.detect.change import run_change, stack_unstable_provider
    from sbf.features.embeddings import DinoV2Encoder

    pairs, s = _pairs(aoi, None, same_era, min_overlap_km2)
    res = run_change(
        aoi,
        pairs,
        diff_size=diff_size,
        embed_size=embed_size,
        encoder=DinoV2Encoder(encoder),
        settings=s,
        unstable_provider=stack_unstable_provider(aoi, s),
    )
    for k, v in res.items():
        typer.echo(f"{k}: diff {v.get('diff', 0)} tiles, embed {v.get('embed', 0)} tiles")
    typer.echo(f"{len(res)} pairs scored")
