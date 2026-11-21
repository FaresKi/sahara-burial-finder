"""CLI sub-app for sbf.features. Prints tile IDs/counts only (D8)."""

from __future__ import annotations

from typing import Annotated

import typer

from sbf.core.config import load_settings
from sbf.features.tiling import make_grid

app = typer.Typer(help="Tiling, embeddings, texture metrics.", no_args_is_help=True)


@app.command()
def grid(
    aoi: Annotated[str, typer.Option(help="AOI id, e.g. REG")],
    size: Annotated[int, typer.Option(help="Tile size in m (0: all sizes in settings)")] = 0,
    write: Annotated[bool, typer.Option(help="Write grid to data/interim/grids/")] = False,
) -> None:
    """Build the tile grid(s) for an AOI and print tile counts (never coordinates)."""
    s = load_settings()
    typer.echo(f"AOI {aoi}: origin snapped to 128 m multiples (coordinates not printed)")
    for size_m in [size] if size else s.tile_sizes_m:
        g = make_grid(aoi, size_m, s.overlap, s)
        typer.echo(f"size {size_m} m, overlap {s.overlap}: {len(g)} tiles")
        if write:
            out = s.data_interim / "grids" / f"{aoi}_{size_m}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            g.to_parquet(out)
            typer.echo(f"wrote {out.relative_to(s.root)}")
