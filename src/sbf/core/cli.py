"""CLI sub-app for sbf.core."""

import typer

from sbf.core.aoi import load_aois, load_reference_features
from sbf.core.config import load_settings

app = typer.Typer(help="Shared config and sanity checks.", no_args_is_help=True)


@app.command()
def status() -> None:
    """Show loaded settings and config summary (reference coordinates only)."""
    s = load_settings()
    typer.echo(f"root: {s.root}")
    typer.echo(f"AOIs: {', '.join(load_aois(s)['aoi_id'])}")
    typer.echo(f"reference features: {', '.join(load_reference_features(s)['feature_id'])}")
