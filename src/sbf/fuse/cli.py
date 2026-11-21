"""`sbf fuse`: Phase 6 fusion and candidate ranking (restricted outputs, human-gated)."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from sbf.fuse.families import FAMILIES

app = typer.Typer(
    help="Phase 6: fusion and candidate ranking (restricted outputs).", no_args_is_help=True
)

GATE_MSG = (
    "SPEC requires human sign-off before producing restricted outputs from real data. "
    "Re-run with --confirm-restricted once the human has approved."
)


def load_weights_file(path: Path) -> dict[str, float]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    bad = set(data) - set(FAMILIES)
    if bad:
        raise typer.BadParameter(f"unknown famil(ies) {sorted(bad)}; known: {list(FAMILIES)}")
    return {f: float(data.get(f, 0.0)) for f in FAMILIES}


@app.command("run")
def run(
    aoi: str = typer.Option(..., "--aoi", help="AOI id (REG or INE)."),
    size: int | None = typer.Option(None, "--size", help="Tile size m (default 64 if present)."),
    top_pct: float = typer.Option(1.0, "--top-pct", help="Top percentile of tiles to keep."),
    weights: Path | None = typer.Option(  # noqa: B008
        None, "--weights", help="Manual weights yaml override."
    ),
    mode: str = typer.Option("max", "--mode", help="Per-family layer combination: max|mean."),
    modern_dates: Path | None = typer.Option(  # noqa: B008
        None, "--modern-dates", help="yaml {layer: YYYY-MM-DD} for modern layers."
    ),
    confirm_restricted: bool = typer.Option(
        False, "--confirm-restricted", help="Human sign-off given for restricted outputs."
    ),
) -> None:
    """Fuse scores, build candidate polygons, write restricted outputs + public aggregate stats."""
    if not confirm_restricted:
        typer.echo(GATE_MSG, err=True)
        raise typer.Exit(2)
    from sbf.core.config import load_settings
    from sbf.detect.exclusions import load_exclusions
    from sbf.detect.harness import evaluate
    from sbf.detect.scores import read_scores
    from sbf.fuse.pipeline import run_fusion, write_outputs

    try:
        from sbf.georef import load_registry
    except ImportError:
        typer.echo("sbf.georef.load_registry is not available yet.", err=True)
        raise typer.Exit(1) from None

    s = load_settings()
    scores = read_scores(aoi, settings=s)
    if scores.empty:
        typer.echo(f"No score tables for {aoi}.", err=True)
        raise typer.Exit(1)
    registry = load_registry(included_only=True)
    mdates = yaml.safe_load(modern_dates.read_text()) if modern_dates else None
    w = load_weights_file(weights) if weights else None
    metrics = None if w else evaluate(scores, settings=s)
    res = run_fusion(
        aoi,
        scores,
        registry,
        weights=w,
        metrics=metrics,
        weights_source=f"manual override ({weights.name})" if weights else None,
        size_m=size,
        top_pct=top_pct,
        mode=mode,
        modern_dates=mdates,
        exclusions=load_exclusions(s),
        settings=s,
    )
    params = {"top_pct": top_pct, "mode": mode, "size_m": res.size_m}
    write_outputs(res, params, s)
    typer.echo(f"{aoi}: {len(res.ranked)} candidates written to outputs/restricted (counts only).")


@app.command("families")
def families() -> None:
    """List fusion families."""
    typer.echo("\n".join(FAMILIES))
