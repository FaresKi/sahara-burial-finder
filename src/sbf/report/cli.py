"""CLI sub-app for sbf.report (review sheets are restricted; public stats are aggregate only)."""

from __future__ import annotations

import typer

app = typer.Typer(help="Reports and figures (public outputs only).", no_args_is_help=True)

GATE_MSG = (
    "SPEC requires human sign-off before producing restricted outputs from real data. "
    "Re-run with --confirm-restricted once the human has approved."
)


@app.command()
def status() -> None:
    """Placeholder command: not implemented yet."""
    typer.echo("sbf report: not implemented yet")


@app.command("sheets")
def sheets(
    aoi: str = typer.Option(..., "--aoi", help="AOI id (REG or INE)."),
    top: int = typer.Option(20, "--top", help="Number of top candidates to render."),
    blind_test_passed: bool | None = typer.Option(
        None,
        "--blind-test-passed/--blind-test-failed",
        help="Override the Phase 4 blind-test outcome (default: read "
        "outputs/public/blind_test_gate.json; absent -> failed, confidence capped low).",
    ),
    confirm_restricted: bool = typer.Option(
        False, "--confirm-restricted", help="Human sign-off given for restricted outputs."
    ),
) -> None:
    """Write restricted top-N review sheets, index and confidence table for an AOI."""
    if not confirm_restricted:
        typer.echo(GATE_MSG, err=True)
        raise typer.Exit(2)
    import geopandas as gpd

    from sbf.core.config import load_settings
    from sbf.core.restricted import restricted_path
    from sbf.report.sheets import key_warning, write_sheets

    try:
        from sbf.georef import load_registry
    except ImportError:
        typer.echo("sbf.georef.load_registry is not available yet.", err=True)
        raise typer.Exit(1) from None

    from sbf.report.confidence import resolve_blind_test

    s = load_settings()
    gpkg = restricted_path(f"candidates_{aoi}.gpkg", settings=s)
    if not gpkg.exists():
        typer.echo(f"No candidates for {aoi}; run `sbf fuse run` first.", err=True)
        raise typer.Exit(1)
    passed, source = resolve_blind_test(blind_test_passed, s.outputs_public)
    typer.echo(f"blind-test gate: {'PASSED' if passed else 'FAILED'} (source: {source})")
    ranked = gpd.read_file(gpkg, layer="candidates")
    paths = write_sheets(
        ranked,
        load_registry(included_only=True),
        aoi,
        str(ranked.crs),
        top_n=top,
        blind_test_passed=passed,
        settings=s,
    )
    typer.echo(f"{aoi}: {len(paths)} restricted files written (counts only).")
    if (w := key_warning()) is not None:
        typer.echo(w, err=True)


@app.command("public-stats")
def public_stats() -> None:
    """Write outputs/public/candidates_public_stats.md (counts only)."""
    from sbf.report.sheets import write_public_stats

    out = write_public_stats()
    if out is None:
        typer.echo("No candidate confidence tables; run `sbf report sheets` first.", err=True)
        raise typer.Exit(1)
    typer.echo(f"wrote {out.name}")


@app.command("blind-test")
def blind_test(
    aoi: list[str] = typer.Option(None, "--aoi", help="Restrict to AOI id(s)."),  # noqa: B008
    no_figures: bool = typer.Option(False, "--no-figures", help="Skip figures."),
) -> None:
    """Evaluate all score tables; write blind_test.md, gate json, figures. Exit 0/4/5."""
    from sbf.report.blind_test import STOP_NOTICE, run_blind_test

    res = run_blind_test(list(aoi) if aoi else None, figures=not no_figures)
    typer.echo(f"blind-test gate: {res.status}")
    if res.status == "FAIL":
        typer.echo(STOP_NOTICE, err=True)
        raise typer.Exit(4)
    if res.status == "NOT_RUN":
        typer.echo("No historical score tables found; gate not run.", err=True)
        raise typer.Exit(5)


@app.command("sam-review")
def sam_review(
    aoi: str = typer.Option(..., "--aoi"),
    detector: str = typer.Option(..., "--detector"),
    layer: str = typer.Option(..., "--layer"),
    size: int = typer.Option(128, "--size", help="Tile size in metres."),
    top: int = typer.Option(50, "--top"),
    raster: str = typer.Option(None, "--raster", help="Chip raster (default: modern stack layer)."),
    confirm_restricted: bool = typer.Option(False, "--confirm-restricted"),
) -> None:
    """Write restricted SAM2 review polygons for the top-K tiles of one detector/layer."""
    if not confirm_restricted:
        typer.echo(GATE_MSG, err=True)
        raise typer.Exit(2)
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds

    from sbf.core.config import load_settings
    from sbf.detect.sam_review import Sam2Segmenter, run_sam_review
    from sbf.detect.scores import read_scores

    s = load_settings()
    sc = read_scores(aoi, detector, layer, settings=s)
    sc = sc[sc["size_m"] == size]
    if sc.empty:
        typer.echo("No matching score table.", err=True)
        raise typer.Exit(1)
    path = raster
    if path is None:
        from sbf.ingest.modern.process import read_manifest, stack_dir

        man = read_manifest(aoi, s)
        if man is None or layer not in man["layers"]:
            typer.echo("No raster for this layer; pass --raster.", err=True)
            raise typer.Exit(1)
        path = str(stack_dir(aoi, s) / man["layers"][layer]["file"])
    try:
        seg = Sam2Segmenter()
    except RuntimeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    with rasterio.open(path) as src:

        def read_chip(b):
            a = src.read(1, window=from_bounds(*b, transform=src.transform), boundless=True,
                         fill_value=np.nan)
            return a.astype("float32")

        out = run_sam_review(sc, seg, read_chip, aoi, detector, layer, size, top, s)
    typer.echo("sam-review: 1 restricted file written." if out else "sam-review: no polygons.")


@app.command("writeup")
def writeup(
    include_restricted: bool = typer.Option(False, "--include-restricted"),
    confirm_restricted: bool = typer.Option(False, "--confirm-restricted"),
    top: int = typer.Option(20, "--top", help="Candidates per AOI in the restricted briefing."),
) -> None:
    """Write outputs/public/report.md from artefacts; the restricted briefing needs both flags."""
    if include_restricted != confirm_restricted:
        typer.echo(GATE_MSG, err=True)
        raise typer.Exit(2)
    from sbf.core.config import load_settings
    from sbf.report.writeup import write_briefing, write_public_report

    s = load_settings()
    typer.echo(f"wrote {write_public_report(s).relative_to(s.root)}")
    if include_restricted:
        b = write_briefing(s, top)
        typer.echo("wrote restricted briefing" if b else "no ranking.csv: no briefing written")


@app.command("check-public")
def check_public_cmd() -> None:
    """Fail (exit 6) if outputs/public holds coordinates that are not public references."""
    from sbf.report.public_check import check_public

    bad = check_public()
    for v in bad:
        typer.echo(str(v), err=True)
    if bad:
        typer.echo(f"{len(bad)} possible leak(s); values are not printed.", err=True)
        raise typer.Exit(6)
    typer.echo("outputs/public: no non-public coordinates found")
