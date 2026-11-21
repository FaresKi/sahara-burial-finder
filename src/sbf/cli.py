"""Top-level `sbf` CLI. Mounts each package's sub-app (ARCHITECTURE.md D4)."""

import typer

from sbf.catalog.cli import app as catalog_app
from sbf.core.cli import app as core_app
from sbf.core.config import load_dotenv
from sbf.detect.cli import app as detect_app
from sbf.docs_mining.cli import app as docs_app
from sbf.features.cli import app as features_app
from sbf.fuse.cli import app as fuse_app
from sbf.georef.cli import app as georef_app
from sbf.ingest.cli import app as ingest_app
from sbf.report.cli import app as report_app

load_dotenv()

app = typer.Typer(help="Sahara Burial Finder: desk analysis pipeline.", no_args_is_help=True)

app.add_typer(core_app, name="core")
app.add_typer(catalog_app, name="catalog")
app.add_typer(ingest_app, name="ingest")
app.add_typer(georef_app, name="georef")
app.add_typer(features_app, name="features")
app.add_typer(detect_app, name="detect")
app.add_typer(docs_app, name="docs")
app.add_typer(fuse_app, name="fuse")
app.add_typer(report_app, name="report")


@app.command("run-all")
def run_all(
    from_stage: str | None = typer.Option(None, "--from", help="Start at this stage."),
    only: str | None = typer.Option(None, "--only", help="Run only this stage."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan; run nothing."),
    force: bool = typer.Option(False, "--force", help="Ignore up-to-date checks."),
    skip: list[str] = typer.Option([], "--skip", help="Skip a stage (repeatable)."),  # noqa: B008
    same_era: bool = typer.Option(
        False, "--same-era", help="Change detection also pairs frames of the same era."
    ),
    include_restricted: bool = typer.Option(
        False,
        "--include-restricted",
        help="Also run restricted stages (needs --confirm-restricted).",
    ),
    confirm_restricted: bool = typer.Option(
        False, "--confirm-restricted", help="Human sign-off for restricted outputs."
    ),
    continue_past_gates: bool = typer.Option(
        False,
        "--continue-past-gates",
        help="Debug only: ignore gates; outputs are stamped 'NOT VALID: gates bypassed'.",
    ),
) -> None:
    """Reproduce the public outputs offline from data/raw (catalogue replay, no network).

    Exit codes: 0 ok, 1 stage failed, 2 usage, 3 gate stop NEEDS_REVIEW, 4 gate stop NO_GO,
    6 public leak check failed. Stages: see `sbf run-all --dry-run`.
    """
    import os

    from sbf.core.config import load_settings
    from sbf.pipeline import Ctx, build_stages, run_pipeline

    if include_restricted != confirm_restricted:
        typer.echo(
            "restricted stages need BOTH --include-restricted and --confirm-restricted "
            "(human sign-off, SPEC section 8).",
            err=True,
        )
        raise typer.Exit(2)
    settings = load_settings()
    os.environ["SBF_ROOT"] = str(settings.root)  # sub-commands resolve the same project root
    ctx = Ctx(
        settings,
        include_restricted=include_restricted,
        bypass_gates=continue_past_gates,
        same_era=same_era,
        skip=set(skip),
    )
    try:
        code, results = run_pipeline(
            ctx, build_stages(), from_stage=from_stage, only=only, dry_run=dry_run,
            force=force, echo=typer.echo,
        )  # fmt: skip
    except KeyError as exc:
        typer.echo(str(exc.args[0]), err=True)
        raise typer.Exit(2) from exc
    if not dry_run:
        typer.echo("\nsummary:")
        for r in results:
            typer.echo(f"  {r.name:<18} {r.status:<10} {r.detail}")
        if continue_past_gates:
            typer.echo("NOT VALID: gates bypassed (outputs/public/NOT_VALID.txt)", err=True)
    raise typer.Exit(code)


if __name__ == "__main__":
    app()
