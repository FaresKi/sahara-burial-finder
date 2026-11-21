"""Declarative stage runner for `sbf run-all` (ticket 18).

A `Stage` is (name, run function, input globs, output globs, optional gate). The runner decides
per stage: run, `up-to-date` (input fingerprint recorded in `data/interim/run_manifest.json`
matches and outputs exist), or `skipped (missing input: ...)`. Nothing is ever fabricated: a stage
without its inputs is reported as skipped and downstream artefacts say so.

Exit codes of `run_pipeline`: 0 ok, 1 a stage failed, 3 gate stop NEEDS_REVIEW, 4 gate stop NO_GO,
6 public leak check failed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sbf.core.config import Settings

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NEEDS_REVIEW = 3
EXIT_NO_GO = 4
EXIT_LEAK = 6

NOT_VALID = "NOT VALID: gates bypassed"
MANIFEST_NAME = "run_manifest.json"


@dataclass
class Ctx:
    settings: Settings
    include_restricted: bool = False
    bypass_gates: bool = False
    same_era: bool = False
    skip: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    @property
    def root(self) -> Path:
        return self.settings.root

    def params(self) -> dict:
        return {
            "include_restricted": self.include_restricted,
            "bypass_gates": self.bypass_gates,
            "same_era": self.same_era,
        }


@dataclass
class Outcome:
    """`status`: ok | skipped | failed | gate_stop. `exit_code` only matters for failed/gate_stop."""

    status: str
    detail: str = ""
    exit_code: int = EXIT_OK

    @classmethod
    def ok(cls, detail: str = "") -> Outcome:
        return cls("ok", detail)

    @classmethod
    def skipped(cls, reason: str) -> Outcome:
        return cls("skipped", reason)

    @classmethod
    def failed(cls, detail: str, code: int = EXIT_FAILED) -> Outcome:
        return cls("failed", detail, code)


Globs = Callable[[Ctx], list[str]]


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    run: Callable[[Ctx], Outcome]
    inputs: Globs = lambda ctx: []
    outputs: Globs = lambda ctx: []
    # returns "missing input: ..." text when the stage cannot run at all
    needs: Callable[[Ctx], str | None] = lambda ctx: None
    gate: str | None = None  # human-readable gate this stage enforces or sits behind
    restricted: bool = False  # only with --include-restricted --confirm-restricted
    always_run: bool = False  # never cached (e.g. the leak check)
    after_go_gate: bool = True  # blocked when the catalogue decision is not GO


def expand(ctx: Ctx, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        for p in sorted(ctx.root.glob(pat)):
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                files.extend(sorted(q for q in p.rglob("*") if q.is_file()))
    return files


def fingerprint(ctx: Ctx, stage: Stage) -> str:
    """Hash of input file identities (relative path, size, mtime) and run parameters."""
    h = hashlib.sha256()
    h.update(json.dumps(ctx.params(), sort_keys=True).encode())
    for p in expand(ctx, stage.inputs(ctx)):
        st = p.stat()
        h.update(f"{p.relative_to(ctx.root)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def outputs_present(ctx: Ctx, stage: Stage) -> bool:
    pats = stage.outputs(ctx)
    return bool(pats) and all(expand(ctx, [p]) for p in pats)


def manifest_path(ctx: Ctx) -> Path:
    return ctx.settings.data_interim / MANIFEST_NAME


def read_manifest(ctx: Ctx) -> dict:
    p = manifest_path(ctx)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {"stages": {}}
    return {"stages": {}}


def write_manifest(ctx: Ctx, man: dict) -> None:
    p = manifest_path(ctx)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(man, indent=2, default=str))


def select(stages: list[Stage], from_stage: str | None, only: str | None) -> list[Stage]:
    names = [s.name for s in stages]
    for n in (from_stage, only):
        if n is not None and n not in names:
            raise KeyError(f"unknown stage {n!r}; stages: {', '.join(names)}")
    if only:
        return [s for s in stages if s.name == only]
    if from_stage:
        return stages[names.index(from_stage) :]
    return list(stages)


def plan_status(ctx: Ctx, stage: Stage, man: dict, force: bool) -> tuple[str, str]:
    """('run'|'up-to-date'|'skipped', detail) without running anything."""
    if stage.name in ctx.skip:
        return "skipped", "skipped by request (--skip)"
    if stage.restricted and not ctx.include_restricted:
        return "skipped", "restricted stage off (needs --include-restricted --confirm-restricted)"
    miss = stage.needs(ctx)
    if miss:
        return "skipped", f"skipped (missing input: {miss})"
    if stage.always_run or force:
        return "run", "forced" if force else "always runs"
    rec = man.get("stages", {}).get(stage.name)
    if (
        rec
        and rec.get("status") == "ok"
        and rec.get("fingerprint") == fingerprint(ctx, stage)
        and outputs_present(ctx, stage)
    ):
        return "up-to-date", "inputs unchanged since last run"
    return "run", "inputs changed or no previous run"


def go_gate_state(ctx: Ctx) -> tuple[str | None, str | None]:
    """(status, detail) from outputs/public/go_decision.json, or (None, None) if absent."""
    p = ctx.settings.outputs_public / "go_decision.json"
    if not p.exists():
        return None, None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None, None
    return d.get("status"), "; ".join(d.get("reasons", []))


def mark_not_valid(ctx: Ctx) -> None:
    """Stamp public outputs when gates were bypassed (debugging only)."""
    pub = ctx.settings.outputs_public
    pub.mkdir(parents=True, exist_ok=True)
    (pub / "NOT_VALID.txt").write_text(
        f"{NOT_VALID}\n\nThis output set was produced with --continue-past-gates. The Phase 1 "
        "go/no-go gate (or another gate) was not satisfied. Do not use for conclusions.\n"
    )


def clear_not_valid(ctx: Ctx) -> None:
    p = ctx.settings.outputs_public / "NOT_VALID.txt"
    if p.exists():
        p.unlink()


@dataclass
class StageResult:
    name: str
    status: str  # ok | up-to-date | skipped | failed | gate_stop | not_run
    detail: str = ""
    seconds: float = 0.0


def run_pipeline(
    ctx: Ctx,
    stages: list[Stage],
    *,
    from_stage: str | None = None,
    only: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    echo: Callable[[str], None] = print,
) -> tuple[int, list[StageResult]]:
    chosen = select(stages, from_stage, only)
    man = read_manifest(ctx)
    results: list[StageResult] = []

    if dry_run:
        upstream_runs = False
        for st in chosen:
            kind, detail = plan_status(ctx, st, man, force)
            if kind == "skipped" and "missing input" in detail and upstream_runs:
                detail = f"undetermined until upstream stages run (now: {detail})"
            upstream_runs = upstream_runs or kind == "run"
            gate = f" [gate: {st.gate}]" if st.gate else ""
            echo(f"{st.name:<18} {kind:<10} {detail}{gate}")
            results.append(StageResult(st.name, kind, detail))
        return EXIT_OK, results

    if ctx.bypass_gates:
        mark_not_valid(ctx)
    else:
        clear_not_valid(ctx)
    exit_code = EXIT_OK
    stopped_at: str | None = None
    for st in chosen:
        if stopped_at:
            results.append(StageResult(st.name, "not_run", f"stopped at gate in {stopped_at}"))
            continue
        kind, detail = plan_status(ctx, st, man, force)
        if kind == "skipped":
            results.append(StageResult(st.name, "skipped", detail))
            echo(f"[{st.name}] {detail}")
            _record(ctx, man, st, "skipped", detail, None)
            continue
        if kind == "up-to-date":
            results.append(StageResult(st.name, "up-to-date", detail))
            echo(f"[{st.name}] up-to-date")
            continue
        if st.after_go_gate and not ctx.bypass_gates:
            status, why = go_gate_state(ctx)
            if status is not None and status != "GO":
                code = EXIT_NEEDS_REVIEW if status == "NEEDS_REVIEW" else EXIT_NO_GO
                msg = f"gate: catalogue decision is {status} ({why}); stop for human review"
                results.append(StageResult(st.name, "gate_stop", msg))
                echo(f"[{st.name}] {msg}")
                exit_code, stopped_at = code, st.name
                continue
        echo(f"[{st.name}] running")
        t0 = time.time()
        try:
            out = st.run(ctx)
        except Exception as exc:  # noqa: BLE001 - a stage crash is reported, never swallowed
            log.debug(traceback.format_exc())
            out = Outcome.failed(f"{type(exc).__name__}: {exc}")
        dt = time.time() - t0
        if out.status == "gate_stop" and ctx.bypass_gates:
            out = Outcome.ok(f"gate bypassed ({out.detail})")
        results.append(StageResult(st.name, out.status, out.detail, dt))
        echo(f"[{st.name}] {out.status}{': ' + out.detail if out.detail else ''} ({dt:.1f}s)")
        fp = fingerprint(ctx, st) if out.status == "ok" else None
        _record(ctx, man, st, out.status, out.detail, fp)
        if out.status == "failed":
            exit_code = out.exit_code or EXIT_FAILED
            if exit_code == EXIT_LEAK:
                continue  # the leak check is last anyway
            stopped_at = st.name
        elif out.status == "gate_stop":
            exit_code = out.exit_code
            stopped_at = st.name
    man["flags"] = ctx.params()
    man["last_run"] = datetime.now(UTC).isoformat(timespec="seconds")
    man["exit_code"] = exit_code
    write_manifest(ctx, man)
    return exit_code, results


def _record(ctx: Ctx, man: dict, st: Stage, status: str, detail: str, fp: str | None) -> None:
    man.setdefault("stages", {})[st.name] = {
        "status": status,
        "detail": detail,
        "fingerprint": fp,
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_manifest(ctx, man)  # per stage: the write-up reads skip reasons of this run
