"""Provenance sidecars (SPEC section 0.6): `<raster>.prov.json`."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import sbf
from sbf.core.config import project_root


def script_commit(root: Path | None = None) -> str:
    """Git HEAD sha; 'uncommitted' if no HEAD or src/config are dirty; 'no-git' if not a repo."""
    cwd = str(root or project_root())

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)

    try:
        if git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
            return "no-git"
        head = git("rev-parse", "HEAD")
    except (OSError, FileNotFoundError):
        return "no-git"
    if head.returncode != 0:
        return "uncommitted"
    if git("status", "--porcelain", "--", "src", "config").stdout.strip():
        return "uncommitted"
    return head.stdout.strip()


def prov_path(raster_path: str | Path) -> Path:
    p = Path(raster_path)
    return p.with_name(p.name + ".prov.json")


def write_prov(
    raster_path: str | Path,
    *,
    source: str,
    source_id: str,
    acquisition_date: str | None,
    steps: list[str],
    inputs: list[str] | None = None,
) -> Path:
    """Write the sidecar and return its path."""
    rec = {
        "source": source,
        "source_id": source_id,
        "acquisition_date": acquisition_date,
        "download_date": datetime.now(UTC).isoformat(),
        "processing_steps": list(steps),
        "inputs": [str(i) for i in (inputs or [])],
        "script_commit": script_commit(),
        "sbf_version": sbf.__version__,
    }
    out = prov_path(raster_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    return out


def read_prov(path: str | Path) -> dict:
    """Read a sidecar; `path` may be the raster or the .prov.json itself."""
    p = Path(path)
    if not p.name.endswith(".prov.json"):
        p = prov_path(p)
    return json.loads(p.read_text())


def append_step(path: str | Path, step: str) -> dict:
    """Append a processing step to an existing sidecar and return the updated record."""
    p = Path(path)
    if not p.name.endswith(".prov.json"):
        p = prov_path(p)
    rec = json.loads(p.read_text())
    rec["processing_steps"].append(step)
    p.write_text(json.dumps(rec, indent=2))
    return rec
