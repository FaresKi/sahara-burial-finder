"""Guarded access to outputs/restricted (SPEC section 0.1)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from sbf.core.config import Settings, load_settings

log = logging.getLogger(__name__)


class RestrictedPathError(ValueError):
    """Raised when a path would resolve outside outputs/restricted."""


def restricted_path(*parts: str | Path, settings: Settings | None = None) -> Path:
    """Path under outputs/restricted; creates parent dirs, refuses anything resolving elsewhere."""
    base = (settings or load_settings()).outputs_restricted.resolve()
    target = base.joinpath(*map(Path, parts)).resolve()
    if target != base and not target.is_relative_to(base):
        raise RestrictedPathError(f"path resolves outside outputs/restricted: {target}")
    (target if not target.suffix else target.parent).mkdir(parents=True, exist_ok=True)
    return target


def encrypt_if_key(path: str | Path) -> Path:
    """Encrypt `path` to `<path>.age` with `age -r $RESTRICTED_KEY` when possible.

    Returns the encrypted path on success, else the original path (and logs a warning).
    The plaintext is left in place for the caller to decide on.
    """
    path = Path(path)
    key = os.environ.get("RESTRICTED_KEY")
    if not key:
        log.warning("RESTRICTED_KEY not set: %s left unencrypted", path.name)
        return path
    age = shutil.which("age")
    if age is None:
        log.warning("`age` not found on PATH: %s left unencrypted", path.name)
        return path
    out = path.with_name(path.name + ".age")
    res = subprocess.run(
        [age, "-r", key, "-o", str(out), str(path)], capture_output=True, text=True, check=False
    )
    if res.returncode != 0:
        log.warning("age failed (%s): %s left unencrypted", res.stderr.strip()[:200], path.name)
        return path
    return out
