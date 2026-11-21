"""Resumable streaming download and safe archive extraction. No network library imported here:
`session` is any object with a requests-like ``get(url, headers=, stream=True, timeout=)``."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any

CHUNK = 1 << 20


class TransferError(RuntimeError):
    """Download or extraction failed. Never swallowed: callers record it, no file is faked."""


def _hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _checksum_algo(expected: str) -> str | None:
    return {32: "md5", 64: "sha256"}.get(len(expected))


def download_file(
    session: Any,
    url: str,
    dest: Path,
    *,
    expected_size: int | None = None,
    checksum: str | None = None,
    timeout: float = 120.0,
    _retry: bool = True,
) -> dict[str, Any]:
    """Download `url` to `dest` via `dest.part` (HTTP Range resume). Returns
    ``{"size", "verified"}`` where verified is "checksum", "size" or "none".

    Incomplete streams leave the `.part` for a later resume and raise TransferError; a size or
    checksum mismatch on a complete file deletes the `.part` and raises.
    """
    if not url.lower().startswith(("http://", "https://")):
        raise TransferError(f"refusing non-http(s) URL scheme: {url[:20]!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        resp = session.get(url, headers=headers, stream=True, timeout=timeout)
    except Exception as exc:
        raise TransferError(f"request failed: {exc!r}") from exc
    try:
        status = resp.status_code
        if status == 416:  # range not satisfiable: part is complete or stale
            if expected_size is not None and have == expected_size:
                return _finalize(part, dest, expected_size, checksum)
            part.unlink(missing_ok=True)
            if _retry:
                return download_file(
                    session,
                    url,
                    dest,
                    expected_size=expected_size,
                    checksum=checksum,
                    timeout=timeout,
                    _retry=False,
                )
            raise TransferError("HTTP 416 on a fresh request")
        if status == 206:
            m = re.match(r"bytes (\d+)-", resp.headers.get("Content-Range", ""))
            if not m or int(m.group(1)) != have:
                part.unlink(missing_ok=True)
                raise TransferError("server returned an unexpected Content-Range; restart")
            mode = "ab"
        elif status == 200:  # server ignored Range (or fresh): start over
            mode, have = "wb", 0
        else:
            raise TransferError(f"HTTP {status}")
        clen = resp.headers.get("Content-Length")
        total = expected_size if expected_size is not None else (have + int(clen) if clen else None)
        try:
            with part.open(mode) as fh:
                for chunk in resp.iter_content(CHUNK):
                    if chunk:
                        fh.write(chunk)
        except Exception as exc:
            raise TransferError(f"stream interrupted ({exc!r}); .part kept for resume") from exc
    finally:
        close = getattr(resp, "close", None)
        if close:
            close()
    size = part.stat().st_size
    if total is not None and size < total:
        raise TransferError(f"incomplete: {size}/{total} bytes; .part kept for resume")
    return _finalize(part, dest, total, checksum)


def _finalize(part: Path, dest: Path, expected: int | None, checksum: str | None) -> dict[str, Any]:
    size = part.stat().st_size
    if expected is not None and size != expected:
        part.unlink(missing_ok=True)
        raise TransferError(f"size mismatch: got {size}, expected {expected}; discarded")
    verified = "size" if expected is not None else "none"
    if checksum:
        algo = _checksum_algo(checksum)
        if algo is None:
            part.unlink(missing_ok=True)
            raise TransferError(f"unrecognised checksum format {checksum!r}; discarded")
        if _hash_file(part, algo).lower() != checksum.lower():
            part.unlink(missing_ok=True)
            raise TransferError(f"{algo} checksum mismatch; discarded")
        verified = "checksum"
    os.replace(part, dest)
    return {"size": size, "verified": verified}


def is_archive(path: Path) -> bool:
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


def _safe_target(dest: Path, name: str) -> Path:
    if not name or name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name):
        raise TransferError(f"unsafe archive member (absolute path): {name!r}")
    target = (dest / name).resolve()
    if target != dest and dest not in target.parents:
        raise TransferError(f"unsafe archive member (path traversal): {name!r}")
    return target


def extract_archive(archive: Path, dest: Path) -> list[str]:
    """Extract a zip/tar/tgz into `dest`. All members are validated BEFORE anything is written;
    absolute paths, `..` traversal, symlinks, hardlinks and device files abort the whole archive.
    Returns the extracted relative file names."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    written: list[str] = []
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            for i in infos:
                _safe_target(root, i.filename)
                if (i.external_attr >> 16) & 0o170000 == 0o120000:
                    raise TransferError(f"unsafe archive member (symlink): {i.filename!r}")
            for i in infos:
                target = _safe_target(root, i.filename)
                if i.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(i) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out, CHUNK)
                written.append(i.filename)
        return written
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            members = tf.getmembers()
            for m in members:
                _safe_target(root, m.name)
                if not (m.isfile() or m.isdir()):
                    raise TransferError(f"unsafe archive member (not a plain file): {m.name!r}")
            for m in members:
                target = _safe_target(root, m.name)
                if m.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                if src is None:
                    raise TransferError(f"cannot read archive member {m.name!r}")
                with src, target.open("wb") as out:
                    shutil.copyfileobj(src, out, CHUNK)
                written.append(m.name)
        return written
    raise TransferError(f"{archive.name} is not a zip or tar archive")
