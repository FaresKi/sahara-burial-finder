"""PDF ingestion: page text, chunking, document registry/manifest."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

import sbf
from sbf.core.config import Settings, load_settings

CHUNK_WORDS = 400
OVERLAP_WORDS = 60


class Chunk(BaseModel):
    doc_id: str
    page: int
    chunk_idx: int
    text: str
    chunk_hash: str


class DocRecord(BaseModel):
    doc_id: str
    filename: str
    sha256: str
    size_bytes: int
    n_pages: int
    n_chunks: int
    source: str = "local file placed by the human in data/raw/docs/"
    ingested_at: str
    sbf_version: str


def docs_raw_dir(settings: Settings) -> Path:
    return settings.data_raw / "docs"


def docs_interim_dir(settings: Settings) -> Path:
    return settings.data_interim / "docs"


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "doc"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_doc_id(path: Path) -> str:
    return f"{slugify(path.stem)}-{sha256_file(path)[:8]}"


def page_texts(path: Path) -> list[tuple[int, str]]:
    """(1-based page number, text) for every page."""
    import pymupdf

    with pymupdf.open(path) as doc:
        return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]


def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def chunk_pages(
    doc_id: str,
    pages: list[tuple[int, str]],
    *,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[Chunk]:
    """Split each page into overlapping word windows. Chunks never cross pages."""
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be < chunk_words")
    step = chunk_words - overlap_words
    out: list[Chunk] = []
    idx = 0
    for page, text in pages:
        words = text.split()
        if not words:
            continue
        start = 0
        while True:
            piece = " ".join(words[start : start + chunk_words])
            out.append(
                Chunk(
                    doc_id=doc_id,
                    page=page,
                    chunk_idx=idx,
                    text=piece,
                    chunk_hash=chunk_hash(piece),
                )
            )
            idx += 1
            if start + chunk_words >= len(words):
                break
            start += step
    return out


def ingest_docs(settings: Settings | None = None) -> list[DocRecord]:
    """Register every PDF in data/raw/docs/, write manifest.json and chunks.jsonl."""
    settings = settings or load_settings()
    raw, interim = docs_raw_dir(settings), docs_interim_dir(settings)
    interim.mkdir(parents=True, exist_ok=True)
    records: list[DocRecord] = []
    all_chunks: list[Chunk] = []
    for path in sorted(raw.glob("*.pdf")) if raw.is_dir() else []:
        doc_id = make_doc_id(path)
        pages = page_texts(path)
        chunks = chunk_pages(doc_id, pages)
        n_pages = len(pages)
        records.append(
            DocRecord(
                doc_id=doc_id,
                filename=path.name,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                n_pages=n_pages,
                n_chunks=len(chunks),
                ingested_at=datetime.now(UTC).isoformat(),
                sbf_version=sbf.__version__,
            )
        )
        all_chunks.extend(chunks)
    (interim / "manifest.json").write_text(json.dumps([r.model_dump() for r in records], indent=2))
    with (interim / "chunks.jsonl").open("w") as f:
        for c in all_chunks:
            f.write(c.model_dump_json() + "\n")
    return records


def load_chunks(settings: Settings | None = None) -> list[Chunk]:
    settings = settings or load_settings()
    p = docs_interim_dir(settings) / "chunks.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `sbf docs ingest` first")
    return [Chunk.model_validate_json(line) for line in p.read_text().splitlines() if line.strip()]
