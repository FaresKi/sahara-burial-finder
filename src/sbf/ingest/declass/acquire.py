"""Browse-image and free-frame acquisition (ticket 04). Free (already scanned) frames only."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from sbf.catalog.download import (
    choose_product,
    download_options,
    request_download,
    retrieve_download,
)
from sbf.catalog.m2m import CatalogClient, M2MError
from sbf.core.provenance import append_step, prov_path, write_prov
from sbf.ingest.declass.transfer import TransferError, download_file, extract_archive, is_archive

BROWSE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


@dataclass
class Failure:
    entity_id: str
    stage: str
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class AcquireReport:
    done: list[str]
    skipped: list[str]
    failures: list[Failure]


def entity_dir(root: Path, entity_id: str) -> Path:
    """`<root>/<entity_id>`; refuses IDs that could escape `root`."""
    if not SAFE_ID.match(str(entity_id)) or ".." in str(entity_id):
        raise TransferError(f"unsafe entity id {entity_id!r}")
    return Path(root) / str(entity_id)


def select_free_frames(coverage: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Scanned (free) frames intersecting >=1 of `features` (default: any reference feature)."""
    want = set(features) if features else None

    def hit(tags: str) -> bool:
        t = {x for x in str(tags or "").split(";") if x}
        return bool(t & want) if want else bool(t)

    df = coverage[coverage["availability"] == "scanned"]
    df = df[df["intersecting_features"].map(hit)]
    return df.drop_duplicates(subset=["entity_id"]).reset_index(drop=True)


def _browse_ext(url: str) -> str:
    ext = Path(urlparse(url).path).suffix.lower()
    return ext if ext in BROWSE_EXTS else ".jpg"


def download_browse(
    coverage: pd.DataFrame, root: Path, session: Any, *, timeout: float = 120.0
) -> AcquireReport:
    """Download `browse.<ext>` for every coverage row to `<root>/<entity_id>/`. Existing files
    are skipped. Rows without a URL or whose download fails are recorded, never faked."""
    rep = AcquireReport([], [], [])
    for r in coverage.drop_duplicates(subset=["entity_id"]).itertuples():
        eid = str(r.entity_id)
        url = str(r.browse_url or "")
        try:
            d = entity_dir(root, eid)
            if not url:
                raise TransferError("no browse_url in coverage")
            dest = d / f"browse{_browse_ext(url)}"
            if dest.exists():
                rep.skipped.append(eid)
                continue
            info = download_file(session, url, dest, timeout=timeout)
            write_prov(
                dest,
                source=str(r.dataset),
                source_id=eid,
                acquisition_date=r.acquisition_date or None,
                steps=[f"browse image download (verified: {info['verified']})"],
            )
            rep.done.append(eid)
        except TransferError as exc:
            rep.failures.append(Failure(eid, "browse", str(exc)))
    return rep


def _filename(url: str, entity_id: str, product_code: str | None) -> str:
    name = Path(urlparse(url).path).name
    name = re.sub(r"[^A-Za-z0-9_.\-]", "_", name).lstrip(".")
    return name or f"{entity_id}_{product_code or 'product'}.bin"


def download_free_frames(
    client: CatalogClient,
    session: Any,
    frames: pd.DataFrame,
    root: Path,
    *,
    extract: bool = True,
    poll_seconds: float = 5.0,
    max_polls: int = 12,
    sleep: Callable[[float], None] = time.sleep,
    timeout: float = 120.0,
) -> AcquireReport:
    """For each scanned frame: download-options -> highest-resolution available product ->
    download-request (poll download-retrieve if preparing) -> resumable download with
    size/checksum verification -> `.prov.json` -> safe archive extraction.

    Failures are recorded per frame and never stop the batch; only free products are requested.
    """
    rep = AcquireReport([], [], [])
    for dataset, grp in frames.groupby("dataset"):
        ids = [str(e) for e in grp["entity_id"]]
        try:
            products = download_options(client, str(dataset), ids)
        except M2MError as exc:
            rep.failures += [Failure(e, "download-options", str(exc)) for e in ids]
            continue
        for r in grp.itertuples():
            eid = str(r.entity_id)
            try:
                _one(
                    client, session, r, [p for p in products if p.entity_id == eid], root, rep,
                    extract, poll_seconds, max_polls, sleep, timeout,
                )  # fmt: skip
            except (TransferError, M2MError) as exc:
                rep.failures.append(Failure(eid, "download", str(exc)))
    return rep


def _one(client, session, r, products, root, rep, extract, poll_seconds, max_polls, sleep, timeout):
    eid = str(r.entity_id)
    d = entity_dir(root, eid)
    product = choose_product(products)
    if product is None:
        raise TransferError("no available (free) download product for this frame")
    ready, preparing = request_download(client, product, label=f"sbf_{eid}")
    polls = 0
    while not ready and preparing and polls < max_polls:
        sleep(poll_seconds)
        polls += 1
        ready, preparing = retrieve_download(client, f"sbf_{eid}")
    if not ready:
        raise TransferError("product still preparing server-side; re-run later")
    link = next((x for x in ready if not x.entity_id or x.entity_id == eid), ready[0])
    dest = d / _filename(link.url, eid, product.code)
    if dest.exists() and prov_path(dest).exists():
        rep.skipped.append(eid)
        return
    info = download_file(
        session,
        link.url,
        dest,
        expected_size=product.filesize,
        checksum=product.checksum,
        timeout=timeout,
    )
    prov = write_prov(
        dest,
        source=str(r.dataset),
        source_id=eid,
        acquisition_date=r.acquisition_date or None,
        steps=[f"m2m download {product.name or product.code or product.product_id}"]
        + ([] if info["verified"] != "none" else ["integrity NOT verified (no size/checksum)"]),
    )
    if extract and is_archive(dest):
        files = extract_archive(dest, d)
        append_step(prov, f"extracted archive safely ({len(files)} files)")
    rep.done.append(eid)
