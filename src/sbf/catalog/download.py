"""M2M download endpoints: `download-options`, `download-request`, `download-retrieve` (ticket 04).

Thin typed helpers over `CatalogClient.call`. No purchasing: these endpoints only ever concern
frames already scanned (free). Ordering unscanned frames ($30/frame) is a human gate and is not
implemented anywhere in this code base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from sbf.catalog.m2m import CatalogClient, M2MError
from sbf.catalog.search import _first

# ---------------------------------------------------------------------------------------
# M2M DOWNLOAD FIELD MAPPING. UNVERIFIED against the live API (no credentials available when
# written); modelled on the documented M2M schema. Correct HERE after the first live run;
# nothing else hard-codes these names. Each entry is a list of candidate keys, first hit wins.
#   download-options  request : {datasetName, entityIds: [..]}      response data: [product, ..]
#   download-request  request : {downloads: [{entityId, productId}], label}
#                     response data: {availableDownloads: [..], preparingDownloads: [..],
#                                     failed: [..]}
#   download-retrieve request : {label}  response data: {available: [..], requested: [..]}
# ---------------------------------------------------------------------------------------
PRODUCT_FIELDS = {
    "entity_id": ["entityId"],
    "product_id": ["id", "productId"],
    "product_name": ["productName", "displayId"],
    "product_code": ["productCode"],
    "available": ["available"],  # True when the product can be downloaded now (scanned/free)
    "filesize": ["filesize", "fileSize"],  # ASSUMED bytes
    "checksum": ["checksum", "md5", "sha256"],  # not documented for M2M; used only if present
}
DOWNLOAD_FIELDS = {
    "download_id": ["downloadId"],
    "entity_id": ["entityId"],
    "url": ["url"],
}
REQUEST_BUCKETS = {
    "ready": ["availableDownloads"],
    "preparing": ["preparingDownloads"],
    "failed": ["failed"],
}
RETRIEVE_BUCKETS = {"ready": ["available"], "preparing": ["requested"]}
BATCH = 50  # entity IDs per download-options call


@dataclass
class Product:
    entity_id: str
    product_id: str
    name: str
    code: str | None
    available: bool
    filesize: int | None
    checksum: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class DownloadLink:
    entity_id: str
    download_id: str | None
    url: str


def parse_product(d: dict) -> Product:
    size = _first(d, PRODUCT_FIELDS["filesize"])
    try:
        size = int(size) if size not in (None, "") else None
    except (TypeError, ValueError):
        size = None
    pid = _first(d, PRODUCT_FIELDS["product_id"])
    return Product(
        entity_id=str(_first(d, PRODUCT_FIELDS["entity_id"]) or ""),
        product_id=str(pid) if pid is not None else "",
        name=str(_first(d, PRODUCT_FIELDS["product_name"]) or ""),
        code=_first(d, PRODUCT_FIELDS["product_code"]),
        available=bool(_first(d, PRODUCT_FIELDS["available"])),
        filesize=size,
        checksum=_first(d, PRODUCT_FIELDS["checksum"]),
        raw=d,
    )


def download_options(client: CatalogClient, dataset: str, entity_ids: list[str]) -> list[Product]:
    """All download products for the given entities of one dataset (raises M2MError)."""
    out: list[Product] = []
    for i in range(0, len(entity_ids), BATCH):
        env = client.call(
            "download-options", {"datasetName": dataset, "entityIds": entity_ids[i : i + BATCH]}
        )
        out += [parse_product(d) for d in (env.get("data") or []) if isinstance(d, dict)]
    return out


def choose_product(products: list[Product]) -> Product | None:
    """Highest-resolution available product: the largest filesize among available products
    (browse/thumbnail-like names excluded). Heuristic: product naming is unverified."""
    cands = [
        p
        for p in products
        if p.available
        and p.product_id
        and not any(w in p.name.lower() for w in ("browse", "thumb", "preview"))
    ]
    if not cands:
        return None
    return max(cands, key=lambda p: p.filesize or 0)


def _links(items: Any) -> list[DownloadLink]:
    out = []
    for d in items or []:
        if not isinstance(d, dict):
            continue
        url = _first(d, DOWNLOAD_FIELDS["url"])
        if url:
            out.append(
                DownloadLink(
                    str(_first(d, DOWNLOAD_FIELDS["entity_id"]) or ""),
                    _first(d, DOWNLOAD_FIELDS["download_id"]),
                    url,
                )
            )
    return out


def _bucket(data: dict, names: list[str]) -> list:
    for n in names:
        if data.get(n):
            return data[n]
    return []


def request_download(
    client: CatalogClient, product: Product, label: str
) -> tuple[list[DownloadLink], int]:
    """Request one (free, scanned) product. Returns (ready links, number still preparing)."""
    env = client.call(
        "download-request",
        {
            "downloads": [{"entityId": product.entity_id, "productId": product.product_id}],
            "label": label,
        },
    )
    data = env.get("data") or {}
    ready = _links(_bucket(data, REQUEST_BUCKETS["ready"]))
    preparing = len(_bucket(data, REQUEST_BUCKETS["preparing"]))
    failed = _bucket(data, REQUEST_BUCKETS["failed"])
    if failed and not ready and not preparing:
        raise M2MError("download-request", f"request failed for {product.entity_id}: {failed!r}")
    return ready, preparing


def retrieve_download(client: CatalogClient, label: str) -> tuple[list[DownloadLink], int]:
    """Poll a labelled request. Returns (ready links, number still preparing)."""
    env = client.call("download-retrieve", {"label": label})
    data = env.get("data") or {}
    return (
        _links(_bucket(data, RETRIEVE_BUCKETS["ready"])),
        len(_bucket(data, RETRIEVE_BUCKETS["preparing"])),
    )


def refine_availability(
    client: CatalogClient, coverage: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict]]:
    """Replace the scene-search availability guess with `download-options` where possible.

    scanned = at least one product is available; needs_order = products listed, none available.
    No answer for a frame (no products, or a failed call) keeps the earlier scene-search guess
    (`availability_source` = "scene-search"). Failures are returned, never filled in.
    """
    out = coverage.copy()
    out["availability_source"] = out.get("availability_source", "scene-search")
    out["availability_source"] = out["availability_source"].fillna("scene-search")
    failures: list[dict] = []
    for dataset, grp in out.groupby("dataset"):
        ids = sorted(set(grp["entity_id"].astype(str)))
        try:
            products = download_options(client, str(dataset), ids)
        except M2MError as exc:
            failures.append(
                {"endpoint": "download-options", "context": str(dataset), "message": str(exc)}
            )
            continue
        by_entity: dict[str, list[Product]] = {}
        for p in products:
            by_entity.setdefault(p.entity_id, []).append(p)
        for idx, row in grp.iterrows():
            ps = by_entity.get(str(row["entity_id"]))
            if not ps:
                continue
            out.loc[idx, "availability"] = (
                "scanned" if any(p.available for p in ps) else "needs_order"
            )
            out.loc[idx, "availability_source"] = "download-options"
    return out, failures
