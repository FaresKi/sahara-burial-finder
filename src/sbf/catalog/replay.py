"""Offline replay of saved M2M responses (SPEC section 9.3: rebuild from raw, no network).

`RawStore` saves every response verbatim to `<dir>/<endpoint>_<hash>.json` and appends the request
to `manifest.jsonl`. `ReplayClient` implements the `CatalogClient` protocol on top of that
directory: a call is answered from the file recorded for the identical (endpoint, payload); a
call with no recording raises `M2MError` (recorded as a failure by `run_search`; never invented).
It never imports `requests` and never needs credentials.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sbf.catalog.m2m import M2MError, _check, request_hash


class ReplayClient:
    last_raw_file: str | None

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.last_raw_file = None
        self.calls: list[tuple[str, dict[str, Any]]] = []
        manifest = self.dir / "manifest.jsonl"
        self.files: dict[str, str] = {}
        self._records: list[tuple[str, dict[str, Any]]] = []
        if manifest.exists():
            for line in manifest.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                self._records.append((rec["endpoint"], rec["request"]))
                self.files[request_hash(rec["endpoint"], rec["request"])] = rec["file"]

    def call(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((endpoint, payload))
        name = self.files.get(request_hash(endpoint, payload))
        path = self.dir / name if name else None
        if path is None or not path.exists():
            self.last_raw_file = None
            raise M2MError(endpoint, "no saved raw response for this request (replay)")
        self.last_raw_file = name
        return _check(endpoint, json.loads(path.read_text()))

    def has_endpoint(self, endpoint: str) -> bool:
        return any(e == endpoint for e, _ in self._records)

    def recorded_range(self) -> tuple[str | None, str | None]:
        """(start, end) acquisition range of the first recorded scene-search, else (None, None)."""
        for endpoint, req in self._records:
            if endpoint == "scene-search":
                f = (req.get("sceneFilter") or {}).get("acquisitionFilter") or {}
                return f.get("start"), f.get("end")
        return None, None

    def searched_aois(self, aois) -> list[str]:
        """AOI ids whose polygon appears as the spatial filter of a recorded scene-search."""
        from shapely.geometry import mapping

        seen = set()
        for endpoint, req in self._records:
            if endpoint == "scene-search":
                gj = ((req.get("sceneFilter") or {}).get("spatialFilter") or {}).get("geoJson")
                if gj is not None:
                    seen.add(json.dumps(gj, sort_keys=True))
        out = []
        for r in aois.itertuples():
            ref = json.dumps(json.loads(json.dumps(mapping(r.geometry))), sort_keys=True)
            if ref in seen:
                out.append(r.aoi_id)
        return out
