"""USGS EarthExplorer M2M client, behind a Protocol so pure logic takes any client.

HARD RULES: credentials come from env only (SPEC 0.3); every response is stored
verbatim (SPEC 0.2/Phase 1 acceptance); failures are raised/recorded, never filled in.

The response envelope is the documented M2M one:
``{"requestId", "version", "data", "errorCode", "errorMessage"}``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol

BASE_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/"
# Endpoints used: login-token, dataset-search, scene-search (sbf.catalog.search) and
# download-options, download-request, download-retrieve (sbf.catalog.download; free frames only).
NO_SAVE_ENDPOINTS = {"login-token"}  # response holds the session API key: never persisted


class M2MError(RuntimeError):
    """Transport failure or non-null M2M `errorCode`. Carries the raw envelope if any."""

    def __init__(self, endpoint: str, message: str, envelope: dict | None = None):
        super().__init__(f"{endpoint}: {message}")
        self.endpoint = endpoint
        self.envelope = envelope


class CatalogClient(Protocol):
    """What the pure catalogue logic needs: one call returning the raw response envelope.

    Implementations must save the raw response (see `RawStore`) and expose
    `last_raw_file` (name of the file holding the most recent response, or None).
    """

    last_raw_file: str | None

    def call(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST `payload` to `endpoint`; return the full envelope. Raise `M2MError` on failure."""
        ...


def request_hash(endpoint: str, payload: dict[str, Any]) -> str:
    blob = json.dumps({"endpoint": endpoint, "payload": payload}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class RawStore:
    """Saves each response verbatim to `<dir>/<endpoint>_<hash>.json`.

    The hash is of (endpoint, request payload); the request is appended to
    `manifest.jsonl` so every file can be traced back to its query.
    """

    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def save(self, endpoint: str, payload: dict[str, Any], envelope: dict[str, Any]) -> str | None:
        if endpoint in NO_SAVE_ENDPOINTS:
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        name = f"{endpoint}_{request_hash(endpoint, payload)}.json"
        (self.dir / name).write_text(json.dumps(envelope, indent=2, sort_keys=False))
        with (self.dir / "manifest.jsonl").open("a") as fh:
            fh.write(json.dumps({"file": name, "endpoint": endpoint, "request": payload}) + "\n")
        return name


def _check(endpoint: str, envelope: dict[str, Any]) -> dict[str, Any]:
    if envelope.get("errorCode"):
        raise M2MError(
            endpoint, f"{envelope['errorCode']}: {envelope.get('errorMessage')}", envelope
        )
    return envelope


class M2MClient:
    """requests-based client. Logs in lazily with USGS_USERNAME + USGS_M2M_TOKEN."""

    def __init__(
        self,
        store: RawStore,
        *,
        base_url: str = BASE_URL,
        session: Any = None,
        timeout: float = 120.0,
    ):
        import requests

        self.store = store
        self.base_url = base_url
        self.session = session or requests.Session()
        self.timeout = timeout
        self.api_key: str | None = None
        self.last_raw_file: str | None = None

    @staticmethod
    def credentials() -> tuple[str, str]:
        user, token = os.environ.get("USGS_USERNAME"), os.environ.get("USGS_M2M_TOKEN")
        if not user or not token:
            raise M2MError("login-token", "USGS_USERNAME and USGS_M2M_TOKEN must be set in the env")
        return user, token

    def login(self) -> None:
        user, token = self.credentials()
        env = self._post("login-token", {"username": user, "token": token}, auth=False)
        self.api_key = env["data"]

    def call(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.api_key is None and endpoint != "login-token":
            self.login()
        env = self._post(endpoint, payload, auth=True)
        return env

    def _post(self, endpoint: str, payload: dict[str, Any], *, auth: bool) -> dict[str, Any]:
        headers = {"X-Auth-Token": self.api_key} if auth and self.api_key else {}
        try:
            resp = self.session.post(
                self.base_url + endpoint, json=payload, headers=headers, timeout=self.timeout
            )
            env = resp.json()
        except Exception as exc:  # network/JSON failure: recorded by caller, never papered over
            raise M2MError(endpoint, f"transport failure: {exc!r}") from exc
        self.last_raw_file = self.store.save(endpoint, payload, env)
        return _check(endpoint, env)


class FixtureClient:
    """Replays recorded envelopes (tests; no network). Also writes raw files if given a store.

    `responses` maps endpoint -> list of envelopes returned in call order (the last one
    repeats), or a callable ``(endpoint, payload) -> envelope``.
    """

    def __init__(self, responses: Any, store: RawStore | None = None):
        self.responses = responses
        self.store = store
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._n: dict[str, int] = {}
        self.last_raw_file: str | None = None

    def call(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((endpoint, payload))
        if callable(self.responses):
            env = self.responses(endpoint, payload)
        else:
            seq = self.responses.get(endpoint)
            if not seq:
                raise M2MError(endpoint, "no fixture recorded for this endpoint")
            i = self._n.get(endpoint, 0)
            self._n[endpoint] = i + 1
            env = seq[min(i, len(seq) - 1)]
        self.last_raw_file = self.store.save(endpoint, payload, env) if self.store else None
        return _check(endpoint, env)
