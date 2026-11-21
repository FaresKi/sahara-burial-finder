"""Strict schema for spatial claims extracted from documents (SPEC Phase 5)."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

ObjectBuried = Literal["aircraft", "vehicles", "drums", "cable", "slag", "vessels", "unknown"]
ConfidenceSource = Literal["testimony", "official", "secondary"]

MAX_QUOTE_WORDS = 25


def normalize_ws(text: str) -> str:
    """NFKC-normalise and collapse all whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _check_quote(v: str) -> str:
    if not v.strip():
        raise ValueError("quote must not be empty")
    if len(v.split()) > MAX_QUOTE_WORDS:
        raise ValueError(f"quote exceeds {MAX_QUOTE_WORDS} words")
    return v


class ClaimDraft(BaseModel):
    """What the model returns for one claim (no doc/page/provenance fields).

    All fields are required (nullable where optional) so the JSON schema is strict.
    """

    model_config = ConfigDict(extra="forbid")

    quote: str
    object_buried: ObjectBuried
    anchor: str
    distance_km: float | None
    distance_uncertainty_km: float | None
    bearing: str | None
    bearing_uncertainty_deg: float | None
    time_period: str | None
    confidence_source: ConfidenceSource
    notes: str | None

    @field_validator("quote")
    @classmethod
    def _quote(cls, v: str) -> str:
        return _check_quote(v)


class ClaimList(BaseModel):
    """Top-level structured-output object."""

    model_config = ConfigDict(extra="forbid")

    claims: list[ClaimDraft]


class SpatialClaim(BaseModel):
    """One spatial claim with provenance. Quote is internal only (never public)."""

    model_config = ConfigDict(extra="forbid")

    extraction_id: str
    doc_id: str
    page: int
    quote: str
    object_buried: ObjectBuried
    anchor: str
    distance_km: float | None = None
    distance_uncertainty_km: float | None = None
    bearing: str | None = None
    bearing_uncertainty_deg: float | None = None
    time_period: str | None = None
    confidence_source: ConfidenceSource
    notes: str | None = None
    describes_reference_feature: bool = False
    model: str
    prompt_version: str

    @field_validator("quote")
    @classmethod
    def _quote(cls, v: str) -> str:
        return _check_quote(v)


def make_extraction_id(doc_id: str, page: int, quote: str, anchor: str) -> str:
    key = f"{doc_id}|{page}|{normalize_ws(quote)}|{normalize_ws(anchor).casefold()}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


_REF_PATTERNS = re.compile(
    r"gerboise|ground\s*zero|\bGZ\s*-?\d?\b|\baugias\b|\bpollen\b|\bE-?2\b|b[ée]ryl|tikertine"
    r"|point z[ée]ro",
    re.IGNORECASE,
)


def describes_reference_feature(anchor: str) -> bool:
    """True if the anchor names a validation feature (GZ1-4, Pollen, E2/Beryl).

    Ticket 15 excludes such records from prior evaluation (leakage). Deliberately
    conservative: over-flagging only costs some evaluation records.
    """
    return bool(_REF_PATTERNS.search(anchor))
