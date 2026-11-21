"""Claim extraction: DocExtractor protocol, Claude + fake implementations, pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from sbf.core.config import Settings, load_settings
from sbf.docs_mining.pdf import Chunk, docs_interim_dir, load_chunks
from sbf.docs_mining.schema import (
    ClaimDraft,
    ClaimList,
    SpatialClaim,
    describes_reference_feature,
    make_extraction_id,
    normalize_ws,
)

PROMPT_VERSION = "v1"
DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_OUTPUT_TOKENS = 4096

SYSTEM_PROMPT = """\
You extract explicit spatial claims from excerpts of documents about French nuclear \
testing in the Algerian Sahara (Reggane / Hamoudia, In Ekker / Tan Afella). Documents \
may be in English or French.

TASK
Find statements that say WHERE contaminated or test-related material was buried, \
abandoned, dumped or left. One record per distinct claim. Return {"claims": []} if the \
excerpt has none. Most excerpts have none.

RULES
1. Extract only what the text explicitly states. Do not infer, estimate, convert or \
complete anything. Do not use outside knowledge.
2. A claim must concern buried / abandoned / dumped / left-behind material (aircraft, \
vehicles, drums, cables, slag, vessels, equipment, waste) AND give some spatial \
information: a distance, a direction, or a named location.
3. quote: copy a verbatim passage from the excerpt, at most 25 words, exactly as \
written (same language, spelling, punctuation). Never translate or paraphrase it. \
Choose the shortest passage that supports the claim.
4. object_buried: one of aircraft, vehicles, drums, cable, slag, vessels, unknown. \
Use "unknown" for any other or unspecified material.
5. anchor: the reference place as named in the text (e.g. "Gerboise Bleue ground zero", \
"Reggane plateau", "E2 portal"), free text, in the document's wording.
6. distance_km / distance_uncertainty_km / bearing / bearing_uncertainty_deg / \
time_period: only if stated in the text; otherwise null. Convert units to km only when \
the text gives a plain unit (miles, metres). Bearing is a compass word or angle as \
written (e.g. "NE"). Uncertainty only if the text states a range or "about"-type \
qualifier you can express numerically; otherwise null.
7. confidence_source: "testimony" if the claim is a witness/veteran/worker account; \
"official" if from an official/government/agency/inspection record or report; \
"secondary" if the document is merely relaying or summarising another source.
8. notes: brief neutral remark on ambiguity (or null). No speculation.
9. If unsure whether something is an explicit spatial claim, omit it.
"""


class ExtractionError(RuntimeError):
    """Extractor could not produce valid output for a chunk."""


class DocExtractor(Protocol):
    model: str
    prompt_version: str

    def extract(self, chunk: Chunk) -> list[ClaimDraft]:
        """Return validated claim drafts for a chunk; raise ExtractionError on failure."""
        ...


def get_model() -> str:
    return os.environ.get("SBF_EXTRACT_MODEL", DEFAULT_MODEL)


def user_message(chunk: Chunk) -> str:
    return (
        f"Document excerpt (doc_id={chunk.doc_id}, page {chunk.page}):\n"
        f"<excerpt>\n{chunk.text}\n</excerpt>"
    )


class ClaudeExtractor:
    """Anthropic SDK extractor with JSON-schema structured output and retries."""

    def __init__(
        self,
        model: str | None = None,
        *,
        client: Any | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.model = model or get_model()
        self.prompt_version = PROMPT_VERSION
        self.max_attempts = max_attempts
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ExtractionError("ANTHROPIC_API_KEY is not set in the environment")
            import anthropic  # lazy: heavy optional dependency (D6)

            self._client = anthropic.Anthropic()
        return self._client

    def extract(self, chunk: Chunk) -> list[ClaimDraft]:
        schema = ClaimList.model_json_schema()
        messages: list[dict] = [{"role": "user", "content": user_message(chunk)}]
        last_err = "no attempt made"
        for _ in range(self.max_attempts):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=SYSTEM_PROMPT,
                messages=messages,
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            try:
                return ClaimList.model_validate_json(text).claims
            except ValidationError as e:
                last_err = str(e)[:500]
                messages = [
                    messages[0],
                    {"role": "assistant", "content": text or "{}"},
                    {
                        "role": "user",
                        "content": "Your output failed validation: "
                        f"{last_err}\nReturn corrected JSON only; quotes must be <= 25 words.",
                    },
                ]
        raise ExtractionError(f"validation failed after {self.max_attempts} attempts: {last_err}")


class FakeExtractor:
    """Deterministic test extractor: returns canned drafts per chunk."""

    def __init__(
        self,
        responses: dict[str, list[ClaimDraft]] | None = None,
        *,
        default: list[ClaimDraft] | None = None,
        fail_on: set[str] | None = None,
        model: str = "fake-model",
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        self.responses = responses or {}
        self.default = default or []
        self.fail_on = fail_on or set()
        self.model = model
        self.prompt_version = prompt_version
        self.calls: list[str] = []

    def extract(self, chunk: Chunk) -> list[ClaimDraft]:
        self.calls.append(chunk.chunk_hash)
        if chunk.chunk_hash in self.fail_on:
            raise ExtractionError("fake failure")
        return list(self.responses.get(chunk.chunk_hash, self.default))


def quote_in_chunk(quote: str, chunk_text: str) -> bool:
    return normalize_ws(quote) in normalize_ws(chunk_text)


# ---- cost estimate (no API call) -------------------------------------------------

# Placeholder USD per million tokens; override with SBF_PRICE_IN / SBF_PRICE_OUT. Verify
# against current pricing before relying on the dollar figure.
DEFAULT_PRICE_IN, DEFAULT_PRICE_OUT = 3.0, 15.0
EST_OUTPUT_TOKENS_PER_CHUNK = 150


@dataclass
class Estimate:
    n_chunks: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str


def estimate_cost(chunks: list[Chunk], model: str | None = None) -> Estimate:
    p_in = float(os.environ.get("SBF_PRICE_IN", DEFAULT_PRICE_IN))
    p_out = float(os.environ.get("SBF_PRICE_OUT", DEFAULT_PRICE_OUT))
    sys_tokens = len(SYSTEM_PROMPT) // 3 + 400  # prompt + schema, rough
    tin = sum(len(c.text) // 3 + 60 + sys_tokens for c in chunks)  # ~3 chars/token: conservative
    tout = EST_OUTPUT_TOKENS_PER_CHUNK * len(chunks)
    return Estimate(len(chunks), tin, tout, (tin * p_in + tout * p_out) / 1e6, model or get_model())


# ---- pipeline --------------------------------------------------------------------


def _cache_path(settings: Settings, chunk: Chunk, model: str, version: str) -> Path:
    key = f"{chunk.chunk_hash}__{model}__{version}".replace("/", "_")
    return docs_interim_dir(settings) / "cache" / f"{key}.json"


@dataclass
class RunSummary:
    chunks: int = 0
    cached: int = 0
    failed: int = 0
    claims: int = 0
    rejected_quotes: int = 0


def run_extraction(
    extractor: DocExtractor,
    settings: Settings | None = None,
    *,
    limit: int | None = None,
) -> RunSummary:
    """Extract claims for all (or the first `limit`) chunks; write extractions.jsonl.

    Failures are recorded in failures.jsonl, never guessed. Claims whose quote is not a
    verbatim substring of the chunk (whitespace-normalised) are rejected and logged.
    """
    settings = settings or load_settings()
    out_dir = docs_interim_dir(settings)
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    chunks = load_chunks(settings)
    if limit is not None:
        chunks = chunks[:limit]
    summary = RunSummary(chunks=len(chunks))
    claims: dict[str, SpatialClaim] = {}
    failures: list[dict] = []
    for chunk in chunks:
        cp = _cache_path(settings, chunk, extractor.model, extractor.prompt_version)
        drafts: list[ClaimDraft] | None = None
        if cp.exists():
            try:
                drafts = ClaimList.model_validate_json(cp.read_text()).claims
                summary.cached += 1
            except ValidationError:
                drafts = None
        if drafts is None:
            try:
                drafts = extractor.extract(chunk)
            except ExtractionError as e:
                summary.failed += 1
                failures.append(
                    {
                        "doc_id": chunk.doc_id,
                        "page": chunk.page,
                        "chunk_hash": chunk.chunk_hash,
                        "error": str(e)[:500],
                    }
                )
                continue
            cp.write_text(ClaimList(claims=drafts).model_dump_json())
        for d in drafts:
            if not quote_in_chunk(d.quote, chunk.text):
                summary.rejected_quotes += 1
                failures.append(
                    {
                        "doc_id": chunk.doc_id,
                        "page": chunk.page,
                        "chunk_hash": chunk.chunk_hash,
                        "error": "quote not verbatim in chunk; claim rejected",
                    }
                )
                continue
            eid = make_extraction_id(chunk.doc_id, chunk.page, d.quote, d.anchor)
            claims.setdefault(
                eid,
                SpatialClaim(
                    extraction_id=eid,
                    doc_id=chunk.doc_id,
                    page=chunk.page,
                    **d.model_dump(),
                    describes_reference_feature=describes_reference_feature(d.anchor),
                    model=extractor.model,
                    prompt_version=extractor.prompt_version,
                ),
            )
    summary.claims = len(claims)
    with (out_dir / "extractions.jsonl").open("w") as f:
        for c in claims.values():
            f.write(c.model_dump_json() + "\n")
    with (out_dir / "failures.jsonl").open("w") as f:
        for r in failures:
            f.write(json.dumps(r) + "\n")
    return summary
