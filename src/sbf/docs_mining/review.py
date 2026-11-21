"""Human review: CSV export, reviewed-CSV loading, precision report (no quotes)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sbf.core.config import Settings, load_settings
from sbf.docs_mining.pdf import docs_interim_dir
from sbf.docs_mining.schema import SpatialClaim

MIN_REVIEWED = 30
REVIEW_COLUMNS = [
    "extraction_id",
    "doc_id",
    "page",
    "quote",
    "object_buried",
    "anchor",
    "distance_km",
    "distance_uncertainty_km",
    "bearing",
    "bearing_uncertainty_deg",
    "time_period",
    "confidence_source",
    "notes",
    "describes_reference_feature",
    "accept",
    "reviewer_note",
]
_YES = {"y", "yes", "1", "true", "accept", "accepted", "x"}
_NO = {"n", "no", "0", "false", "reject", "rejected"}


def load_extractions(settings: Settings | None = None) -> list[SpatialClaim]:
    settings = settings or load_settings()
    p = docs_interim_dir(settings) / "extractions.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `sbf docs extract` first")
    return [SpatialClaim.model_validate_json(x) for x in p.read_text().splitlines() if x.strip()]


def export_review(settings: Settings | None = None) -> Path:
    """Write review.csv; keeps accept/reviewer_note already entered for the same ids."""
    settings = settings or load_settings()
    path = docs_interim_dir(settings) / "review.csv"
    prior: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open(newline="") as f:
            prior = {r["extraction_id"]: r for r in csv.DictReader(f)}
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for c in load_extractions(settings):
            row = {k: ("" if v is None else v) for k, v in c.model_dump().items()}
            old = prior.get(c.extraction_id, {})
            row["accept"] = old.get("accept", "")
            row["reviewer_note"] = old.get("reviewer_note", "")
            w.writerow({k: row.get(k, "") for k in REVIEW_COLUMNS})
    return path


@dataclass
class Precision:
    total: int
    reviewed: int
    accepted: int
    rejected: int
    precision: float | None
    sufficient: bool
    by_source: dict[str, tuple[int, int]] = field(default_factory=dict)  # (accepted, reviewed)
    reference_feature_records: int = 0


def parse_decision(value: str) -> bool | None:
    v = (value or "").strip().lower()
    if v in _YES:
        return True
    if v in _NO:
        return False
    return None


def compute_precision(csv_path: Path) -> Precision:
    with Path(csv_path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    acc = rej = 0
    by_source: dict[str, list[int]] = {}
    ref = 0
    for r in rows:
        if str(r.get("describes_reference_feature", "")).strip().lower() == "true":
            ref += 1
        d = parse_decision(r.get("accept", ""))
        if d is None:
            continue
        s = by_source.setdefault(r.get("confidence_source", "") or "unspecified", [0, 0])
        s[1] += 1
        if d:
            acc += 1
            s[0] += 1
        else:
            rej += 1
    reviewed = acc + rej
    return Precision(
        total=total,
        reviewed=reviewed,
        accepted=acc,
        rejected=rej,
        precision=(acc / reviewed) if reviewed else None,
        sufficient=reviewed >= MIN_REVIEWED,
        by_source={k: (v[0], v[1]) for k, v in by_source.items()},
        reference_feature_records=ref,
    )


def render_report(p: Precision) -> str:
    lines = ["# Document extraction precision", ""]
    if p.sufficient and p.precision is not None:
        lines.append(
            f"**Precision: {p.precision:.1%}** ({p.accepted} accepted / {p.reviewed} reviewed)"
        )
    else:
        lines.append(
            f"**Precision: insufficient** ({p.reviewed} reviewed; at least {MIN_REVIEWED} required)"
        )
        if p.precision is not None:
            lines.append(f"(provisional, not reportable: {p.accepted}/{p.reviewed})")
    lines += [
        "",
        f"- Extracted records: {p.total}",
        f"- Reviewed: {p.reviewed} (accepted {p.accepted}, rejected {p.rejected})",
        "- Records describing a reference feature (excluded from prior evaluation): "
        + str(p.reference_feature_records),
    ]
    if p.by_source:
        lines += ["", "| confidence_source | accepted | reviewed |", "|---|---|---|"]
        lines += [f"| {k} | {a} | {n} |" for k, (a, n) in sorted(p.by_source.items())]
    lines += [
        "",
        f"Generated {datetime.now(UTC).date().isoformat()}. Quotes are internal and not included.",
        "",
    ]
    return "\n".join(lines)


def write_precision_report(settings: Settings | None = None) -> tuple[Path, Precision]:
    settings = settings or load_settings()
    csv_path = docs_interim_dir(settings) / "review.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found; run `sbf docs export-review` first")
    p = compute_precision(csv_path)
    out = settings.outputs_public / "docs_extraction_precision.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(p))
    return out, p
