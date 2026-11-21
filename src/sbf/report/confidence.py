"""Candidate confidence (low/medium/high) with a human-readable justification (SPEC Phase 6).

Pure functions; all thresholds live in the constants below.

Rules
-----
* high:   label ops_1961_1968 AND >= 2 families incl. a historical one AND mean_S >= 0.9
          AND supporting-frame checkpoint RMSE <= 10 m.
* medium: >= 2 families AND mean_S >= 0.7 AND RMSE <= 20 m (any label).
* low:    everything else.
Caps (applied after, never raise a level):
* only one historical date shows the candidate      -> at most medium
* label post_2025_likely_cleanup / pre_1960_natural_or_unknown -> at most low
* Phase 4 blind-test gate failed                    -> at most low
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

LEVELS = ("low", "medium", "high")
HISTORICAL_FAMILIES = frozenset({"hist_change", "hist_anomaly"})

HIGH_LABEL = "ops_1961_1968"
HIGH_MIN_FAMILIES = 2
HIGH_MIN_S = 0.9
HIGH_MAX_RMSE_M = 10.0
MEDIUM_MIN_FAMILIES = 2
MEDIUM_MIN_S = 0.7
MEDIUM_MAX_RMSE_M = 20.0
LOW_CAP_LABELS = frozenset({"post_2025_likely_cleanup", "pre_1960_natural_or_unknown"})
SINGLE_DATE_CAP = "medium"

STATEMENTS = (
    "Candidate area for verification — not a confirmed burial site.",
    "Requires ground verification by competent authorities.",
)
INVISIBLE_CAVEAT = (
    "Invisible-target caveat: material buried deeper than ~1–2 m, or inside tunnels, may "
    "leave no surface or radar signature; absence of a candidate is not evidence of absence."
)
LABEL_NOISE_CAVEAT = (
    "Label-noise caveat: the date bracket and label derive from a small set of frames with "
    "imperfect georeferencing and may be wrong; reference positions are approximate."
)


@dataclass
class Confidence:
    level: str
    justification: list[str] = field(default_factory=list)


def _cap(level: str, cap: str) -> str:
    return LEVELS[min(LEVELS.index(level), LEVELS.index(cap))]


def _families(families) -> list[str]:
    if isinstance(families, str):
        return [f.strip() for f in families.split(",") if f.strip()]
    return list(families or [])


def assess_confidence(
    *,
    label: str,
    mean_S: float,
    families,
    best_rmse_m: float | None,
    n_hist_dates: int,
    prior_contribution: float | None = None,
    blind_test_passed: bool = True,
) -> Confidence:
    """Confidence level plus one justification line per rule that fired."""
    fams = _families(families)
    n_fam = len(fams)
    has_hist = any(f in HISTORICAL_FAMILIES for f in fams)
    rmse_ok = lambda lim: (
        best_rmse_m is not None and not math.isnan(best_rmse_m) and best_rmse_m <= lim
    )
    why: list[str] = []
    rmse_txt = (
        "unknown" if best_rmse_m is None or math.isnan(best_rmse_m) else f"{best_rmse_m:.1f} m"
    )
    why.append(
        f"inputs: label={label}, mean_S={mean_S:.2f}, {n_fam} famil(ies) "
        f"[{', '.join(fams) or 'none'}], best checkpoint RMSE {rmse_txt}, "
        f"{n_hist_dates} historical date(s)"
    )
    if prior_contribution is not None and not math.isnan(prior_contribution):
        why.append(f"document prior contributes {prior_contribution:.2f} (informational only)")

    if (
        label == HIGH_LABEL
        and n_fam >= HIGH_MIN_FAMILIES
        and has_hist
        and mean_S >= HIGH_MIN_S
        and rmse_ok(HIGH_MAX_RMSE_M)
    ):
        level = "high"
        why.append(
            f"high: {HIGH_LABEL}, >= {HIGH_MIN_FAMILIES} families incl. historical, "
            f"mean_S >= {HIGH_MIN_S}, RMSE <= {HIGH_MAX_RMSE_M:g} m"
        )
    elif n_fam >= MEDIUM_MIN_FAMILIES and mean_S >= MEDIUM_MIN_S and rmse_ok(MEDIUM_MAX_RMSE_M):
        level = "medium"
        why.append(
            f"medium: >= {MEDIUM_MIN_FAMILIES} families, mean_S >= {MEDIUM_MIN_S}, "
            f"RMSE <= {MEDIUM_MAX_RMSE_M:g} m"
        )
    else:
        level = "low"
        why.append("low: does not meet the medium criteria (families, mean_S or georef RMSE)")

    if n_hist_dates <= 1 and level != "low" and LEVELS.index(level) > LEVELS.index(SINGLE_DATE_CAP):
        level = SINGLE_DATE_CAP
        why.append(f"capped at {SINGLE_DATE_CAP}: only a single historical date supports it")
    if label in LOW_CAP_LABELS and level != "low":
        level = "low"
        why.append(f"capped at low: label {label}")
    elif label in LOW_CAP_LABELS:
        why.append(f"label {label} caps confidence at low")
    if not blind_test_passed and level != "low":
        level = "low"
        why.append("capped at low: Phase 4 blind-test gate failed")
    elif not blind_test_passed:
        why.append("Phase 4 blind-test gate failed: maximum confidence is low")
    return Confidence(level, why)


GATE_FILE = "blind_test_gate.json"


def resolve_blind_test(flag: bool | None, public_dir) -> tuple[bool, str]:
    """Blind-test outcome -> (passed, human-readable source).

    Explicit flag wins; else ``<public_dir>/blind_test_gate.json`` (status PASS -> passed;
    FAIL / NOT_RUN / unreadable -> failed); else assume failed.
    """
    import json
    from pathlib import Path

    if flag is not None:
        return flag, "explicit command-line flag"
    p = Path(public_dir) / GATE_FILE
    if not p.exists():
        return False, f"no {GATE_FILE} and no flag, assuming FAILED (confidence capped low)"
    try:
        status = str(json.loads(p.read_text()).get("status", "")).upper()
    except (OSError, ValueError, AttributeError):
        return False, f"{GATE_FILE} unreadable, treating as FAILED"
    return status == "PASS", f"{GATE_FILE} (status {status or 'missing'})"
