"""Resolution class from camera/mission text (SPEC section 4)."""

from __future__ import annotations

import re

CLASSES = ("<=3m", "3-10m", ">10m", "unknown")

# Nominal ground resolution in metres (low, high) per camera/mission, from SPEC section 4
# and public USGS Declass descriptions. A system is classed by its WORST (high) value so that
# a range straddling a boundary is never promoted to a finer class than it can guarantee.
# Order matters: more specific patterns first (KH-4B before KH-4A before KH-4).
RESOLUTION_TABLE: list[tuple[str, re.Pattern, tuple[float, float]]] = [
    # USGS "Camera Resolution" metadata values (seen live 2026-11-21). They are checked first
    # because the dataset alias "corona2" would otherwise hit the generic CORONA row below.
    # Mapping per the USGS Declass 1 data dictionary; cross-checked against mission numbers in
    # the first live catalogue (1101-1117 = KH-4B -> Stereo High, 10xx = KH-4A -> Stereo Medium,
    # 9xxxA = ARGON -> Vertical Low). Confirm "Stereo Low"/"Vertical High" if they ever appear.
    ("KH-4B (Stereo High)", re.compile(r"\bstereo\s+high\b", re.IGNORECASE), (1.8, 1.8)),
    ("KH-4A (Stereo Medium)", re.compile(r"\bstereo\s+medium\b", re.IGNORECASE), (2.75, 2.75)),
    ("KH-4 (Stereo Low)", re.compile(r"\bstereo\s+low\b", re.IGNORECASE), (7.5, 7.5)),
    ("LANYARD (Vertical High)", re.compile(r"\bvertical\s+high\b", re.IGNORECASE), (1.8, 1.8)),
    ("ARGON (Vertical Low)", re.compile(r"\bvertical\s+low\b", re.IGNORECASE), (140.0, 140.0)),
    (
        "KH-9 pan (2 to 4 feet)",
        re.compile(r"\b2\s*to\s*4\s*f(ee|oo)t\b", re.IGNORECASE),
        (0.6, 1.2),
    ),
    ("KH-4B", re.compile(r"KH[\s_-]?4[\s_-]?B", re.IGNORECASE), (1.8, 1.8)),
    ("KH-4A", re.compile(r"KH[\s_-]?4[\s_-]?A", re.IGNORECASE), (2.75, 2.75)),
    ("KH-4", re.compile(r"KH[\s_-]?4(?![\w])|CORONA", re.IGNORECASE), (2.0, 7.5)),
    ("KH-7", re.compile(r"KH[\s_-]?7|GAMBIT", re.IGNORECASE), (0.6, 1.2)),
    (
        "KH-9 mapping",
        re.compile(r"KH[\s_-]?9.*(mapping)|(mapping).*KH[\s_-]?9", re.IGNORECASE),
        (6.0, 9.0),
    ),
    (
        "KH-9 pan",
        re.compile(
            r"KH[\s_-]?9.*(panoramic|\bpan\b)|(panoramic|\bpan\b).*KH[\s_-]?9", re.IGNORECASE
        ),
        (0.6, 1.2),
    ),
    ("ARGON", re.compile(r"ARGON|KH[\s_-]?5(?![\w])", re.IGNORECASE), (140.0, 140.0)),
    ("LANYARD", re.compile(r"LANYARD|KH[\s_-]?6(?![\w])", re.IGNORECASE), (1.8, 1.8)),
]


def match_system(*texts: str | None) -> str | None:
    """Name of the first matching system in RESOLUTION_TABLE, or None."""
    blob = " | ".join(t for t in texts if t)
    for name, pat, _ in RESOLUTION_TABLE:
        if pat.search(blob):
            return name
    return None


def class_from_metres(high: float) -> str:
    return "<=3m" if high <= 3 else "3-10m" if high <= 10 else ">10m"


# USGS labels KH-4 (Mural, missions 9031-9062, 1962-63, ~7.5 m) "Stereo Medium" like KH-4A
# (missions 10xx, ~2.75 m); seen live 2026-11-21 (mission 9062 = 1963-12-26 "Stereo Medium").
# A numeric 90xx mission therefore overrides the label, rounding to the coarser class.
_KH4_MURAL_MISSION = re.compile(r"^\s*90(3[1-9]|[45]\d|6[0-2])(?:-\d+)?\s*$")


def resolution_class(*texts: str | None) -> str:
    """`<=3m` | `3-10m` | `>10m` | `unknown` from any camera/mission/dataset text."""
    if any(t and _KH4_MURAL_MISSION.match(str(t)) for t in texts):
        return class_from_metres(7.5)
    name = match_system(*texts)
    if name is None:
        return "unknown"
    high = next(r for n, _, r in RESOLUTION_TABLE if n == name)[1]
    return class_from_metres(high)


RANK = {c: i for i, c in enumerate(CLASSES)}
