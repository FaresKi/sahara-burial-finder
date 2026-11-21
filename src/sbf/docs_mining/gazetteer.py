"""Anchor gazetteer (ticket 15): resolve anchor names from documents to public points.

``config/anchors.geojson`` is seeded ONLY with public reference points (GZ1-GZ4, TA1, PO1;
BASE stays unresolved, null geometry, until refined) plus the approximate Reggane town point
taken from an existing project file. Coordinates are never invented: an entry with null
geometry has ``needs_human_location: true`` and resolves to an unresolved anchor.

Reference-feature anchors take their geometry from ``load_reference_features(refined=True)``
when available, so refinements flow through; the stored geometry is the fallback.

Alias normalisation: Unicode NFKD accent stripping, case folding, hyphens/underscores/punctuation
to spaces, whitespace collapsed, and a space between letters and digits removed
("GZ 1" == "GZ-1" == "gz1"). ``resolve_anchor`` matches the whole text first, then falls back to
the longest alias occurring as whole words in the text (so "ground zero of Gerboise Bleue"
resolves). Ties between different anchors are ambiguous and resolve to None.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

from shapely.geometry import Point, shape

from sbf.core.aoi import load_reference_features
from sbf.core.config import Settings, load_settings

ANCHORS_FILE = "anchors.geojson"


def normalise_alias(text: str) -> str:
    """Case/accent/hyphen-insensitive key for anchor names."""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c)).casefold()
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    t = re.sub(r"\b([a-z]{1,3}) (\d+)\b", r"\1\2", t)
    return t


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    aoi_id: str
    aliases: tuple[str, ...]
    geometry: Point | None  # lon/lat EPSG:4326
    precision: str
    source: str
    needs_human_location: bool = False
    reference_feature_id: str | None = None

    @property
    def resolved(self) -> bool:
        return self.geometry is not None and not self.geometry.is_empty


@dataclass
class Gazetteer:
    anchors: list[Anchor]
    _index: dict[str, set[str]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {a.anchor_id: a for a in self.anchors}
        for a in self.anchors:
            for al in (a.anchor_id, *a.aliases):
                self._index.setdefault(normalise_alias(al), set()).add(a.anchor_id)

    def get(self, anchor_id: str) -> Anchor | None:
        return self._by_id.get(anchor_id)

    def resolve(self, text: str) -> Anchor | None:
        """Anchor whose alias matches ``text`` (None if no match or ambiguous)."""
        key = normalise_alias(text or "")
        if not key:
            return None
        if key in self._index:
            ids = self._index[key]
            return self._by_id[next(iter(ids))] if len(ids) == 1 else None
        padded = f" {key} "
        best_len, best_ids = 0, set()
        for alias, ids in self._index.items():
            if alias and f" {alias} " in padded:
                if len(alias) > best_len:
                    best_len, best_ids = len(alias), set(ids)
                elif len(alias) == best_len:
                    best_ids |= ids
        return self._by_id[next(iter(best_ids))] if len(best_ids) == 1 else None


def load_gazetteer(settings: Settings | None = None) -> Gazetteer:
    s = settings or load_settings()
    data = json.loads((s.config_dir / ANCHORS_FILE).read_text())
    ref = load_reference_features(s, refined=True).set_index("feature_id")
    anchors = []
    for f in data["features"]:
        p = f["properties"]
        geom = shape(f["geometry"]) if f.get("geometry") else None
        rid = p.get("reference_feature_id")
        if rid and rid in ref.index:
            g = ref.at[rid, "geometry"]
            if g is not None and not g.is_empty:
                geom = g
        anchors.append(
            Anchor(
                anchor_id=p["anchor_id"],
                aoi_id=p["aoi_id"],
                aliases=tuple(p.get("aliases", [])),
                geometry=geom,
                precision=p["precision"],
                source=p["source"],
                needs_human_location=bool(p.get("needs_human_location", geom is None)),
                reference_feature_id=rid,
            )
        )
    return Gazetteer(anchors)


def resolve_anchor(text: str, settings: Settings | None = None) -> Anchor | None:
    """Resolve anchor text with the project gazetteer; None when unknown or ambiguous.

    A returned anchor may still be unresolved in space (``anchor.resolved`` False,
    ``needs_human_location`` True): callers report it rather than guess a location.
    """
    return load_gazetteer(settings).resolve(text)
