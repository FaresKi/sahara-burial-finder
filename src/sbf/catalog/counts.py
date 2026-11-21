"""Modern/other-sensor scene counts (ticket 03).

`SceneCounter` is defined here so sbf.catalog does not depend on sbf.ingest. A GEE-backed
implementation can be wired later. Nothing is ever filled in: no counter => `not_run`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

from shapely.geometry.base import BaseGeometry

SENSORS = ("landsat", "sentinel2")  # counted via a SceneCounter
LANDSAT_START = "1972-01-01"  # SPEC section 6, Phase 1


class SceneCounter(Protocol):
    def count_scenes(self, sensor: str, aoi: BaseGeometry, start: str, end: str) -> int:
        """Number of scenes of `sensor` ('landsat' | 'sentinel2') intersecting `aoi` in range."""
        ...


@dataclass
class SensorCount:
    sensor: str  # landsat | sentinel2 | spot
    aoi_id: str
    status: str  # counted | not_run | failed | unverified
    count: int | None = None
    detail: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def collect_counts(
    counter: SceneCounter | None, aois: dict[str, BaseGeometry], start: str, end: str
) -> list[SensorCount]:
    """Landsat + S2 counts per AOI plus SPOT as 'unverified' (no API access confirmed)."""
    out: list[SensorCount] = []
    for aoi_id, geom in aois.items():
        for sensor in SENSORS:
            s = max(start, LANDSAT_START) if sensor == "landsat" else start
            if counter is None:
                out.append(
                    SensorCount(sensor, aoi_id, "not_run", None, "no scene counter configured")
                )
                continue
            try:
                n = int(counter.count_scenes(sensor, geom, s, end))
                out.append(SensorCount(sensor, aoi_id, "counted", n, f"{s}..{end}"))
            except Exception as exc:  # noqa: BLE001  recorded, never filled in (SPEC 0.2)
                out.append(SensorCount(sensor, aoi_id, "failed", None, repr(exc)))
        out.append(
            SensorCount(
                "spot",
                aoi_id,
                "unverified",
                None,
                "SPOT World Heritage (CNES REGARDS) coverage not checked; no API access",
            )
        )
    return out
