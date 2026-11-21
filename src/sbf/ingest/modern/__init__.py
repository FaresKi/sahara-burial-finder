"""Modern baseline stack (ticket 05): GEE export (human-gated) + local co-registration."""

from sbf.ingest.modern.exporter import (
    EarthEngineExporter,
    ExportReport,
    GeeExporter,
    GeeNotConfiguredError,
    ProductResult,
)
from sbf.ingest.modern.fake import FakeExporter
from sbf.ingest.modern.grid import grid_from_bounds, modern_grid
from sbf.ingest.modern.layers import LAYERS, PRODUCTS
from sbf.ingest.modern.process import (
    build_stack,
    local_std,
    read_manifest,
    reproject_to_grid,
    slope_degrees,
    terrain_ruggedness_index,
)

__all__ = [
    "LAYERS",
    "PRODUCTS",
    "EarthEngineExporter",
    "ExportReport",
    "FakeExporter",
    "GeeExporter",
    "GeeNotConfiguredError",
    "ProductResult",
    "build_stack",
    "grid_from_bounds",
    "local_std",
    "modern_grid",
    "read_manifest",
    "reproject_to_grid",
    "slope_degrees",
    "terrain_ruggedness_index",
]
