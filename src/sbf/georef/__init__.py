"""Phase 3: georeferencing of historical frames.

Seams for ticket 08 (strips, QGIS fallback, batch): `coarse.coarse_candidates`,
`gcps.find_gcps`, `fit.select_model` / `fit.fit_transform`, `warp.warp_frame`,
`pipeline.solve_frame` (pure) and `pipeline.run_frame` (IO).
"""

from sbf.georef.registry import load_registry, load_support

__all__ = ["load_registry", "load_support"]
