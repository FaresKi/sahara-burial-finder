"""Report schema and shared constants for georeferencing."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

UTM_CRS = "EPSG:32631"
RMSE_WARN_M = 10.0
RMSE_EXCLUDE_M = 20.0  # SPEC Phase 3: frames above this are excluded from change detection
FEW_GCPS = 20  # below this a `few_gcps` warning is raised
# Inclusion gate (SPEC Phase 3, hardened): spatial-CV RMSE <= RMSE_EXCLUDE_M AND at least this
# fraction of the 5x5 frame blocks holds >= 3 GCPs. Below it the fit is unconstrained over most of
# the frame whatever the checkpoints say.
MIN_BLOCK_OCCUPANCY = 0.4
CV_GAP_RATIO = 2.0  # `cv_vs_checkpoint_gap` warning when cv_rmse_m > ratio * checkpoint_rmse_m
SUPPORT_DIST_FRAC = 0.1  # support mask: within this fraction of the frame diagonal of the GCP hull

Method = Literal["poly1", "poly2", "poly3", "tps", "strips", "none"]
GcpSource = Literal["auto", "manual_gcps"]


class GZOffset(BaseModel):
    """Placeholder for a HUMAN-measured offset between the IAEA position and the visible feature.

    Never filled by code (SPEC 0.2: no invented numbers).
    """

    in_frame: bool
    dx_m: float | None = None
    dy_m: float | None = None
    measured_by: str | None = None


class StripReport(BaseModel):
    """One along-film strip of a panoramic frame (ticket 08)."""

    index: int
    axis_range_px: tuple[float, float] = Field(description="strip extent incl. overlap, frame px")
    method: str = "none"
    n_gcps: int = 0
    n_checkpoints: int = 0
    checkpoint_rmse_m: float | None = None
    train_rmse_m: float | None = None
    cv_rmse_m: float | None = None
    random_checkpoint_rmse_m: float | None = None
    gcp_hull_fraction: float | None = Field(None, description="hull area / strip area (px)")
    gcp_block_occupancy: float | None = None
    warnings: list[str] = Field(default_factory=list)  # e.g. strip_fallback
    fallback_from: str | None = Field(None, description="'strip:<i>' or 'frame' when inherited")


class GeorefReport(BaseModel):
    entity_id: str
    aoi_id: str | None = None
    method: Method
    poly_order: int | None = None
    tps_smoothing: float | None = None
    n_candidates: int = Field(0, description="matches surviving RANSAC, before dedup/trim")
    n_gcps: int = 0
    n_checkpoints: int = 0
    checkpoint_rmse_m: float | None = Field(None, description="spatially blocked hold-out RMSE")
    train_rmse_m: float | None = None
    cv_rmse_m: float | None = Field(None, description="spatial k-fold RMSE: primary accuracy")
    random_checkpoint_rmse_m: float | None = Field(
        None, description="random 20% hold-out (optimistic)"
    )
    gcp_hull_fraction: float | None = Field(None, description="GCP convex hull area / frame area")
    gcp_block_occupancy: float | None = Field(None, description="5x5 blocks with >= 3 GCPs")
    support_file: str | None = None
    candidate_rmse_m: dict[str, float | None] = Field(default_factory=dict)
    resolution_m: float | None = None
    reference: str = ""
    matcher: str = ""
    coarse_variant: int | None = Field(None, description="footprint corner ordering chosen (0-7)")
    corner_variant_inliers: list[int] = Field(default_factory=list)
    stable_fraction: float | None = None
    panoramic: bool = False
    warnings: list[str] = Field(default_factory=list)
    included_for_change_detection: bool = False
    seed: int = 0
    frame_shape: tuple[int, int] | None = None
    gcps_file: str | None = None
    georef_file: str | None = None
    overlay_file: str | None = None
    gz_offsets: dict[str, GZOffset] = Field(default_factory=dict)
    gcp_source: GcpSource = "auto"
    strips: list[StripReport] = Field(default_factory=list)
    strip_axis: str | None = None
    assembly_file: str | None = None


def apply_quality_gate(rep: GeorefReport, *, has_model: bool = True) -> None:
    """Quality warnings + `included_for_change_detection` from the honest numbers.

    Included iff cv_rmse_m <= RMSE_EXCLUDE_M and gcp_block_occupancy >= MIN_BLOCK_OCCUPANCY.
    Warnings: rmse_gt_20m_excluded / rmse_gt_10m (cv_rmse_m), no_cv, low_gcp_coverage,
    cv_vs_checkpoint_gap (cv > CV_GAP_RATIO x blocked checkpoint RMSE)."""
    cv, occ = rep.cv_rmse_m, rep.gcp_block_occupancy
    if cv is None:
        rep.warnings.append("no_cv")
    else:
        if cv > RMSE_EXCLUDE_M:
            rep.warnings.append("rmse_gt_20m_excluded")
        if cv > RMSE_WARN_M:
            rep.warnings.append("rmse_gt_10m")
        ck = rep.checkpoint_rmse_m
        if ck is not None and cv > CV_GAP_RATIO * ck:
            rep.warnings.append("cv_vs_checkpoint_gap")
    if occ is None or occ < MIN_BLOCK_OCCUPANCY:
        rep.warnings.append("low_gcp_coverage")
    rep.included_for_change_detection = bool(
        has_model
        and cv is not None
        and cv <= RMSE_EXCLUDE_M
        and occ is not None
        and occ >= MIN_BLOCK_OCCUPANCY
    )
