# Phase 3 notes: georeferencing (tickets 07: one frame, 08: strips, QGIS, batch)

Status: implemented and tested on synthetic data only. **No live run** (needs real frames from ticket 04
and the S2/DEM stack from ticket 05).

## Run
```
uv run sbf georef frame --entity-id <ID> [--raster PATH] [--aoi REG|INE] \
    [--method auto|poly1|poly2|poly3|tps] [--reference VHR.tif] [--reference-layer pan|B4|hillshade] [--seed N]
```
Reads `outputs/public/coverage.csv` (footprint, camera) and `data/raw/declass/{ID}/*.tif` (parts assembled, else the largest non-browse tif).
Writes `data/processed/historical/{ID}/`: `{ID}_georef.tif` (COG + `.prov.json`), `georef_report.json`, `gcps.csv`
(id, px, py, x, y, role, residual_m), and `outputs/public/georef/{ID}_gz.png` (GZ1-GZ4 chips, public reference positions only).
Exit 2 when no model could be fitted (report still written).

## Design
- Pixel coordinates everywhere are the rasterio "corner" convention (pixel edge = integer). cv2 centre convention is converted only in `gcps.py`.
- Coarse placement: footprint corners sorted CCW, 8 dihedral variants -> homography frame px -> UTM. SIFT/ORB are
  rotation invariant, so inlier count cannot tell a 90-degree-wrong order from the right one; the winner is the
  variant whose RANSAC residual affine is near identity (|rot| <= 20 deg, scale 0.6-1.6), ties by inlier count.
  Mirrored variants simply fail to match. (Choosing a rotated variant biased GCPs by ~0.5 px via SIFT's sub-pixel
  offset; fixed by this rule, covered by a test against ground truth.)
- Matching happens on the frame warped into the reference grid at reference resolution (CLAHE + INTER_AREA
  downsample first; `scale` maps back to full-res px). SIFT first, ORB fallback when SIFT gives < 8 inliers.
  `Matcher` is a Protocol (`detect`, `match`) so LoFTR can be added.
- RANSAC: affine, 5 reference px threshold (generous so real nonlinear distortion survives), then one outlier trim
  against a poly1/poly2 fit (max(3 robust sigma, 10 m)), then seeded 20% checkpoint split.
- Models: own least-squares polynomials (order 1-3, need n_train >= 2 x terms per axis: 6/12/20), TPS
  (`scipy.RBFInterpolator`, smoothing configurable, needs >= 10). `auto` picks the lowest checkpoint RMSE; a more
  complex model must win by >5%. Reported RMSE is from the train-only model; the warp uses a refit on ALL GCPs.
- Warp: forward model inverted on a 40x40 lattice (TPS), evaluated per 1024-px block on a 16-px lattice, upsampled,
  `cv2.remap` bilinear. Output res = estimated native ground res (2 decimals), grid snapped to multiples of it.
- Stable mask (`mask.stable_terrain_mask`): dune = S2 local std (21 px = 210 m) in [0.008, 0.045] AND slope >= 3 deg
  (roughness >= 1.5 m if no slope), dilated 2 px. Applied to reference keypoints only. **Thresholds are
  uncalibrated heuristics**: check on the first real chips. If < 5% of pixels survive the mask is ignored (warning).
- Reference: S2 median mean of B2,B3,B4,B8 (`pan`), a single band, or DEM hillshade (unmasked). Esri/Google tiles are never used.

## Seams for ticket 08
`coarse.coarse_candidates`, `gcps.find_gcps` (takes any quad + array), `fit.select_model/fit_transform`,
`warp.warp_frame(src, model, resolution_m=, bounds=)` (common bounds => mosaic strips), `pipeline.solve_frame`
(pure, returns report + models + GCP table), `pipeline.write_gcps_csv`, `pipeline.is_panoramic`.
Panoramic frames get `panoramic_needs_strips` and are still solved frame-wide.

## Known limitations / open
- No real-data validation: SIFT may find few features on 1960s film vs S2 (different sensors, dune-dominated desert).
- Frame-wide fit on panoramic frames will be poor by construction (SPEC section 7); ticket 08 solves them per strip.
- `included_for_change_detection` uses checkpoint RMSE <= 20 m; with few GCPs the checkpoint set is tiny, hence `few_gcps` (< 20).
- GZ offsets (`gz_offsets` in the report) are placeholders for the human; dx/dy/measured_by stay null.
- Warp nodata is 0 for integer frames (scan borders are black anyway); float frames use NaN.

# Ticket 08

Status: synthetic tests only; **live run pending** real frames (04) and the S2 stack (05).

## Run
```
uv run sbf georef frame --entity-id ID [--strips/--no-strips] [--n-strips 8] [--strip-overlap 0.1] [--strip-axis auto|x|y]
uv run sbf georef qgis-export --entity-id ID          # processed/historical/ID/qgis/ID_seed.points + README.md
uv run sbf georef qgis-import --entity-id ID --points FILE.points [--method auto|polyN|tps]
uv run sbf georef batch [--aoi REG|INE] [--only-missing]  # -> data/processed/historical/registry.parquet, outputs/public/georef_summary.md
```
`frame` and `batch` use strips automatically for panoramic cameras (KH-4/4A/4B/7, KH-9 pan). Use
`sbf.georef.registry.load_registry(included_only=True)` from ticket 11.

## Multi-part scans (`assemble.py`)
- Parts are found by `DEFAULT_PART_PATTERN = ^(?P<base>.+?)_(?P<part>[a-z])\.tiff?$` (`_a`, `_b`, ...), sorted by letter.
  **UNVERIFIED against a real download**: check the actual file names and override via `part_pattern`.
  The letter order is assumed to follow the strip; the placement (right/left/below/above) of each part relative to the
  previous is found by trying all four and keeping the most RANSAC inliers (>= 12 needed, else `AssemblyError`).
- Matching: SIFT on decimated parts (longest side <= 2500 px), overlap zone = 35% of the edge, RANSAC similarity;
  scale must be 1 +- 3% and rotation <= 2 deg (< 0.1 deg is treated as noise -> integer translation). Composite is blockwise at full
  resolution with distance-to-edge feathering into `processed/ID/assembled.tif` (tiled, no georeference) + `assembly.json`
  (order, offsets, rotation, inliers, side). Seam accuracy is about the decimation factor in full-res pixels
  (`seam_accuracy_about_Npx` warning). Re-running reuses the cache unless the part files change.
- Single part: passthrough, nothing written.

## Memory
`frame_io.FrameSource` (`RasterSource` = lazy rasterio, `ArraySource`): matching uses `read_decimated` (rasterio `out_shape`,
overviews used when present); `warp.sample_source` reads only the source window of each output block;
`warp_to_file` / `strips.warp_strips_to_file` stream blocks into a tiled GeoTIFF which is then converted to COG.
Test: warp with `block=64` equals the in-memory result. Still in memory: the reference crop, the decimated frame,
GCPs. The final COG conversion is done by GDAL (streaming).

## Panoramic strips (`strips.py`)
- Frame split along `--strip-axis` (auto = the longer film dimension, the scan direction along which scale follows ~1/cos^2) into N
  strips with 10% of a strip length shared with each neighbour. NOTE the architect wording "along-track": the split axis is the one
  along which the distortion varies; check on the first real KH-4 scan that `auto` picks it.
- Frame-wide `solve_frame` runs first (seed + `candidate_rmse_m["frame_wide"]` for the report). Strips are solved centre-out; each strip is seeded
  with the nearest solved strip's model extrapolated (local affine over its own extent), the first with the frame-wide affine
  (a single global affine cannot follow the scale change: the plausible-residual check rejected the outer strips in the synthetic test).
- Models are in full-frame pixel coordinates. Strip with < 8 GCPs / no model -> inherits the nearest strip (else frame-wide) -> `strip_fallback`.
- Overall RMSE = pooled checkpoint residuals (each scored by its own strip's train-only model). Warp: one grid (union bounds, finest
  strip resolution), linear weight ramps across the overlap, `rmse` flags as before. `gcps.csv` gets an extra `strip` column.
- CAVEAT (seen in the synthetic test): checkpoint RMSE only measures where GCPs are. A frame-wide fit had 4 m checkpoint RMSE but
  85 m error against truth over the whole frame. Rely on the GZ overlays and GCP spread, not on RMSE alone.

## QGIS fallback (`qgis.py`)
`.points`: `#CRS:` line + `mapX,mapY,sourceX,sourceY,enable,dX,dY,residual`; QGIS stores `sourceY = -row` (handled both ways; all-positive
file accepted with a note; mixed signs rejected). Disabled points skipped, non-UTM `#CRS` reprojected. Import calls
`run_frame(manual_gcps=...)`: no matching, no reference needed, no outlier trim, seeded 20% checkpoints, `gcp_source = "manual_gcps"`
(registry `method = manual_gcps`). No `.qgs` project is written: the Georeferencer keeps its state outside the project file, so the
generated README is the instruction. **Unverified in a real QGIS**: row sign and half-pixel convention (we assume pixel edge = integer).

## Registry (`registry.py`)
Columns: entity_id, dataset, camera, acquisition_date, era, aoi, method, n_gcps, checkpoint_rmse_m, resolution_m,
included_for_change_detection, warnings (`;`-joined), georef_file, status (ok|unsolved|failed), error.
Rule: RMSE > 20 m excluded, 10-20 m included and flagged (`rmse_gt_10m`), failed/unsolved/no georef file excluded.
`era`: `pre_1967` (< 1967), `dismantling_1967` (1967), `post_1967` (> 1967), `undated`; same boundary as `sbf.catalog.go.era`
(whose labels use hyphens). Failure tracebacks go to `processed/historical/ID/georef_error.txt`.

## Open
- Strip count/overlap defaults (8 / 10%) and the `MIN_STRIP_PX = 64` (small-image px) floor are untuned: real frames are far larger than the synthetic ones.
- The seam error of assembled parts and per-block lattice phase give pixel-level (not sub-pixel) differences between in-memory and streamed warps.
- Overlay chips read windows from the output file; GZ detection is unchanged.

# Honest accuracy (spatial CV, coverage, support mask) - follow-up to ticket 08

Problem: a random 20% hold-out puts checkpoints next to training GCPs; a frame-wide fit scored 4 m yet was ~85 m wrong where GCPs were sparse.
All numbers below are from synthetic tests only.

- **Blocked hold-out** (`fit.split_checkpoints(..., strategy="blocked"|"random")`, default blocked): the frame (or strip extent) is cut into a k x k grid
  (k = 5, shrunk to >= 2 until >= 5 blocks hold GCPs); whole blocks (seeded order) are held out until ~20% of the GCPs. `checkpoint_rmse_m` now comes
  from this split. Without `px` the split falls back to random.
- **`cv_rmse_m`** = spatial 5-fold CV (blocks dealt into folds; pooled out-of-fold residuals; folds whose training set is too small for the model are skipped).
  This is the PRIMARY accuracy number and also what `select_model` ranks candidates by (blocked checkpoint RMSE if CV is not computable). `random_checkpoint_rmse_m`
  = old-style random 20% split with the chosen method, for comparison only.
- **Coverage**: `gcp_hull_fraction` (convex hull area / frame area, px space), `gcp_block_occupancy` (share of a 5x5 frame grid with >= 3 GCPs). Per strip on
  `StripReport` (strip extent), overall on pooled non-fallback GCPs over the whole frame.
- **Gate** (`models.apply_quality_gate`, `MIN_BLOCK_OCCUPANCY = 0.4`): included iff `cv_rmse_m <= 20` AND `gcp_block_occupancy >= 0.4`. Warnings:
  `rmse_gt_20m_excluded`/`rmse_gt_10m` (on cv), `no_cv`, `low_gcp_coverage`, `cv_vs_checkpoint_gap` (cv > 2x blocked checkpoint). Strips: cv pools every
  strip's out-of-fold residuals.
  CAVEAT: CV tests interpolation between GCP blocks, not extrapolation into empty regions; that part is covered by the occupancy rule and the support mask.
  Occupancy 0.4 is an untuned heuristic; 5x5 with >= 3 GCPs per cell needs >= ~30 well-spread GCPs, so GCP-poor frames are now excluded by design.
- **Support mask** (`support.py`): `{ID}_support.tif` (uint8 COG, 1 = supported) on exactly the `{ID}_georef.tif` grid; supported = maps back to a frame pixel within
  10% of the frame diagonal (`SUPPORT_DIST_FRAC`) of the convex hull of the GCPs. Strips: per-strip hull and 10% of the strip diagonal, restricted to the strip's
  range; strips with a fallback model (no own GCPs) never support. `registry.load_support(entity_id, settings=None) -> (uint8 array, Affine) | None`.
  Change detection must ignore pixels where the mask is 0.
- **Registry**: columns ADDED at the end (existing ones and order unchanged): `cv_rmse_m, random_checkpoint_rmse_m, gcp_hull_fraction, gcp_block_occupancy,
  support_file`. `included_for_change_detection` follows the new gate (reports without `cv_rmse_m`, e.g. written by older code, are therefore excluded until re-run).
  `georef_summary.md` headline is the CV RMSE.
- Synthetic clustered-GCP test (4 clumps of 12 GCPs in a 1000 px frame, TPS, smooth non-polynomial distortion): random checkpoint RMSE 1.5 m, blocked checkpoint 5.4 m,
  cv_rmse_m 43.8 m, occupancy 0.28, hull 0.52, excluded. Well covered (100 uniform GCPs): cv 9.4 m, occupancy 0.72, included.
