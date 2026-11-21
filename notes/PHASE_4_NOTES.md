# Phase 4 notes (ticket 09: tiling, scoring harness, texture baseline)

## Decisions
- Tile lattice origin = AOI bbox SW corner (EPSG:32631) snapped DOWN to a multiple of 128 m.
  `col` grows east, `row` grows north. Not aligned to the 10 m modern grid on purpose;
  `tile_windows` rounds offsets and returns fixed-shape NaN-padded chips.
- `make_grid(..., bounds=None)` has one extra optional kwarg (restrict to a raster footprint,
  IDs identical to the full grid). `tile_geometries(tile_ids)` rebuilds polygons from IDs, so the
  harness never needs the full grid (REG at 64 m / 50 % overlap is ~3.6 M tiles).
- `score_norm` is rank-normalised per (aoi, detector, layer, size_m) (contract says without
  size_m; identical for single-size tables, and raw scores of different sizes are not comparable).
- Harness recall@K is feature-level (feature hit if any of its tiles is in the top K); tile-level
  recall and AP (tile-level) are reported too. Random expectations are analytical (hypergeometric).
- Excluded tiles are removed before ranking; a positive tile that is excluded disappears too.
- Only `sbf.detect.harness` imports `load_reference_features` (D7).

## Human gate: exclusions mask
`config/exclusions.geojson` is a DRAFT (`status: draft_needs_human_review`): one rough Reggane
town bbox, not digitised from imagery. In Ekker/Hamoudia base/roads/2026 cleanup works are still
missing. Metrics produced under a draft mask carry `exclusions_status = draft`. NOT reviewed.

## Open
- Texture on 10 m Sentinel-2: a 64 m tile is only ~6x6 px, so GLCM stats are weak there; expect the
  texture baseline to be informative mainly on historical (sub-3 m) layers. 128 m tiles help.
- Real-data results not run (no rasters yet); only synthetic integration test.

# Ticket 12: SAR and DEM anomaly detectors (`sbf.detect.sar_dem`)

- `sar_z`: per-pixel z vs ring background (default inner 30 m, outer 120 m; ring excludes the
  centre and the target footprint so 10-60 m features are not absorbed into the background).
  Mean/std by default, `robust=True` for median/1.4826*MAD. `min_sd` floor 0.25 dB (about S1
  radiometric noise) stops flat backgrounds exploding. Modern stack layers are already dB
  (stack.json units `dB`); `units="linear"` converts with 10*log10. Assumption: dB unless the
  manifest entry says linear.
- Tile score = p95 of |z| (`--agg`, `--tail both|high|low`); tiles need >= 50 % valid z pixels.
- `dem_relief`: layer `dem` = DEM - Gaussian(sigma 50 m) in metres (or disk median); layer
  `dem_roughness` = robust local z of the roughness layer. GLO-30 is ~30 m native resampled to
  10 m: micro-relief below ~30 m wavelength is not resolvable (warning logged every run).
- Memory: filters run in 1024 px blocks with halo = filter radius (equals whole-array result,
  tested); the per-pixel result goes to a temp GeoTIFF and tiles are read via `tile_windows`.
- Ring background is NaN-aware (normalised convolution); pixels with < 30 % valid ring are NaN.
- Raster resolves from `data/processed/modern/{AOI}/stack.json` when `--raster` is omitted.
- CLI: `sar_dem_app` in `sbf.detect.sar_dem_cli` (commands `sar`, `dem`); to be mounted by the
  architect.
- Open: no real-data results; a bright patch larger than the inner radius contaminates its own
  ring, and defaults are untuned on real rasters.

# Ticket 10: embedding anomaly detector
- Modules: `sbf.features.embeddings` (`TileEncoder` Protocol, `DinoV2Encoder`, `chip_to_rgb`),
  `sbf.detect.embedding` (`compute_embeddings`, `run_embedding_scores`, cache helpers),
  `sbf.detect.embedding_cli` (`embed_app`: `embed`, `score`; to be mounted by the architect).
- Embedding = DINOv2 CLS token concatenated with mean of patch tokens (768-d for dinov2-small).
  Input: single band -> `sbf.features.preprocess` (CLAHE default, applied per chip), NaN filled with
  chip median, replicated to 3 channels, resized to 224, ImageNet-normalised. fp32 (fp16 off on MPS).
  Device: mps if available, else cpu.
- Cache: `data/interim/embeddings/{aoi}/{layer}__{size}__{encoder}/part-NNNNNN.parquet`
  (`tile_id`, `emb` fixed-size list float32), atomic shards, resume skips cached tile_ids. The
  encoder name embeds the preprocess (`dinov2-small_clahe`), so changing it never mixes caches.
  Only tiles fully inside the raster bounds with >= min_valid_frac valid pixels are embedded.
  Tiles skipped for nodata are not recorded, so they are re-read (cheaply) on resume.
- Scoring: optional seeded PCA (default 64 comps) then `fit_score` for iforest/knn/maha ->
  `embed_iforest|embed_knn|embed_maha` tables. Never sees labels (D7).
- Scale warning: tile < 16 px across (e.g. 64 m on S2 10 m) warns that embeddings are mostly
  upsampling noise. Use 128 m for modern layers, 64 m for historical frames of 3 m or finer.
- Human step: the first real `embed` run downloads `facebook/dinov2-small` from the Hugging Face
  hub (network). Tests never download; the heavy test skips unless weights are already cached.
  Not run on real data yet.

# Ticket 11: change detection (`sbf.detect.change`)
- Modules: `sbf.detect.change` (naming, `select_pairs`, `prepare_pair`, `diff_scores`, `embed_scores`,
  `run_change`), `sbf.detect.change_cli` (`change_app`, mounted as `sbf detect change`: `pairs`, `diff`,
  `embed`, `run`).
- Layer names: historical `{camera_short}_{entity_id}` lower-case (kh4, kh4a, kh4b, kh7, kh9map,
  kh9pan, argon, lanyard, else hist; from `sbf.catalog.resolution.match_system`); pair layer
  `pair_{A}_{B}` (A earlier); vs modern `pair_{A}_s2_median`. `fuse.families.family_of` maps any
  `pair_*` layer of change_diff/change_embed to `hist_change` and single-date historical names to
  `hist_anomaly`: no disagreement with fuse (tested).
- Pairs: included frames only; pre_1967 x post_1967, pre_1967 x dismantling_1967, each frame vs
  `s2_median`; `--same-era` adds same-era pairs; undated frames only pair with modern. Footprint
  overlap (bounding boxes of the georef files) must be >= `--min-overlap-km2` (default 0.25).
- Co-registration: intersection of footprints (and `sbf.georef.load_support` masks when that function
  exists), coarser native resolution, finer side block-averaged (`Resampling.average`), grid aligned
  to the tile lattice origin. Then CLAHE on both and quantile (histogram) matching of B to A over
  stable terrain. S2 pan = mean(B2,B3,B4). Whole intersection is held in memory (float32): fine for
  AOI-sized pairs, not for full frames at 1 m (revisit if a real run is slow/large).
- `change_diff`: shift-min trick: per tile, 1-SSIM and normalised |A-B| minimised over shifts of B
  within +-r px, r = ceil(max(RMSE)/pixel) capped at 3 (RMSE = `cv_rmse_m` if the registry has it,
  else `checkpoint_rmse_m`; modern 0). Score = mean of within-pair ranks. Cost ~(2r+1)^2 SSIMs/tile.
- `change_embed`: cosine distance of A/B chip embeddings (harmonised arrays fed to the encoder, which
  applies its own CLAHE again). Cache layer keys include the pair grid (`{name}-g{res}m{x}x{y}`, B also
  `-hm{A}`) because B depends on A through matching. No shift search: use 128 m tiles.
- Dropped tiles (counted in the log): > 50 % unstable (dune-like, `stable_terrain_mask` computed on the
  modern stack with a 300 m context pad) or < 50 % valid pixels in both dates. If no modern stack
  exists the dune mask is skipped with a warning.
- write_scores keeps one table per (detector, layer): running a second tile size for the same pair
  replaces the first.
- Open: real runs pending (needs registry + stack); thresholds untuned; bounding-box overlap overstates
  true footprint overlap for rotated frames (NaN edges are dropped at tile level).

# Ticket 13: blind-test report and gate
- Modules: `sbf.report.blind_test` (`evaluate_gate`, `build_report`, `run_blind_test`, `positive_ranks`),
  `sbf.report.blind_test_figures`, `sbf.detect.sam_review`. CLI (added to `sbf report`): `blind-test`
  (exit 0 PASS, 4 FAIL with stop notice, 5 NOT_RUN), `sam-review ... --confirm-restricted` (exit 2 without).
- Gate (constants at top of `blind_test.py`; operationalises "clearly beats random"): some
  detector not in {random, prior}, historical layer (`family_of` in hist_anomaly/hist_change, any AOI and
  tile size) with feature recall@50 >= 0.5, recall@50 >= random + 0.2, and AP >= 2 x random AP. Modern-layer
  best is reported but never gates. NOT_RUN when no historical rows. Output `outputs/public/blind_test_gate.json`
  (status, criteria, best_rows, margin, modern_best, n_reference_features_scored, warnings, generated_at,
  exclusions_status); warns when <= 6 features scored in an AOI.
- Figures (`outputs/public/figures/`): recall@K curves, positive-rank histogram, and per-feature
  imagery + score panel where only footprint-intersecting tiles are coloured (rest = neutral mask colour;
  leak test). No AOI-wide heatmaps. Per-feature panels need a modern stack with an `s2_*` layer, else skipped.
- SAM review: `Segmenter` Protocol, `Sam2Segmenter` (checkpoint from `$SBF_SAM2_CHECKPOINT`, never downloads:
  weight download is a human step), `FakeSegmenter`. Output only `outputs/restricted/sam_review/{aoi}_{det}_{layer}_{size}m.gpkg`
  (+ age encryption when keyed). CLI chip raster defaults to the modern stack layer; `--raster` overrides.
- Open: real run pending (needs scores + human sign-off); the gate thresholds are an interpretation of SPEC and
  should be confirmed by the human; the SAM CLI path is untested against a real checkpoint.
