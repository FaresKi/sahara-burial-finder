# Phase 2 notes: modern baseline stack (ticket 05)

## Human steps (credentials are a human gate; agents never do these)
1. Make sure Phase 1 is a GO.
2. Create/choose a Google Cloud project registered for Earth Engine (non-commercial) and note its ID.
3. `uv run earthengine authenticate` (browser login), then `export GEE_PROJECT=<project-id>`.
4. Dry check of dataset IDs (no export): `uv run sbf ingest modern export --aoi REG --verify-only`.
   Any ID that does not verify is reported and skipped; fix the candidate list in
   `src/sbf/ingest/modern/layers.py` (never fill gaps by hand). In particular check that PALSAR
   bands are HH/HV and that values are DN (the build step converts DN to dB with 10*log10(DN^2)-83;
   if the collection is already in dB the step is skipped automatically, see the layer provenance).

## Run order
```
uv run sbf ingest modern export --aoi REG        # raw GeoTIFFs -> data/raw/modern/REG/ + export_report.json
uv run sbf ingest modern export --aoi INE
uv run sbf ingest modern build-stack --aoi REG   # local, offline -> data/processed/modern/REG/*.tif + stack.json
uv run sbf ingest modern build-stack --aoi INE
```
Then open `notebooks/02_modern_detectability.ipynb`, look at the chips and fill in the visible /
ambiguous / not-visible table.

## Design notes
- Export uses `getDownloadURL` with tiling (not Drive): synchronous, scriptable, aligned to the final grid.
  A full REG export is several hundred small requests; failed tiles are retried 3 times, a failing
  product is recorded in `export_report.json` and the rest continue. Re-running overwrites raw files.
- Time window: last 2 years ending today (`--end-date` to pin it for reproducibility; the window is in the report).
- Common grid: `sbf.ingest.modern.modern_grid(aoi_id) -> (transform, width, height, crs)`; origin snapped
  outward to absolute multiples of 10 m in EPSG:32631. Ticket 09's tile grid must use origins/steps that are
  multiples of 10 m to be pixel-aligned.
- Coarse sources (PALSAR 25 m, DEM 30 m) are bilinearly resampled to 10 m; slope and roughness are computed
  on the resampled DEM, so they carry 30 m information content only. Roughness = std of elevation in a 5x5 px window.
- Raw layout: `data/raw/modern/{AOI}/{product}.tif` (float32, nodata -9999, band descriptions set).
  Stack: `data/processed/modern/{AOI}/{layer}.tif` (COG, float32, NaN nodata) + `.prov.json` + `stack.json`.
- Memory: REG is about 4800 x 7800 px; build-stack processes one layer at a time (peak a few hundred MB).

## Declass acquisition (ticket 04)
Code: `sbf.ingest.declass` (CLI `declass_app`, to be mounted as `sbf ingest declass`), M2M helpers in
`sbf.catalog.download`. Status: done, tested with fixtures/fake HTTP; **no live run** (credentials gate).

Commands (all read `outputs/public/coverage.csv`):
- `browse`: `data/raw/declass/{entity_id}/browse.<ext>` for every coverage row (+ `.prov.json`); feeds the
  browse review behind `outputs/public/browse_review.csv` (see PHASE_1_NOTES). Failures:
  `data/interim/declass/browse_failures.json`, exit 1.
- `availability`: M2M `download-options` for every frame; rewrites `coverage.csv` availability
  (`scanned` = some product available, `needs_order` = products listed, none available). Frames with no
  answer keep the scene-search guess (`availability_source` = `scene-search`). Run before `download`/`order-list`.
- `download [--feature GZ1 ...] [--dry-run]`: scanned frames intersecting reference features;
  `download-options` -> highest-res product (largest filesize among available, excluding browse/thumb names)
  -> `download-request` (polls `download-retrieve` if preparing; otherwise a "re-run later" failure) ->
  streamed to `<name>.part` with HTTP Range resume -> verified (size from `download-options.filesize`, and
  md5/sha256 if the API supplies a checksum; mismatch discards the file) -> `<file>.prov.json`
  (source = dataset, source_id = entity_id, step `m2m download <product>`) -> tgz/zip extracted into the
  entity folder. Extraction validates every member first and rejects absolute paths, `..`, symlinks and
  non-regular files for the whole archive. Failures never create files; `download_failures.json`, exit 1.
  `--dry-run` lists frames, no network, no credentials.
- `order-list`: `outputs/public/order_list.csv` (rank, entity_id, dataset, acquisition_date,
  resolution_class, features, n_features, score, estimated_cost_usd, cumulative_cost_usd, browse_url) for
  `needs_order` frames over reference features; prints the total. **Never places an order**: purchasing is a
  human gate. `unknown` availability frames are not listed (run `availability` first).

Decisions / assumptions:
- Score = resolution weight (<=3m 3, 3-10m 2, unknown 1, >10m 0) + date weight (pre-1967 3, 1967-68 2,
  later 1, undated 0) + number of reference features covered. Weights are heuristics, not from SPEC.
  Ties: earlier date first. Cost: `COST_PER_FRAME_USD = 30` (SPEC 0.1 / source table).
- UNVERIFIED M2M mapping (`PRODUCT_FIELDS`, `DOWNLOAD_FIELDS`, `REQUEST_BUCKETS`, `RETRIEVE_BUCKETS` in
  `sbf/catalog/download.py`): product `id`/`productName`/`available`/`filesize` (assumed bytes),
  request response `availableDownloads`/`preparingDownloads`/`failed`, retrieve `available`/`requested`,
  checksum key names (M2M documents none, so verification is size-only unless one appears). Also unknown:
  whether download URLs need an auth header (none is sent) and whether "highest resolution" is
  distinguishable by filesize. Check against the first raw `download-*` responses in `data/raw/catalog/`.
- Entity IDs are validated as safe path components before being used as folder names.

## Reference refinement (ticket 06)

Human-in-the-loop step. `config/reference_features.geojson` (SPEC-sourced) is never modified; refinements
are appended to `config/reference_refinements.geojson` (one Feature per refinement, EPSG:4326; properties
`feature_id, geometry_type, method, evidence{layer|entity_id, raster_path, screenshot_path, sha256},
refined_by, refined_at, notes, supersedes_precision`). `load_reference_features(refined=True)` applies the
latest record per feature_id and adds `refined`, `original_geometry` (WKT), `refinement_evidence`;
`refined=False` gives the originals. The harness already uses Polygon geometry as is.

Workflow: `sbf ingest refine chip --feature TA1 --raster R.tif` -> inspect/digitise in QGIS ->
`sbf ingest refine import --feature TA1 --file poly.geojson --evidence-raster R.tif --screenshot S.png`.
BASE has no geometry: the human supplies `--center-lon/--center-lat` to the chip (labelled "human-supplied
centre"). Validation refuses missing raster/screenshot, geometry outside the AOI or raster bounds, and
moving an IAEA-precision feature > 500 m (`--max-move-m`) without `--force-note`.
Agents may propose a geometry, but `refined_by` must be truthful (`agent` unless a human digitised it).
No real refinements exist yet: they need real imagery plus a human.
