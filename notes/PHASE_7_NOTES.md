# Phase 7 notes: write-up and reproducible run-all (ticket 18)

## What exists
- `sbf run-all` (`src/sbf/pipeline/{core,stages}.py`): declarative stage list, up-to-date check via
  input size+mtime fingerprints in `data/interim/run_manifest.json`, `--from/--only/--dry-run/--force/--skip`.
- `sbf.catalog.replay.ReplayClient`: answers catalogue calls from `data/raw/catalog/*.json` +
  `manifest.jsonl` (hash of endpoint + payload). Needs the same `config/aois.geojson` as the live search.
- `sbf report writeup` / `sbf report check-public` (`report/writeup.py`, `report/public_check.py`).
- End-to-end synthetic test: `tests/pipeline/test_end_to_end.py` (about 16 s).

## Human-run checklist, in order
1. **Credentials (gate).** Request M2M access; set `USGS_USERNAME`, `USGS_M2M_TOKEN` (notes/PHASE_1).
2. `sbf catalog search --aoi REG --aoi INE`; check the field mapping in `sbf/catalog/search.py`
   against `data/raw/catalog/`. `sbf catalog report` then **Gate 1: Phase 1 go/no-go**
   (exit 1 NO_GO, 3 NEEDS_REVIEW: fill `usable` in `outputs/public/browse_review.csv`).
3. `sbf ingest declass browse`, `availability`, `download --dry-run`, `download`; `order-list`
   is a list only. **Gate 2: any purchase ($30/frame) needs explicit approval.**
4. **Gate 3: GEE credentials.** `earthengine authenticate`, `GEE_PROJECT`, `sbf ingest modern export --aoi REG`
   and `--aoi INE`; fill the visible/ambiguous/not-visible table in `notebooks/02_modern_detectability.ipynb`
   (or write `outputs/public/detectability.csv` with columns `feature_id,layer,verdict`: the write-up
   then answers Q2 from it).
5. Reference refinement (`sbf ingest refine chip|import`), human digitising in QGIS. Manual GCPs for
   frames that fail auto-georeferencing: `sbf georef qgis-export|qgis-import`.
6. **Gate 4: exclusions mask.** `sbf detect draft-exclusions`, edit `config/exclusions.geojson` from
   imagery, set it reviewed. Metrics under a draft mask carry `exclusions_status = draft`.
7. DINOv2 weights (network, one time): run `sbf detect embedding embed ...` once so
   `~/.cache/huggingface` holds `facebook/dinov2-small`; otherwise `detect-embedding` is skipped.
8. **Gate 5: Phase 5.** Place PDFs in `data/raw/docs/`; `sbf docs ingest`; `sbf docs extract --dry-run`;
   **spend approval**, `sbf docs extract`; `sbf docs export-review`; review >= 30 rows (`accept` y/n).
9. `sbf run-all` (or `make all`). Read `outputs/public/report.md`, `blind_test.md`,
   `georef_summary.md`. **Gate 6: Phase 4 blind test.** On FAIL the report states a negative result
   and fusion is skipped; the human decides what happens next.
10. **Gate 7: before restricted outputs from real data.** Human sign-off, set `RESTRICTED_KEY`
    (age) if available, then `sbf run-all --include-restricted --confirm-restricted`. Review
    `outputs/restricted/` (ranking, sheets, briefing.md). Never copy restricted files out of that tree.
11. Run `sbf report check-public` once more before publishing anything from `outputs/public/`.

## Decisions and caveats
- Gates: catalogue NO_GO/NEEDS_REVIEW stops the pipeline (exit 4/3). `--continue-past-gates` writes
  `outputs/public/NOT_VALID.txt` and stamps `report.md`. A blind-test FAIL does not stop the pipeline
  (negative result is a valid output) but blocks fusion (SPEC 9.2).
- `report.md` answers: positive / negative / not yet answered. Q2 has no automatic visual test: it is
  "positive (indicative)" only when the blind-test gate PASSED, else "not yet answered" unless a human
  `detectability.csv` exists. Q5 comes from `prior_eval_summary.json` (written by `docs-prior-eval`).
- Leak guard (`report/public_check.py`): adjacent lat/lon or easting/northing pairs inside AOI ranges
  and tile IDs must match a reference feature, AOI vertex or catalogue footprint vertex (tolerance
  0.003 deg, tile margin 64 m). Violations print file:line and kind only, never values.
- Up-to-date checks ignore code changes. Use `--force` after upgrading code.
- Reproducibility gaps: (a) `catalog-replay` replays `search` and, if recorded, `download-options`
  responses; it cannot replay downloads, GEE exports, document extraction (API, cached in
  `data/interim/docs/cache`) or human steps (browse review, refinements, exclusions, review.csv).
  (b) Landsat/S2/SPOT counts stay `not_run`/`unverified` (no counter wired). (c) `prior_eval.md` and
  `fusion_summary.md` now cover all AOIs in one file. (d) The embedding stage is untested against real
  weights (skipped in tests). (e) Texture on `s2_median` uses band B4; modern layers other than
  `s2_median`, SAR and DEM are not scored with texture.
- Integration bugs fixed on the way (see git diff): `sbf.georef` exported no `load_registry`
  (fuse/report CLIs failed), polygon reference features crashed `catalog.report.coverage_map` and
  `georef.overlay.gz_in_frame`, `ranking.csv` and `fusion_summary.md` were overwritten per AOI.
