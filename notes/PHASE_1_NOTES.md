# Phase 1 notes (tickets 02, 03)

## Status
Code and tests are done; **no live run has happened** (no credentials, no network by design).
The live run, and therefore the go/no-go decision, is a human gate.

## Human steps before a live run
1. Request M2M API access for the USGS EROS account (ERS account -> "Machine-to-Machine" access
   request via the EROS contact form; approval is manual and can take days).
2. Generate an M2M application token (EROS Registration System -> Access Tokens) and export:
   `USGS_USERNAME`, `USGS_M2M_TOKEN` (env only, never committed; SPEC 0.3).
3. Run `sbf catalog status`, then `sbf catalog search --aoi REG --aoi INE` (1959-01-01..1984-12-31),
   then `sbf catalog report`. Exit code 1 from `report` means NO_GO (see `outputs/public/NO_GO.md`).
4. Review `outputs/public/coverage_summary.md` and decide go/no-go. Stop here; no orders ($30/frame)
   without explicit approval.
5. Check the **field mapping table** (`SCENE_FIELDS`, `DATASET_FIELDS`, `DECLASS_PATTERN` in
   `src/sbf/catalog/search.py`) against the first raw responses in `data/raw/catalog/`.

## Decisions
- Fixtures in `tests/fixtures/catalog/` are synthetic (`TEST_FIXTURE_*`, `test_declass_*`) and modelled
  on the documented M2M envelope; the real M2M field names are unverified.
- `login-token` responses are never saved (they contain the session key); all other responses are
  saved verbatim to `data/raw/catalog/<endpoint>_<hash>.json`, requests in `manifest.jsonl`.
  Failed calls (non-null `errorCode`) are also saved and listed in `data/interim/catalog/failures.json`;
  `sbf catalog search` exits 1 if any failure occurred.
- Resolution class uses the worst (upper) bound of the nominal range, so KH-4 (2-7.5 m) is `3-10m`;
  KH-4A/4B, KH-7, KH-9 pan, LANYARD are `<=3m`; KH-9 mapping `3-10m`; ARGON `>10m`. KH-9 without a
  mapping/panoramic discriminator is `unknown`. Table: `sbf/catalog/resolution.py`.
- Availability: `scanned` if the scene's `options.download` is true, `needs_order` if `options.order`,
  else `unknown`. This is only a guess; `sbf ingest declass availability` (ticket 04) replaces it with
  `download-options` results (column `availability_source` records which), keeping the guess as fallback.
- Go rule is **three-state** (`evaluate_go` -> `GoDecision.status`): cloud strictly `< 30`.
  - `GO`: criteria A (>=1 usable <=3m frame over GZ1-GZ4/TA1) and B (pre- and post-1967 frame over one
    feature) hold using only frames with known cloud < 30 (or a human "yes").
  - `NEEDS_REVIEW`: holds only if unknown-cloud frames (common in Declass) are counted. The report lists
    exactly those entity IDs and writes `outputs/public/browse_review.csv` (entity_id, dataset, date,
    resolution_class, features, browse_url, blank `usable`). Run `sbf ingest declass browse`, look at the
    images, fill `usable` yes/no, re-run `sbf catalog report`: answers are applied (yes = usable, no =
    excluded; they never override a known cloud value; earlier answers are kept when the sheet is rewritten).
  - `NO_GO`: fails even counting unknown-cloud frames. Writes `NO_GO.md`.
  - Exit codes of `sbf catalog report`: GO 0, NO_GO 1, NEEDS_REVIEW 3, bad input 2. The old
    `unknown_cloud_ok` flag is removed.
  Pre-1967 = year < 1967, post-1967 = year > 1967; 1967 frames count as neither.
- Footprint/feature intersection tests the reference *point* against the footprint polygon.
  BASE has no geometry and is skipped.
- Browse images and downloads: see ticket 04 / notes/PHASE_2_NOTES.md. Not done: Landsat/S2 counts (the
  `SceneCounter` Protocol is in `sbf.catalog.counts`; no implementation wired, so the report shows
  `not_run`), SPOT recorded as `unverified`.
