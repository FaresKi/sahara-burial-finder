# Phase 5 notes: document extraction and review (ticket 14)

## Documents the human must place in `data/raw/docs/` (SPEC section 4)
PDFs only; any filename (doc_id = slug of filename + first 8 hex of sha256).
- ICAN / Boell report, English and French versions
- IAEA Pub1215 (2005)
- Kabini 2025
- Makhijani / IEER 2022
- Optional: CIA CREST documents, NSA Archive EBB 184
Scanned PDFs without a text layer yield no chunks (no OCR); note which ones.

## Run
```
sbf docs ingest                    # manifest.json + chunks.jsonl in data/interim/docs/
sbf docs extract --dry-run         # chunk count, token and cost estimate; no API call
sbf docs extract [--limit N]       # HUMAN GATE: spends money, needs ANTHROPIC_API_KEY in env
sbf docs export-review             # data/interim/docs/review.csv
sbf docs precision                 # outputs/public/docs_extraction_precision.md
```
- Model: `claude-sonnet-5-5`, override with `SBF_EXTRACT_MODEL`. Prompt version `v1`
  (`sbf.docs_mining.extract.PROMPT_VERSION`); bump it when the prompt changes.
- Dry-run prices are placeholders (override `SBF_PRICE_IN` / `SBF_PRICE_OUT`, USD per
  million tokens); check current pricing.
- Responses are cached in `data/interim/docs/cache/` by (chunk hash, model, prompt
  version); re-runs are free. Chunks never cross page boundaries (claims spanning a page
  break are missed).
- Claims whose quote is not a verbatim substring of the chunk (whitespace-normalised) are
  rejected; these and extractor failures go to `data/interim/docs/failures.jsonl`.
- Quotes (<= 25 words) are internal only: they live in `data/interim`, never in `outputs/public`.

## Review step (Fares)
Open `data/interim/docs/review.csv`. For each row fill `accept` with `y` or `n` (blank means
unreviewed) and optionally `reviewer_note`. Check the quote supports the structured fields
(object, anchor, distance, bearing). Review at least 30 records, ideally a random sample.
Re-running `export-review` keeps existing decisions (matched by `extraction_id`).
`sbf docs precision` reports accepted / reviewed, or "insufficient" below 30 reviewed.

## For ticket 15
`describes_reference_feature` is true when the anchor names a GZ (Gerboise, ground zero,
GZn), Pollen, E2 or Beryl; exclude those records from prior evaluation (leakage).
Use only rows with `accept` = y.

# Spatial prior and evaluation (ticket 15)

Code built and tested on synthetic data; the real run is pending reviewed extractions
(`accept` = y in review.csv) and Phase 4 score tables.

```
sbf docs anchors                      # resolution status of every accepted claim (no coordinates)
sbf docs prior --aoi REG [--min-accepted N]   # refuses with no accepted rows (human gate)
sbf docs prior-eval --aoi REG         # writes outputs/public/prior_eval.md
```
- Gazetteer `config/anchors.geojson`: GZ1-GZ4, TA1, PO1 (public reference points; geometry is
  taken from the refined reference features when available), BASE (unresolved,
  `needs_human_location`), and Reggane town (approximate point from the existing draft exclusion
  in `sbf.detect.exclusions`, not digitised). Add aliases there as new anchor spellings appear;
  never add guessed coordinates. Unresolved anchors are listed by `sbf docs anchors` and skipped.
- Prior constants live at the top of `sbf/docs_mining/prior.py`: weights official 1.0 / testimony
  0.7 / secondary 0.4; distance sigma default 50 % of distance; bearing sigma default 45 deg;
  isotropic sigma 2 km (also used for bearing without distance); floor EPS = 0.01; prior in
  [EPS, 1]. Outputs: `data/processed/prior/prior_{AOI}.tif` (COG, provenance lists extraction
  ids) and score tables (detector `prior`, layer `docs_prior`, tile score = mean prior).
- Evaluation combines `score_norm x prior_norm` and `score_norm x (0.5 + 0.5 x prior_norm)`
  and reports delta recall@K / delta AP via the harness. Leakage control: the evaluation prior
  drops claims with `describes_reference_feature`, and any claim whose own unit-mass surface
  puts >= 2 % of its mass on any reference-feature footprint (`LEAK_MASS_THRESHOLD`). Excluded
  IDs and reasons are listed in the report (IDs only).
- Caveats: mass of a claim beyond the AOI grid is renormalised away; anchor position
  uncertainty is ignored; the 2 % threshold is a judgement call to revisit with real claims.
