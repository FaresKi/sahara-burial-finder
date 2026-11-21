# Sahara Burial Finder

An experiment, not a public map. It tests whether declassified 1960s-80s satellite imagery
(CORONA, KH-7, KH-9), modern open remote sensing (Sentinel-1/2, PALSAR, Copernicus DEM) and
AI can flag surface or near-surface disturbances consistent with waste burial at the former
French nuclear test sites in Algeria (Reggane/Hamoudia, In Ekker/Tan Afella, Adrar Tikertine).
The deliverable is evidence that the method works or does not. See `SPEC.md` (what) and
`ARCHITECTURE.md` (shared contracts).

> **Status:** experimental research code. Phase 1 (coverage catalogue) has run on live data;
> later phases are implemented and tested on synthetic data only. No candidate sites are
> published by this repository, by design.

## Responsible use

- Newly detected candidate areas are never published: they are written only to the git-ignored
  `outputs/restricted/`. Report anything sensitive privately (see [SECURITY.md](SECURITY.md)),
  never in issues or pull requests.
- Results are "candidate areas for verification" for authorised Algerian teams and competent
  authorities, not findings. Nothing here supports visiting the sites.
- This repository contains code only. Imagery and data come from USGS EROS, Copernicus (via
  Google Earth Engine) and JAXA under their own terms; see [NOTICE](NOTICE).

## Hard rules (SPEC section 0)

1. No public disclosure of newly detected candidate locations. They live only in
   `outputs/restricted/` (git-ignored), written through `sbf.core.restricted`.
2. Never fabricate identifiers, dates, dataset names or accuracy numbers. Record failures.
3. Ask the human before spending money or using credentials. Credentials come from env vars only.
4. Desk analysis only. No fieldwork instructions: the sites are restricted and contaminated.
5. Honest output: every result carries a confidence level and validation score; say "candidate
   area for verification", never "waste found".
6. Every raster has a `*.prov.json` sidecar (source, ID, acquisition date, download date,
   steps, script commit).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned; no system GDAL is needed
(rasterio/pyogrio wheels bundle it).

```
uv sync                  # core + dev
uv sync --all-extras     # plus ml, sam, gee, llm
cp .env.example .env     # fill in USGS_*, GEE_PROJECT, ANTHROPIC_API_KEY, RESTRICTED_KEY (optional)
```

Extras: `ml` (torch, transformers, timm), `sam` (sam2), `gee` (earthengine-api, geemap),
`llm` (anthropic). Heavy imports are lazy; tests needing them use `@pytest.mark.heavy`.

## Run

```
uv run sbf --help
uv run sbf core status
uv run pytest
uv run ruff check src tests
make all                 # = uv run sbf run-all : offline, public-only rebuild from data/raw
```

### `sbf run-all` (ticket 18)

Rebuilds every public output from `data/raw/` with no network and no credentials. Stages, in
order: `catalog-replay` (saved M2M JSON -> `coverage.csv`) -> `catalog-report` (**go/no-go gate**) ->
`modern-stack` -> `georef-batch` -> `detect-texture` -> `detect-embedding` (skipped without torch
and cached DINOv2 weights) -> `detect-sar-dem` -> `detect-change` -> `blind-test` -> `docs-precision`
-> `docs-prior` -> `docs-prior-eval` (these three only with human-accepted rows) -> `fuse` and
`candidate-sheets` (restricted, off by default) -> `writeup` -> `check-public` (leak guard, last).

```
uv run sbf run-all --dry-run          # print the plan, run nothing
uv run sbf run-all --from georef-batch
uv run sbf run-all --only writeup
uv run sbf run-all --force            # ignore the up-to-date check
uv run sbf run-all --same-era         # change detection also pairs same-era frames
uv run sbf run-all --include-restricted --confirm-restricted   # restricted fusion + briefing
uv run sbf run-all --continue-past-gates   # debugging only; stamps "NOT VALID: gates bypassed"
```

A stage whose inputs are missing is reported as `skipped (missing input: ...)` and the report says
so; nothing is fabricated. Up-to-date checks use input size and mtime, recorded per stage in
`data/interim/run_manifest.json` (code changes do not invalidate; use `--force`). Exit codes: 0 ok,
1 stage failed, 2 usage, 3 gate stop NEEDS_REVIEW, 4 gate stop NO_GO, 6 public leak found.
A blind-test FAIL is a recorded negative result, not a pipeline error: the report is still written
and fusion is skipped.

Other commands added by ticket 18: `sbf report writeup` (writes `outputs/public/report.md`; with
`--include-restricted --confirm-restricted` also `outputs/restricted/briefing.md`) and
`sbf report check-public` (fails if `outputs/public` holds coordinates that are not public
references).

### Human-run command sequence (needs credentials, network or judgement)

1. `sbf catalog search --aoi REG --aoi INE`, then `sbf catalog report` (**gate: go/no-go**).
2. `sbf ingest declass browse|availability|download` (**gate: any purchase** via `order-list`).
3. `sbf ingest modern export --aoi REG|INE` (GEE), then `sbf ingest refine ...` (reference refinement).
4. `sbf detect draft-exclusions`, edit and review `config/exclusions.geojson` (**gate: exclusions**).
5. `sbf docs ingest`, `sbf docs extract` (**gate: API spend**), `sbf docs export-review`, review
   (**gate: extraction review**, at least 30 records).
6. `sbf run-all` (the offline rebuild; stops at the gates above when unmet).
7. `sbf run-all --include-restricted --confirm-restricted` only after human sign-off (**gate:
   restricted outputs**).

Details and the ordered checklist: `notes/PHASE_7_NOTES.md`.

### Environment variables

`USGS_USERNAME`, `USGS_M2M_TOKEN` (catalogue and downloads), `GEE_PROJECT` (modern export),
`ANTHROPIC_API_KEY` (document extraction), `RESTRICTED_KEY` (age recipient for restricted outputs),
optional `SBF_ROOT` (project root), `SBF_EXTRACT_MODEL`, `SBF_SAM2_CHECKPOINT`, `HF_HOME` (DINOv2
cache). `run-all` itself needs none.

Sub-apps: `catalog`, `ingest`, `georef`, `features`, `detect`, `docs`, `fuse`, `report`, `core`.

## Layout

`config/` AOIs, reference features, settings. `src/sbf/<phase>/` code. `data/` (git-ignored)
raw/interim/processed. `outputs/public/` known-feature results only. `outputs/restricted/`
(git-ignored) candidates. `notes/` per-phase notes. Tests never touch the network.

## Contributing, license, citation

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under the [Apache License 2.0](LICENSE).
If you use this work, please cite it via [CITATION.cff](CITATION.cff).
