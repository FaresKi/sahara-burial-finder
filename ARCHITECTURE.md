# Architecture — Sahara Burial Finder

Companion to `SPEC.md`. SPEC says *what*; this file fixes the *shared contracts* every
module codes against, so phases can be built in parallel without collisions.
SPEC §0 hard rules override everything here.

## Decisions (ADR-lite)

| # | Decision | Why |
|---|---|---|
| D1 | Python 3.12 via `uv`, `requires-python = ">=3.11,<3.13"`. | Wheels for torch/rasterio/sam exist; system python is 3.15. |
| D2 | No system GDAL. Use `rasterio`/`pyogrio` wheels (bundled GDAL). The `gdal` Python package is **not** a dependency. | Not installed on target laptop; wheels avoid brew pain. |
| D3 | Both AOIs use **EPSG:32631** (UTM 31N). INE lon 4.6–5.2 E is inside zone 31 (0–6 E). | One metric CRS per AOI, simpler tiling. |
| D4 | One CLI `sbf` (typer). Each package owns a sub-app: `sbf catalog …`, `sbf ingest …`, `sbf georef …`, `sbf detect …`, `sbf docs …`, `sbf fuse …`, `sbf report …`, plus `sbf run-all`. | Packages register independently; no shared-file edits. |
| D5 | All external services sit behind a thin client class (`M2MClient`, `GeeExporter`, `DocExtractor`) injected into pure logic. Tests use recorded fixture JSON in `tests/fixtures/`; **tests never touch the network**. | Credentials gated by human (SPEC §0.3); keeps CI green offline. |
| D6 | Heavy optional deps (torch, transformers, sam2, earthengine-api, anthropic) live in extras and are imported lazily inside functions. Tests needing them are marked `@pytest.mark.heavy` / skipped when absent. | Fast core install and test runs. |
| D7 | Detectors never see reference labels. The scoring harness is the only module that loads `reference_features.geojson` for scoring. | SPEC blind-test protocol. |
| D8 | Anything derived from candidate scores with coordinates is written only through `sbf.core.restricted`. Logging of geometries/coordinates from detection, fusion and candidates is forbidden; log tile counts and IDs only. | SPEC §0.1. |

## Shared contracts (owned by `sbf.core`, ticket 01)

### Paths & config
- `sbf.core.config.load_settings() -> Settings` (pydantic) reading `config/settings.yaml`.
  Exposes `root`, `data_raw`, `data_interim`, `data_processed`, `outputs_public`,
  `outputs_restricted`, per-AOI `crs` (`"EPSG:32631"`), tile sizes `[64, 128]`, overlap `0.5`.
- `sbf.core.aoi.load_aois() -> GeoDataFrame` (cols: `aoi_id`, `geometry`, EPSG:4326).
- `sbf.core.aoi.load_reference_features() -> GeoDataFrame`
  (cols: `feature_id`, `name`, `aoi_id`, `precision` ∈ {`iaea`,`approximate`,`to_locate`}, `geometry`).
  Point features are buffered to their expected-area footprint by the harness, not stored as polygons.

### Provenance (SPEC §0.6)
`sbf.core.provenance.write_prov(raster_path, *, source, source_id, acquisition_date, steps, inputs=None)`
writes `<raster>.prov.json`:
```json
{"source": "...", "source_id": "...", "acquisition_date": "YYYY-MM-DD|null",
 "download_date": "ISO8601", "processing_steps": ["..."], "inputs": ["path", "..."],
 "script_commit": "<git sha | 'uncommitted' | 'no-git'>", "sbf_version": "x.y.z"}
```
`read_prov(path)` and `append_step(path, step)` complete the API.

### Rasters
`sbf.core.raster.write_cog(path, array, profile, *, prov=...)` — always COG, always a sidecar.

### Tile grid (ticket 09)
`sbf.features.tiling.make_grid(aoi_id, size_m, overlap) -> GeoDataFrame`
cols: `tile_id` (str, `"{aoi}-{size}-{col:05d}-{row:05d}"`), `aoi_id`, `size_m`, `geometry` (EPSG:32631).
Deterministic: same inputs ⇒ same IDs. Grid is aligned to a fixed origin per AOI.

### Score table (the integration contract between detect, scoring, prior, fusion)
Parquet at `data/processed/scores/{aoi}/{detector}__{layer}__{size_m}m.parquet` (one file per tile size, so 64 m and 128 m runs never overwrite each other):

| column | type | meaning |
|---|---|---|
| `tile_id` | str | from tile grid |
| `aoi_id` | str | `REG`/`INE` |
| `size_m` | int | 64/128 |
| `detector` | str | `texture`, `embed_iforest`, `embed_knn`, `embed_maha`, `change_embed`, `change_diff`, `sar_z`, `dem_relief`, `random`, `prior` (Phase 5 spatial prior, `layer = "docs_prior"`) |
| `layer` | str | e.g. `s2_median`, `palsar_hv`, `kh7_<entity>`, `pair_<a>_<b>` |
| `score` | float | higher = more anomalous/changed; raw |
| `score_norm` | float | per (aoi, detector, layer) rank-normalised to [0,1] |

`sbf.detect.scores.write_scores(df)` / `read_scores(aoi, detector=None, layer=None)` validate the schema.
Score tables contain tile IDs, not coordinates; they are fine under `data/` (git-ignored) but must never be
printed with geometry.

### Restricted outputs
`sbf.core.restricted.restricted_path(*parts) -> Path` (under `outputs/restricted/`, creates dirs,
refuses paths that resolve elsewhere). `encrypt_if_key(path)` uses `age` with `RESTRICTED_KEY` if set,
else warns. A test asserts no file under `outputs/public/` contains coordinates of non-reference geometry.

## Layout
As SPEC §5. Every package has `cli.py` exposing `app = typer.Typer()`; `sbf.cli` mounts them.
Phase notes go in `notes/PHASE_N_NOTES.md`.

## Human gates (agents stop and report; never bypass)
Phase 1 go/no-go · any purchase · credential setup · exclusions mask review ·
Phase 5 extraction review · before producing restricted outputs from real data.

## Implementation notes (ticket 01)
- `Settings` also has `config_dir`, `tile_sizes_m`, `overlap`, `modern_resolution_m`, `aois[aoi].crs` and `Settings.crs(aoi_id)`.
  `load_settings(root=None)`: root defaults to `$SBF_ROOT` else the repo root. Loaders/guards take an optional `settings=`.
- `load_aois()` has an extra `name` column; `load_reference_features()` also returns `expected_area_km2`, `notes`.
- `sbf.core.crs`: `transform_geometry`, `aoi_crs`, `lonlat_to_xy`, `xy_to_lonlat`, `buffer_point_m`.
- `write_cog(path, array, profile, *, prov: dict)`: `prov` is the kwargs of `write_prov`; `profile` needs `crs`, `transform`.
- `encrypt_if_key(path) -> Path` (returns `<path>.age` on success, else `path`). `script_commit()` is also 'uncommitted' when src/config are dirty.
- Extra sub-app `sbf core status`. Test fixture: `tmp_project` in `tests/conftest.py` (sets `SBF_ROOT`).

## Architect notes (post wave 3)
- Tile origin (multiple of 128 m) and the 10 m modern grid are not pixel-aligned; `tile_windows` rounds to the nearest pixel (≤ 5 m error). Accepted: overlap steps (32/64 m) cannot be multiples of 10 m anyway.
- `score_norm` is rank-normalised per (aoi, detector, layer, size_m).
- Recall@K is feature-level (a reference feature is hit if any of its tiles is in the top K); tile-level recall and AP are reported alongside.
- Phase 1 go rule is three-state: GO / NEEDS_REVIEW (unknown-cloud frames need a human browse check) / NO_GO.
- Era convention everywhere: `pre_1967` (< 1967), `dismantling_1967` (= 1967), `post_1967` (1968–1984).
- Blocking edges in the tickets gate *real-data runs*; code for downstream tickets may be built earlier against these contracts with synthetic data.
