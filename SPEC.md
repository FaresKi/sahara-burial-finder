# SPEC — Sahara Burial Finder (experimental, solo)

> Build a reproducible pipeline that tests whether declassified 1960s–80s satellite imagery, modern open remote-sensing data and AI can locate surface/near-surface waste buried at France's former nuclear test sites in Algeria (Reggane/Hamoudia, In Ekker/Tan Afella, Adrar Tikertine).
>
> Owner: Fares (human). Implementers: Claude agents.
> Status: experiment. **The deliverable is evidence that the method works (or doesn't), not a public map.**

---

## 0. Hard rules for agents (read first, never violate)

1. **No public disclosure of candidate locations.** Never commit or print to logs/issues/PRs the coordinates of *newly detected* candidate burial sites. Candidates go only into `outputs/restricted/` (git-ignored, encrypted at rest if possible). Already-public reference coordinates (Section 3) are fine.
2. **Never fabricate data identifiers.** No invented entity IDs, frame dates, dataset names or accuracy numbers. If an API call fails, record the failure; don't fill gaps.
3. **Ask the human before spending money or using credentials** (USGS scan orders at $30/frame, commercial imagery, cloud compute). Credentials come from env vars only, never hard-coded or committed.
4. **No fieldwork instructions.** The project produces desk analysis only. Do not plan or suggest site visits; the sites are restricted military zones and contaminated.
5. **Honest output.** Every result carries a confidence level and its validation score. Phrase findings as "candidate area for verification", never as "waste found".
6. **Log provenance.** Every raster in the pipeline has a sidecar `*.prov.json`: source, ID, acquisition date, download date, processing steps, script commit hash.

---

## 1. Background (condensed)

- France conducted 17 tests in Algeria in 1960–66: 4 atmospheric at Hamoudia (≈50 km S of Reggane) and 13 underground in tunnels under Taourirt Tan Afella (In Ekker), plus plutonium dispersion trials (Pollen at Adrar Tikertine; Augias at the Gerboise Rouge site).
- Contaminated equipment (aircraft, vehicles, engines, drums, cable, cemented vessels, lava/slag) was buried in bulldozed trenches and pits, mostly during operations and the 1966–67 dismantling. Sources: ICAN France / Observatoire des armements, *Radioactivity Under the Sand* (2020); IAEA STI/PUB/1215 (2005); Kabini, Princeton SGS preprint (2025).
- Algeria says France never handed over the burial maps; France says it handed a file in Aug 2022. Algeria began cleanups at In Ekker in 2026 (second operation at Béryl E2, 1 Nov 2026) without public maps.
- Hypothesis: (a) US reconnaissance film (CORONA, KH-7, KH-9) captured the excavations; (b) hyperarid desert surfaces preserve disturbance for decades; (c) unsupervised/few-shot ML can flag these disturbances; (d) document mining can turn testimony into a spatial prior.

**Out of scope:** the underground tunnel interiors (invisible to any remote sensor), radiation dose assessment, attribution/legal claims.

---

## 2. Research questions the experiment must answer

| ID | Question | Answered by |
|---|---|---|
| Q1 | Does declassified imagery cover the AOIs, at which dates and resolutions? | Phase 1 |
| Q2 | Are known features (ground zeros, base, lava apron, Pollen pad) visible in historical and modern imagery? | Phase 2–3 |
| Q3 | Can the historical frames be georeferenced to < 20 m RMSE (target < 10 m)? | Phase 3 |
| Q4 | Does unsupervised anomaly/change detection recover the known features in a blind test? (recall@K) | Phase 4 |
| Q5 | Can LLM document mining produce a useful spatial prior (does it raise the rank of known features)? | Phase 5 |
| Q6 | What candidate areas emerge, ranked, with confidence? (restricted) | Phase 6 |

A clear **negative** answer to Q1 or Q4 is a valid, publishable result.

---

## 3. Areas of interest and reference features

Coordinates of ground zeros are from IAEA 2005 (public). Approximate values are marked; agents must flag them as such and refine from imagery where possible.

### 3.1 AOIs (WGS84, EPSG:4326)

| AOI | Bounding box (lon_min, lat_min, lon_max, lat_max) | Notes |
|---|---|---|
| `REG` Reggane / Hamoudia | (-0.25, 26.05, 0.25, 26.75) | Covers ground zeros, Hamoudia base, plateau, Reggane oasis. Generous on purpose (testimony mentions sites 5–10 km from GZ / NE of plateau). |
| `INE` In Ekker / Tan Afella / Adrar Tikertine | (4.60, 23.80, 5.20, 24.20) | Tunnel portals ≈ 24.00–24.07 N, 5.03–5.06 E (approx.); Pollen ≈ 23.92 N, 4.72 E (approx., read from IAEA figure). |

Store as `config/aois.geojson`.

### 3.2 Reference (validation) features — `config/reference_features.geojson`

| ID | Feature | Lat | Lon | Precision | Expected signature |
|---|---|---|---|---|---|
| GZ1 | Gerboise Bleue ground zero (13 Feb 1960) | 26.3117 | -0.0572 | IAEA, arc-second | Disturbed/fused-sand area ≈ 1 km² |
| GZ2 | Gerboise Blanche ground zero (1 Apr 1960; surface shot, later filled) | 26.1661 | -0.1025 | IAEA | ≈ 1 km² |
| GZ3 | Gerboise Rouge / Augias (27 Dec 1960) | 26.3536 | -0.1236 | IAEA | ≈ 0.2 km² |
| GZ4 | Gerboise Verte (25 Apr 1961) | 26.3217 | -0.0733 | IAEA | ≈ 0.1 km²; equipment left ~2 yrs before burial |
| TA1 | Tan Afella tunnel portal zone (E2 Béryl etc.) | ~24.03 | ~5.05 | approximate | Portals, lava/slag apron, roads |
| PO1 | Pollen site, Adrar Tikertine (~3 ha tarmacked) | ~23.92 | ~4.72 | approximate | Tarmac pad, low visual cues today |
| BASE | Hamoudia base / Reggane plateau installations | to locate | to locate | to derive from imagery | Buildings, runway, roads (non-target, used for georeferencing + negative class) |

Agents must **refine approximate features from imagery** and record the refined geometry plus the evidence (frame ID, screenshot path).

**Blind-test protocol:** the detection models in Phase 4 must not see these labels during fitting. Labels are used only for scoring.

---

## 4. Data sources

| Layer | Source | Access | Notes |
|---|---|---|---|
| CORONA KH-4/4A/4B, ARGON, LANYARD (1960–72) | USGS EarthExplorer, "Declassified Data → Declass 1" | Free account; M2M API needs an access request | Free if already scanned, else $30/frame (**ask human**) |
| KH-7 GAMBIT (1963–67, 0.6–1.2 m), KH-9 mapping camera (1973–80, 6–9 m) | EarthExplorer "Declass 2" | same | KH-7 is the highest-value layer if it exists over AOIs |
| KH-9 panoramic (1971–84, 0.6–1.2 m) | EarthExplorer "Declass 3" | same | Footprints approximate; many unscanned |
| Sentinel-2 L2A | GEE `COPERNICUS/S2_SR_HARMONIZED` | GEE non-commercial account | 10 m, modern optical baseline |
| Sentinel-1 GRD | GEE `COPERNICUS/S1_GRD` | GEE | C-band backscatter, temporal stats |
| ALOS PALSAR L-band yearly mosaic | GEE `JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH` | GEE, JAXA attribution | 25 m; L-band penetrates dry sand ~1–2 m |
| Copernicus DEM GLO-30 | GEE `COPERNICUS/DEM/GLO30` | GEE | Modern terrain, bedrock GCP support |
| Landsat 1–9 | GEE / USGS | free | 1972→ change baseline |
| SPOT 1–5 (1986–2015) | CNES "SPOT World Heritage" (REGARDS) | free, Etalab licence | Coverage of AOIs unverified |
| VHR basemap for GCPs | Esri World Imagery / Google satellite tiles (viewing only, respect ToS) | free view | Do not redistribute tiles |
| Documents | ICAN/Böll report PDF (EN/FR), IAEA Pub1215 PDF, Kabini 2025 PDF, Makhijani/IEER 2022 PDF, CIA CREST reading room, NSA Archive EBB 184 | public | For Phase 5 |

Agents must **discover exact dataset names via the M2M `dataset-search` endpoint** rather than hard-code them, and must verify GEE IDs at runtime.

**Environment variables:** `USGS_USERNAME`, `USGS_M2M_TOKEN`, `GEE_PROJECT`, `ANTHROPIC_API_KEY` (Phase 5), optional `RESTRICTED_KEY` (age/GPG key for restricted outputs).

---

## 5. Repository layout

```
sahara-burial-finder/
├── README.md                # what, why, how to run, hard rules
├── SPEC.md                  # this file
├── pyproject.toml           # uv or poetry
├── config/
│   ├── aois.geojson
│   ├── reference_features.geojson
│   └── settings.yaml        # paths, CRS (UTM 31N for REG = EPSG:32631, UTM 31N/32N for INE), tile sizes
├── src/sbf/
│   ├── catalog/             # Phase 1: EarthExplorer M2M search, coverage report
│   ├── ingest/              # downloads, GEE exports, provenance sidecars
│   ├── georef/              # Phase 3: GCP tooling, warp, RMSE report
│   ├── features/            # tiling, embeddings, texture metrics
│   ├── detect/              # Phase 4: anomaly + change detection, ranking
│   ├── docs_mining/         # Phase 5: LLM extraction → spatial prior
│   ├── fuse/                # Phase 6: combine scores, candidate polygons
│   └── report/              # notebooks → HTML/PDF reports
├── notebooks/               # exploration only; logic lives in src/
├── data/                    # git-ignored
│   ├── raw/  interim/  processed/
├── outputs/
│   ├── public/              # coverage tables, method metrics, figures of KNOWN features only
│   └── restricted/          # git-ignored: candidate polygons, scores, crops
└── tests/
```

`.gitignore` must include `data/`, `outputs/restricted/`, `.env`.

**Stack:** Python 3.11+, `requests`, `geopandas`, `shapely`, `rasterio`, `gdal`, `pyproj`, `numpy`, `opencv-python`, `scikit-image`, `scikit-learn`, `torch` (MPS on Apple Silicon), `transformers` / `timm` (DINOv2), `segment-anything` or `sam2`, `earthengine-api`, `geemap`, `pymupdf`, `anthropic`, `pydantic`, `typer` (CLI), `pytest`. Optional: NASA Ames Stereo Pipeline, HEXIMAP, QGIS (manual GCPs).

Target hardware: a single Apple Silicon laptop. No GPU cluster assumed.

---

## 6. Phases, tasks and acceptance criteria

### Phase 1 — Coverage catalogue (answers Q1) · go/no-go

Tasks
- `sbf catalog search --aoi REG --aoi INE` : query every Declass 1/2/3 dataset via M2M with the AOI polygons, 1959-01-01 → 1984-12-31.
- Collect per scene: entity ID, dataset, mission, camera, acquisition date, cloud cover, download availability (scanned / needs order), browse image URL, footprint polygon.
- Intersect footprints with each reference feature; download browse images.
- Same for Landsat (1972–), SPOT World Heritage (if API accessible), Sentinel-2 (count only).

Outputs
- `outputs/public/coverage.csv`, `coverage_map.html` (footprints + reference features), `coverage_summary.md`.

Acceptance
- Every row traceable to an API response saved in `data/raw/catalog/*.json`.
- Summary states, per reference feature: best historical frame (date, resolution class), whether free or $30, and whether it predates/postdates the 1966–67 dismantling.
- **Go** if ≥ 1 usable (cloud < 30%) frame ≤ 3 m resolution class exists over GZ1–GZ4 *or* TA1, and at least one pre-1967 and one post-1967 frame exist for some reference feature. Otherwise write `NO_GO.md` explaining what's missing and stop for human review.

### Phase 2 — Data acquisition and modern baseline (Q2, part)

Tasks
- Download all free scanned frames that intersect reference features (highest resolution scan available). Produce an **order list** for unscanned frames, ranked by value, with total cost → stop for human approval.
- GEE exports clipped to AOIs: Sentinel-2 median composite (cloud-masked, last 2 years) + per-season medians; Sentinel-1 VV/VH temporal mean and std; PALSAR HH/HV latest + multi-year; Copernicus DEM + slope/roughness.
- Write provenance sidecars.

Acceptance
- `data/processed/modern/{REG,INE}/` contains co-registered COGs in UTM, one stack per AOI.
- A notebook shows each reference feature in each modern layer, with a short visual-detectability note (visible / ambiguous / not visible).

### Phase 3 — Georeferencing historical frames (Q3)

Tasks
- Implement a semi-automatic GCP workflow:
  1. Coarse placement from the scene footprint corners.
  2. Feature matching (SIFT/ORB or LoFTR) between frame and modern Sentinel-2/VHR, **masked to non-dune terrain** (use DEM roughness + S2 texture to exclude dune fields; prefer bedrock, regs, wadis, roads).
  3. RANSAC; polynomial (order 2–3) or thin-plate-spline warp; panoramic frames may need piecewise warping per strip.
  4. Fallback: export a QGIS project for the human to place GCPs manually; read them back.
- Optional: ASP / HEXIMAP pipeline for KH-9 mapping-camera stereo DEMs.
- Hold out 20% of GCPs for checkpoint RMSE.

Acceptance
- Per frame: `georef_report.json` with GCP count, checkpoint RMSE (m), method, warning flags.
- Target RMSE < 10 m for ≤ 3 m frames; frames > 20 m RMSE are flagged and excluded from change detection.
- Visual overlay check of GZ1–GZ4 positions on each georeferenced frame; offsets logged.

### Phase 4 — Unsupervised detection + blind test (Q4)

Tile every georeferenced frame and modern stack (e.g., 64×64 m and 128×128 m tiles, 50% overlap).

Detectors (implement all, compare):
1. **Embedding anomaly**: DINOv2 (or a remote-sensing foundation model if available locally: Clay, Prithvi, SatMAE) embeddings per tile → per-AOI background model (Isolation Forest, kNN distance, or Gaussian/Mahalanobis) → anomaly score.
2. **Texture/statistics baseline**: GLCM contrast/entropy, local variance, edge density → same anomaly models. (Cheap baseline; must be reported.)
3. **Change detection**: for co-registered date pairs (esp. pre-1967 vs post-1967), embedding cosine distance and image-difference metrics per tile → change score. Also historical-vs-modern.
4. **SAR/DEM anomalies (modern)**: PALSAR HV/HH and S1 backscatter z-scores vs local background; DEM micro-relief (high-pass residual) anomalies.
5. **Segmentation for review**: run SAM on top-K tiles to produce polygons for human review (not for scoring).

Masking: exclude known infrastructure and settlements (roads, buildings, Reggane oasis, recent cleanup works visible in 2026 imagery) via a `config/exclusions.geojson` the agent builds and the human reviews.

Blind-test scoring:
- Compute **recall@K** (K = 10, 50, 100 tiles) and **average precision** for the reference features (positive tiles = tiles intersecting the reference geometries; negatives = everything else, acknowledging label noise).
- Report per detector, per data layer, per AOI. Compare against a random baseline.

Acceptance
- `outputs/public/blind_test.md` with a metrics table, figures of the known features only, and an honest interpretation.
- Success criterion (experimental): at least one detector achieves recall@50 ≥ 0.5 on reference features in at least one historical layer and clearly beats random. If not, document it as a negative result and stop for human review.

### Phase 5 — LLM document mining → spatial prior (Q5)

Tasks
- Ingest PDFs (ICAN/Böll EN + FR, IAEA 2005, Kabini 2025, IEER 2022) and any CREST documents the human downloads into `data/raw/docs/`.
- Extract with the Claude API into a strict schema (pydantic, JSON output). One record per spatial claim:
  ```json
  {
    "doc_id": "...", "page": 31, "quote": "<= 25 words verbatim, for internal audit only>",
    "object_buried": "aircraft | vehicles | drums | cable | slag | vessels | unknown",
    "anchor": "Gerboise Bleue GZ | Reggane plateau | E2 portal | ...",
    "distance_km": 10, "distance_uncertainty_km": 3,
    "bearing": "NE", "bearing_uncertainty_deg": 45,
    "time_period": "1961-1967", "confidence_source": "testimony | official | secondary",
    "notes": "..."
  }
  ```
- Human-review step: export extractions to CSV for Fares to accept/reject before use.
- Convert accepted records into probability surfaces (annular sectors with Gaussian falloff around resolved anchors), sum with weights by source confidence → `prior_{AOI}.tif`.

Acceptance
- Extraction precision checked by the human on ≥ 30 records (report the rate).
- Evaluate whether multiplying the Phase-4 scores by the prior improves recall@K on reference features (note: possible leakage if a document describes a reference feature itself; exclude such records from the evaluation and say so).
- Verbatim quotes stay internal (not in public outputs).

### Phase 6 — Fusion and candidate ranking (Q6) · restricted

Tasks
- Fuse normalised scores: `S = w1·hist_change + w2·hist_anomaly + w3·modern_optical + w4·SAR_L + w5·DEM + w6·prior`, weights chosen from Phase-4/5 validation (no tuning on unseen data).
- Merge adjacent high-score tiles into polygons, compute area, mean score, contributing layers and dates of first appearance of disturbance.
- Classify each candidate's **disturbance date bracket** (appears between frame dates X and Y). Disturbances first appearing 1961–1968 rank highest; post-2025 disturbances are likely cleanup works and are down-weighted.
- Generate a review sheet per candidate (crops from each date, scores, prior contribution).

Outputs (**restricted only**)
- `outputs/restricted/candidates_{AOI}.gpkg`, `candidate_sheets/*.html`, `ranking.csv`.
- Public: only aggregate stats (count of candidates per score band, no coordinates).

Acceptance
- Top 20 candidates per AOI have complete review sheets.
- Each candidate states confidence (low/medium/high) with justification and "requires ground verification by competent authorities".

### Phase 7 — Write-up

- `outputs/public/report.md`: method, data coverage, georef accuracy, blind-test metrics, limits, negative results, recommendations for field verification partners (drone magnetometry at 2–3 m AGL, gamma screening, GPR; done by authorised Algerian teams).
- `outputs/restricted/briefing.md`: for the rehabilitation agency / ICAN France / Princeton SGS, with candidates.
- Clean, documented code; `make all` (or `sbf run-all`) reproduces public outputs from raw data.

---

## 7. Known pitfalls (agents should design for these)

- **Desert film saturation**: bright sand clips in scans; use local contrast normalisation (CLAHE) before matching/embeddings.
- **Dune migration**: dunes move tens of metres per decade; never use dunes for GCPs; mask them in change detection.
- **Panoramic distortion** (CORONA, KH-9 pan): strong across-track scale change; frame-wide polynomials will fail; work in strips.
- **Clutter**: Algerian army presence since 1966, scavenging for cable/rails, 2026 cleanup excavations → many true disturbances that are not burials. Dating disturbances is essential.
- **Label noise**: reference geometries are approximate; Chornobyl work showed archive-derived labels can drop accuracy from ~98% to ~65%. Treat metrics as indicative.
- **Invisible targets**: anything inside tunnels or buried deeper than ~1–2 m may have no surface or L-band signature. State this in every report.
- **Resolution floor**: 1960s CORONA (≈2–3 m) can show trenches and vehicle parks, not individual drums.

---

## 8. Agent working conventions

- Work phase by phase; open a short `PHASE_N_NOTES.md` with decisions, failures and open questions.
- Stop and ask the human at: Phase 1 go/no-go, any purchase, credential setup, the exclusions mask review, Phase 5 extraction review, before producing restricted outputs.
- Unit tests for geometry/CRS utilities, tiling and scoring; a tiny synthetic dataset (draw fake trenches on a real modern tile) as an integration test for detectors.
- Prefer small, verifiable increments; every figure in a report must be generated by committed code.
- No network calls outside the listed sources without asking.

---

## 9. Definition of done (experiment)

The experiment is complete when:
1. Q1–Q5 each have a documented answer (positive or negative) backed by artefacts in `outputs/public/`.
2. If Phase 4 passes, a restricted ranked candidate set exists with review sheets.
3. The public report can be reproduced from raw data with one command.
4. No newly detected candidate coordinates appear anywhere outside `outputs/restricted/`.

---

## 10. References

- IAEA (2005). *Radiological Conditions at the Former French Nuclear Test Sites in Algeria*, STI/PUB/1215.
- Collin, J.-M. & Bouveret, P. (2020). *Radioactivity Under the Sand*. ICAN France / Observatoire des armements / Heinrich Böll Foundation.
- Kabini, J. T. (2025). *Reassessing Radiological Contamination and Hazards in the French Nuclear Test Site in Algeria*. Princeton SGS preprint.
- Makhijani, A. (2022). *Summary of Health and Environmental Impacts of French Nuclear Testing in Algeria*. IEER for ICAN.
- Briechle, S. et al. (2020). Detection of radioactive waste sites in the Chornobyl exclusion zone using UAV-based lidar data and multispectral imagery. *ISPRS J. Photogramm. Remote Sens.* 167.
- Barthelme, P. et al. (2024). Detecting Vietnam War bomb craters in declassified historical KH-9 satellite imagery. *Science of Remote Sensing* 10.
- Pistola, A. et al. (2025). AI-ming backwards: automatic detection of sites on CORONA imagery. *PLOS ONE* 20(8).
- Dehecq, A. et al. (2020). Automated processing of declassified KH-9 Hexagon satellite images for global elevation change analysis. *Frontiers in Earth Science*.
- McCauley, J. F. et al. (1982). Subsurface valleys and geoarcheology of the eastern Sahara revealed by Shuttle radar. *Science* 218.
- Belnap, J. & Warren, S. (2002). Patton's tracks in the Mojave Desert. *Arid Land Research and Management*.
- USGS EROS: Declassified Satellite Imagery 1, 2, 3 (EarthExplorer).
