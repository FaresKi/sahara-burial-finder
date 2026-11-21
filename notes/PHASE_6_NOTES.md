# Phase 6 notes (fusion, ticket 16)

- Code: `src/sbf/fuse/{families,candidates,pipeline,cli}.py`. Built and tested on synthetic data only; the real run is pending Phase 4/5 outputs and human sign-off (`--confirm-restricted`).
- Families: hist_change, hist_anomaly, modern_optical, SAR_L, SAR_C (S1, default weight 0 unless validation gives it one; SPEC lists only L-band), DEM, prior. Per-family tile score = max (or mean) `score_norm` over its tables.
- Weights: `max(0, AP - random_AP)` of the best table per family, normalised; manual `--weights` yaml overrides and is recorded in `fusion_manifest_{AOI}.json` (restricted) and `outputs/public/fusion_weights.md`.
- Fusion renormalises over present weighted families; `n_families` recorded. One tile size per run.
- Date bracket: historical single-date layers only (hist_anomaly family, matched to the registry by `<camera>_<entity_id>` suffix). Modern layers need dates via `--modern-dates` yaml; undated modern layers are ignored, so `post_2025_likely_cleanup` needs dated modern layers.
- Label precedence: pre_1960 (anomalous in the earliest frame, dated < 1960-02) > ops_1961_1968 (first_seen in window, or bracket overlaps it) > post_2025 > other.
- Open: review sheets (`candidate_sheets/*.html`, top-20) and the low/medium/high confidence column from SPEC Phase 6 acceptance are NOT in ticket 16's scope and not yet built.
- Open: the harness evaluates `random_ap` as prevalence; families with tiny prevalence may get noisy weights.

# Review sheets and restricted outputs (ticket 17)

- Code: `src/sbf/report/{confidence,sheets,cli}.py`, templates in `src/sbf/report/templates/`. Synthetic-data tests only; real run pending human sign-off (`--confirm-restricted`).
- Confidence thresholds live as constants in `confidence.py` (high/medium rules, single-date cap -> medium, post_2025/pre_1960 labels -> low, blind-test fail -> low). Default CLI blind-test state is "failed" (conservative); pass `--blind-test-passed` explicitly.
- Supporting frame RMSE = best checkpoint RMSE among included registry frames whose raster covers the candidate centroid. `n_hist_dates` uses a `n_present` column if present in the candidates table, otherwise the number of distinct covering frame dates (the gpkg does not carry n_present; consider adding it in sbf.fuse so the single-date cap is exact).
- Per-family score table and prior contribution need a `family_table` (index cand_id, columns families) passed to `write_sheets`; the CLI does not have one yet (FusionResult/gpkg lack it), so sheets list contributing families without numbers and prior shows n/a. Follow-up: persist per-candidate family scores in sbf.fuse.
- `sbf report sheets` writes restricted sheets, `index.html` and `candidate_confidence_{AOI}.csv` (all candidates); `sbf report public-stats` aggregates those CSVs into `outputs/public/candidates_public_stats.md` (AOI x confidence x label x score band counts only).
- Modern chips use stack layers s2_median, palsar_hv and dem (relief residual is not a stack layer).

# Fusion <-> sheets integration (follow-up to tickets 16/17)

- Supersedes the two "gpkg lacks n_present / per-family scores" notes above. `sbf.fuse` now carries per candidate `fam_<family>` (mean family score over the candidate's tiles, all families incl. prior, NaN when absent), `contrib_<family>` (w_f*mean_f / sum over present weighted families; 0 for a present zero-weight family, NaN when absent), `n_present` and `n_frames` (from `date_bracket`). They are in the candidates table, `candidates_{AOI}.gpkg` and `ranking.csv` (restricted only); existing columns unchanged.
- Sheets show a full per-family table (mean and contribution) and the prior contribution as a number; the single-date cap uses `n_present` exactly. Older gpkgs without these columns fall back to covering-frame dates and a family list without numbers, with a logged warning (re-run `sbf fuse run`).
- `sbf report sheets` gate: explicit `--blind-test-passed/--blind-test-failed` wins; else `outputs/public/blind_test_gate.json` (`status` PASS -> passed; FAIL/NOT_RUN/unreadable -> failed); else assumed failed (confidence capped low). The source used is printed. Logic: `confidence.resolve_blind_test`.
