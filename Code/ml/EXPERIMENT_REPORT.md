# LEAD optimization report — frozen full-year building-disjoint split (160 train / 40 val)

Split: `GroupShuffleSplit(test_size=0.2, random_state=42)` on `building_id`, full-year filtered data
(1,641,816 rows with `meter>0`; train 1,307,573 / val 334,243). All rows below use these buildings.
Rolling features are past-only per building (`shift(1)`, no label use, no cross-building leakage).
Threshold for P/R/F1 is 98th pct of val scores unless stated (Exp4). ROC/PR are threshold-independent.

| Experiment | Features | Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Fit |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | `log_meter,hour_x,hour_y` (3) | IF 100/auto/1.0 | 0.751 | 0.123 | 0.147 | 0.137 | 0.142 | 4.3s |
| Exp1 time | +`weekday_x,y,month_x,y,is_holiday` (8) | same | 0.745 | 0.108 | 0.062 | 0.052 | 0.057 | 3.4s |
| Exp2a | +`robust_z_24,z_168` (5) | same | 0.779 | 0.380 | 0.517 | 0.445 | 0.478 | 3.7s |
| Exp2b | +`dLag24` (6) | same | 0.773 | 0.369 | 0.503 | 0.434 | 0.466 | 3.6s |
| Exp2c | +`robust_z_168` only (4) | same | 0.785 | 0.411 | 0.515 | 0.441 | 0.475 | 3.4s |
| Exp3 grid best | Exp2c feats (4) | IF 200/0.5/0.5 | 0.786 | 0.286 | 0.360 | 0.307 | 0.332 | 34s |
| Exp3+confirm best-PR | Exp2c feats (4) | IF 200/256/1.0 | 0.784 | 0.431 | 0.519 | 0.443 | 0.478 | 6.8s |

Grid notes: `max_samples="sqrt"` is **invalid** for `IsolationForest` (must be `"auto"`, int, or float in (0,1]) — rejected with `InvalidParameterError`. All `max_features=1.0` cells with large `max_samples` collapsed (PR ~0.08); `max_features=0.5` rescued them. Default `max_samples="auto"` (=256 rows/tree) beats every large-sample grid cell on PR at ~5–10x less fit time.

Threshold sweep (Exp2c + IF 100/auto/1.0, ROC 0.785 / PR 0.411 fixed):

| pct | thr | flags | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 90 | 0.567 | 33,011 | 0.136 | 0.580 | 0.220 |
| 95 | 0.584 | 16,505 | 0.238 | 0.508 | 0.324 |
| 97 | 0.597 | 9,903 | 0.364 | 0.465 | 0.408 |
| 98 | 0.608 | 6,640 | 0.515 | 0.441 | 0.475 |
| 99 | 0.643 | 3,317 | 0.852 | 0.365 | 0.511 |
| 99.5 | 0.657 | 1,745 | 0.823 | 0.186 | 0.303 |

## Answers

1. **Best-performing:** Exp2c feats + `IF(200, max_samples=256, max_features=1.0)` — ROC 0.784, PR 0.431 (highest PR), F1 ~0.48 at p98.
2. **Best transferable for IITD:** same 4 features with either `IF(100, auto, 1.0)` (ROC 0.785/PR 0.411, 3.4s) or the 200/256/1.0 above (6.8s). All 4 exist in `preprocess_pipeline.py` already (`log_energy, hour_x/y, robust_z_168h`); no holiday calendar, no `gte_*`/weather/meta. Prefer 200/256/1.0 if 7s fit is affordable, else defaults.
3. **What helped:** `robust_z_168` (past-168h median/IQR on log scale) alone: ROC +0.034, PR ×3.3 (0.123→0.411), F1 ×3.3 (0.142→0.475). Small `max_samples` (256) over large (0.5–1.0 of 1.3M). Time extras hurt (F1 0.142→0.057); `dLag24`/`z_24` add nothing over `z_168`.
4. **Keep:** `log_meter, hour_x, hour_y, robust_z_168` (past-only, `min_periods=72`). Drop `weekday_x/y, month_x/y, is_holiday, dLag24, robust_z_24`.
5. **Final params:** `IsolationForest(n_estimators=200, max_samples=256, max_features=1.0, contamination=0.02, random_state=42, n_jobs=-1)`. Budget option: `(100, "auto", 1.0)` — same ROC within noise, PR 0.41 vs 0.43.
6. **Threshold:** 99th percentile of scores (≈0.64 on reference model; recompute per final model) — P 0.85 / R 0.37 / F1 0.51, best F1. Balanced alternative: 98th (≈0.61) — P 0.51 / R 0.44. procedure, not a constant: take the percentile on each IITD building's own scores (LEAD cutoffs do not transfer).
7. **Tradeoffs:** Exp2a matches Exp2c F1 but needs an extra window for no gain — reject. 300 trees adds ~0.001 ROC for 8x runtime — reject. `max_features=0.5` only helps when `max_samples` is large; with 256-row subsamples, full features win. IITD alignment needed: `preprocess_pipeline.py` computes rolling **unshifted** (includes current hour); switch to `shift(1)` past-only + `min_periods 72` for the 168h window to match this model exactly.
