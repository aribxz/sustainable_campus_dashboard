# CCH — Implementation Report
**Sustainable Campus Dashboard (Version A — historical-data hackathon demo)**

## What it is
Frozen LEAD-trained Isolation Forest flags unusual energy use across 6 IIT Delhi building groups, served via Flask JSON API + self-contained frontend pages. All statuses are anomaly-health, not confirmed faults.

## ML Pipeline (`Code/ml/`)
- **Model:** `IsolationForest(200, max_samples=256, max_features=1.0, contamination=0.02)` trained on LEAD normal rows only. Validation: ROC-AUC 0.784 / PR-AUC 0.431.
- **Features (4):** `[log_meter, hour_x, hour_y, robust_z_168]` — past-only 168h median/IQR baseline (`shift(1)`, `min_periods=72`).
- **Scripts:** `preprocessing_pipeline.py` (LEAD train), `preprocess_campus.py` (minute → hourly, gaps kept as NaN, <50% coverage masked), `score_campus.py` + `score_library.py` (per-building p98/p99 thresholds, LEAD cutoff never reused), `test_pipeline.py`, `EXPERIMENT_REPORT.md`.
- **Outputs:** `Datasets/IITD/preprocessed/*_hourly.csv` → `Datasets/IITD/scored/*_scored.csv` + `campus_anomalies.csv`.

## Backend (`Code/backend/`)
- **App:** `app.py` — Flask, serves frontend + 5 JSON endpoints. Run: `python Code/backend/app.py` (`http://127.0.0.1:5000`).
- **API:** `GET /api/buildings`, `GET /api/buildings/<id>`, `GET .../trend?range=24h|7d|30d|all`, `GET .../anomalies?limit&offset&severity=p98|p99&start&end`, `GET .../insights` (placeholder). Full contract in `API.md`.
- **Logic:** `data/building_registry.py` (6 IDs: `a-block,b-block,c-block,library,hostels,mess`; single source of truth), `services/building_service.py` (overview + normal/attention/critical rule on last 168h p98 rate), `services/analytics_service.py` (trend slice, worst-first anomaly table).
- **Runtime:** `requirements.txt` pinned (Flask, pandas, numpy, gunicorn) for Render.

## Frontend (`Code/frontend/`)
- 8 single-file pages: `index.html` (campus map), `about.html`, `a-block,b-block,c-block,library,hostels,mess.html` (Sections 1–4: overview, trend chart, anomaly explorer, insights stub).
- Same-origin fetch only, no CSV/analytics. Light/dark mode with persistence, scroll-reveal, sortable tables.

## Data (`Datasets/`)
- `IITD/`: raw mains CSVs (transformers 1-3, mess, hostels, library) — git-ignored (>100MB). Only 6 small `scored/` CSVs versioned per `.gitignore`.
- `LEAD/`: `train/test` + features + predictions. `Statistics.xlsx`: model progression sheet.

## Current State / Next
- Done: end-to-end frozen pipeline → API → UI works locally.
- Stubs: `/insights` returns "Recommendation engine will be connected here"; frontend "Coming Next" cards list Recommendation Engine + Anomaly Simulation.
