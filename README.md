# Sustainable Campus Dashboard

Version A — historical-data hackathon demo. A frozen LEAD-trained Isolation
Forest flags unusual energy use across six IIT Delhi campus buildings,
served through a Flask JSON API with self-contained frontend pages.

## Layout

```text
CCH/
  Code/backend/    # Flask app, services, building registry, API.md
  Code/frontend/   # index, about + 6 building pages (single-file HTML)
  Code/ml/         # LEAD training, IITD preprocessing, scoring scripts
  Datasets/        # meter data (NOT versioned, see below)
    IITD/          # raw mains CSVs, preprocessed/, scored/, campus_anomalies.csv
    LEAD/          # train.csv, test.csv, train_features.csv, test_features.csv
```

## Data (not in git)

Meter CSVs are excluded via `.gitignore` (GitHub rejects files over 100 MB).
Place them locally as:

```text
Datasets/IITD/{transformer_1,transformer_2,transformer_3,mess_build_mains,hostels,library_build_mains}.csv
Datasets/LEAD/{train,test,train_features,test_features}.csv
```

Then regenerate derivatives: `Code/ml/preprocess_campus.py` →
`Code/ml/score_campus.py` (+ `score_library.py` for the Library).

## Run the backend

```text
python Code/backend/app.py   # http://127.0.0.1:5000
```

API (full contract in `Code/backend/API.md`):

```text
GET /api/buildings
GET /api/buildings/<a-block|b-block|c-block|library|hostels|mess>
GET /api/buildings/<id>/trend?range=24h|7d|30d|all
GET /api/buildings/<id>/anomalies?limit=&offset=&severity=p98|p99&start=&end=
GET /api/buildings/<id>/insights   # recommendation-engine placeholder
```

Pages: `/` (campus map), `/building/<id>`, `/about`.

## ML pipeline (frozen — do not retrain)

Features `[log_meter, hour_x, hour_y, robust_z_168]` with past-only
168-hour baselines; `IsolationForest(200, max_samples=256, max_features=1.0,
contamination=0.02, random_state=42)` fit on LEAD normal rows only
(validation ROC-AUC 0.79). IITD buildings get per-building p98/p99
thresholds; the LEAD cutoff is never reused. Statuses are anomaly-health,
never confirmed faults.

## Frontend

Drop-in single-file pages in `Code/frontend/` (served by the backend, same
origin). Light/dark mode with persistence, scroll-reveal campus map,
per-building trend charts, sortable anomaly explorer. No CSV reading or
analytics in frontend code.
