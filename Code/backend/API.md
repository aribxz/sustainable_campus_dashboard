# Campus Anomaly Demo — Backend API

Historical-data demo (Version A). All data comes from precomputed scored
CSVs; nothing is live, nothing is rescored. Statuses are anomaly-health,
never confirmed faults.

Base URL (dev): `http://127.0.0.1:5000`

Building IDs: `a-block, b-block, c-block, library, hostels, mess`
(`ground` is a cosmetic map placeholder with no API).

## `GET /api/buildings`

Campus overview for the index map (one card per building).

Response: JSON array of:
```json
[{ "id": "a-block", "name": "A Block", "status": "attention",
   "anomaly_rate": 0.02, "latest_energy_kwh": 27.4,
   "latest_timestamp": "2017-12-31T18:00:00+00:00" }]
```
`status` rule: p98 rate over last 168 scorable hours →
`0 = normal`, `< 0.05 = attention`, `>= 0.05 = critical`.

## `GET /api/buildings/<building_id>`

Section 1 overview + page metadata. No parameters.

```json
{ "id": "a-block", "name": "A Block", "status": "attention",
  "status_rule": "p98 rate over last 168 scorable hours: ...",
  "latest": { "timestamp": "...", "energy_kwh": 27.4 },
  "energy": { "mean_kwh": 27.8, "max_kwh": 120.4,
              "scorable_observations": 23321 },
  "anomalies": { "p98_count": 467, "p98_rate": 0.02,
                 "p99_count": 234, "p99_rate": 0.01,
                 "latest_anomaly_timestamp": "..." },
  "worst": { "anomaly_score": 0.674, "robust_z_168": -7.6 } }
```

## `GET /api/buildings/<building_id>/trend?range=<span>`

Section 2 — Historical Energy Trend (never "live").
`span`: `24h | 7d (default) | 30d | all`.

```json
{ "building_id": "a-block", "range": "7d",
  "label": "Historical Energy Trend",
  "points": [{ "timestamp": "...", "energy_kwh": 27.4,
    "anomaly_score": 0.51, "anomaly_flag_p98": 0,
    "robust_z_168": -0.3 }] }
```
Masked/gap hours appear with `energy_kwh: null` (gaps stay visible).

## `GET /api/buildings/<building_id>/anomalies`

Section 3 — worst-first anomaly table.
Params: `limit` (default 20, max 500), `offset` (default 0),
`severity` (`p98` default | `p99`), `start`, `end` (ISO timestamps).

Example: `/api/buildings/a-block/anomalies?limit=20&severity=p99`

```json
{ "building_id": "a-block", "severity": "p98", "total": 467,
  "limit": 20, "offset": 0,
  "records": [{ "timestamp": "...", "energy_kwh": 1.01,
    "anomaly_score": 0.674, "robust_z_168": -4.2,
    "p98_threshold": 0.597, "p99_threshold": 0.617,
    "is_p98_anomaly": true, "is_p99_anomaly": true,
    "coverage": 1.0 }] }
```

## `GET /api/buildings/<building_id>/insights`

Section 4 — rule-based recommendation engine (deterministic, historical only).
Params: `severity` (`p98` default | `p99`), `limit` (default 10, max 50),
`start`, `end` (ISO timestamps, optional).

Response preserves the original envelope (`building`, `available`, `message`)
and adds structured fields:

```json
{ "building": "A Block", "building_id": "a-block", "available": true,
  "message": "6 historical insight(s) for A Block (p98). Patterns only — not confirmed faults.",
  "severity": "p98", "rules_version": "2.0", "total": 6, "limit": 10,
  "recommendations": [{
    "id": "a-block-unusual_drop-2013-12-07T020000p0000",
    "building_id": "a-block", "building_name": "A Block",
    "category": "unusual_drop", "severity": "high",
    "title": "Unusually low historical reading — A Block",
    "summary": "One flagged hour ... Historical pattern only — not a confirmed fault.",
    "evidence": { "timestamp": "...", "energy_kwh": 1.01,
      "robust_z_168": -9.35, "anomaly_score": 0.674 },
    "possible_causes": ["..."], "recommended_actions": ["..."],
    "timestamp": "...", "start": "...", "end": "...",
    "confidence": "strong", "status": "needs_review",
    "rule": "A: flagged && robust_z_168 <= -3.0" }] }
```

`category` is one of `unusual_drop | unusual_spike | repeated_anomalies |
overnight_pattern | data_quality | general_review`. `severity` is
`low | medium | high` per the rules documented in
`services/recommendation_service.py` (v2.0). `confidence` (`limited |
moderate | strong`) mirrors evidence strength. Old consumers using only
`message` keep working. All pre-2.0 response keys are retained
(`masked_ratio` included); 2.0 only adds evidence keys.

v2.0 evidence notes:
- `overnight_pattern` uses campus-local **IST 00:00–06:00 (UTC+05:30)**
  (`evidence.overnight_window`) and fires only when the night anomaly rate
  exceeds the day rate. Evidence carries `night_rate`, `day_rate`,
  `enrichment_vs_day`, `share_of_nights`, plus a trailing-90d recent view
  (`recent_nights_affected/total/share`, `most_recent_night`); severity keys
  off `recent_share` (≥0.25 high, ≥0.10 medium).
- `repeated_anomalies` adds `density_per_24h` and `direction`
  (`low-use | high-use | mixed`, from mean `robust_z_168` at ±1.5); severity
  is density-based (count≥5 and ≥8/day high, ≥4/day medium).
- `data_quality` headlines `unscorable_share` (`unscorable_hours` /
  `total_hours`) with `masked_hours` as a labeled subset.

## Pages (throwaway proof UI)

`/` — campus map order: A Block, Library, C Block + B Block,
cosmetic Ground box, Hostels + Mess.
`/building/<building_id>` — Sections 1–4 wired to the APIs above.

## Errors

Unknown building → `404 {"error": "unknown building: <id>"}`.
Bad `range`/`severity` → `400 {"error": "..."}`.
All numbers are JSON-safe (NaN/Infinity serialized as `null`).

## Connecting a future frontend

Replace files under `Code/frontend/` freely; keep calling the five JSON
endpoints above. No CSV reading or analytics in frontend code — the
registry (`Code/backend/data/building_registry.py`) and services stay the
single source of truth. To run: `python Code/backend/app.py` from the `CCH`
folder (or any cwd — paths resolve from the file location).
