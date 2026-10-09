import numpy as np
import pandas as pd

from services.building_service import _clean, _iso, load_building

TREND_RANGES = {"24h": 24, "7d": 24 * 7, "30d": 24 * 30, "all": None}


def get_trend(building_id: str, span: str = "7d") -> dict:
    if span not in TREND_RANGES:
        raise ValueError(f"range must be one of {sorted(TREND_RANGES)}")
    df = load_building(building_id)
    n = TREND_RANGES[span]
    window = df if n is None else df.tail(n)
    return {
        "building_id": building_id,
        "range": span,
        "label": "Historical Energy Trend",
        "points": [
            {
                "timestamp": _iso(row.timestamp),
                "energy_kwh": _clean(row.energy_kwh),
                "anomaly_score": _clean(row.anomaly_score),
                "anomaly_flag_p98": int(row.anomaly_flag_p98)
                if not pd.isna(row.anomaly_flag_p98)
                else 0,
                "robust_z_168": _clean(row.robust_z_168),
            }
            for row in window.itertuples()
        ],
    }


def get_anomalies(
    building_id: str,
    limit: int = 20,
    offset: int = 0,
    severity: str = "p98",
    start: str | None = None,
    end: str | None = None,
) -> dict:
    if severity not in ("p98", "p99"):
        raise ValueError("severity must be 'p98' or 'p99'")
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    df = load_building(building_id)
    col = "anomaly_flag_p98" if severity == "p98" else "anomaly_flag_p99"
    flagged = df[df[col] == 1].copy()
    if start:
        flagged = flagged[flagged["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        flagged = flagged[flagged["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    total = len(flagged)
    page = flagged.sort_values("anomaly_score", ascending=False).iloc[
        offset : offset + limit
    ]
    p98_thr = float(np.nanquantile(df["anomaly_score"], 0.98))
    p99_thr = float(np.nanquantile(df["anomaly_score"], 0.99))
    return {
        "building_id": building_id,
        "severity": severity,
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "records": [
            {
                "timestamp": _iso(row.timestamp),
                "energy_kwh": _clean(row.energy_kwh),
                "anomaly_score": _clean(row.anomaly_score),
                "robust_z_168": _clean(row.robust_z_168),
                "p98_threshold": p98_thr,
                "p99_threshold": p99_thr,
                "is_p98_anomaly": bool(row.anomaly_flag_p98 == 1),
                "is_p99_anomaly": bool(row.anomaly_flag_p99 == 1),
                "coverage": _clean(row.coverage) if "coverage" in df.columns else None,
            }
            for row in page.itertuples()
        ],
    }


def get_insights(building_id: str) -> dict:
    from data.building_registry import BUILDINGS

    return {
        "building": BUILDINGS[building_id]["name"],
        "available": False,
        "message": "Recommendation engine will be connected here.",
    }
