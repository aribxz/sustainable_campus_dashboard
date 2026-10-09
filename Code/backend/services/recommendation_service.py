import numpy as np
import pandas as pd

from data.building_registry import BUILDINGS
from services.building_service import _clean, _iso, load_building

RULES_VERSION = "2.0"

DROP_Z = -3.0
SPIKE_Z = 3.0
HIGH_Z = 8.0
MED_Z = 5.0

CLUSTER_WINDOW_H = 24
CLUSTER_MIN_COUNT = 3
HIGH_DENSITY = 8.0
MED_DENSITY = 4.0
HIGH_COUNT_GATE = 5
MAX_CLUSTERS = 3
DIR_Z = 1.5

IST_OFFSET = pd.Timedelta(hours=5, minutes=30)
OVERNIGHT_IST_START = 0
OVERNIGHT_IST_END = 6
OVERNIGHT_MIN_COUNT = 5
OVERNIGHT_MIN_DATES = 3
HIGH_NIGHT_SHARE = 0.25
MED_NIGHT_SHARE = 0.10
RECENT_DAYS = 90

UNSCORABLE_RATIO_THRESHOLD = 0.05
MED_UNSCORABLE_RATIO = 0.2
MIN_GAP_H = 24
MED_GAP_H = 168

DEFAULT_LIMIT = 10
MAX_LIMIT = 50

SEVERITIES = ("p98", "p99")


def _severity_for_z(z) -> str:
    az = abs(float(z))
    if az >= HIGH_Z:
        return "high"
    if az >= MED_Z:
        return "medium"
    return "low"


def _severity_for_cluster(n: int, density: float) -> str:
    if n >= HIGH_COUNT_GATE and density >= HIGH_DENSITY:
        return "high"
    if density >= MED_DENSITY:
        return "medium"
    return "low"


def _cluster_direction(mean_z) -> str:
    if mean_z is None or (isinstance(mean_z, float) and np.isnan(mean_z)):
        return "mixed"
    try:
        z = float(mean_z)
    except (TypeError, ValueError):
        return "mixed"
    if pd.isna(z):
        return "mixed"
    if z >= DIR_Z:
        return "high-use"
    if z <= -DIR_Z:
        return "low-use"
    return "mixed"


def _ist_parts(df: pd.DataFrame):
    ist = df["timestamp"] + IST_OFFSET
    return ist.dt.hour, ist.dt.date


def _severity_for_overnight(recent_share: float) -> str:
    if recent_share >= HIGH_NIGHT_SHARE:
        return "high"
    if recent_share >= MED_NIGHT_SHARE:
        return "medium"
    return "low"


def _confidence(severity: str) -> str:
    return {"high": "strong", "medium": "moderate"}.get(severity, "limited")


def _flag_col(severity: str) -> str:
    if severity not in SEVERITIES:
        raise ValueError("severity must be 'p98' or 'p99'")
    return "anomaly_flag_p98" if severity == "p98" else "anomaly_flag_p99"


def _get(df: pd.DataFrame, row, col, default=None):
    if col not in df.columns:
        return default
    try:
        v = row[col] if isinstance(row, dict) else getattr(row, col, default)
    except Exception:
        return default
    if pd.isna(v):
        return default
    return v


def _filter_window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    out = df
    if start:
        out = out[out["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        out = out[out["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    return out


def _find_clusters(flagged: pd.DataFrame) -> list:
    if len(flagged) < CLUSTER_MIN_COUNT:
        return []
    ts = flagged.sort_values("timestamp").reset_index(drop=True)
    times = ts["timestamp"].tolist()
    windows = []
    j = 0
    for i in range(len(times)):
        j = max(j, i)
        while j + 1 < len(times) and (times[j + 1] - times[i]).total_seconds() <= CLUSTER_WINDOW_H * 3600:
            j += 1
        if j - i + 1 >= CLUSTER_MIN_COUNT:
            windows.append((i, j))
    if not windows:
        return []
    merged = [windows[0]]
    for s, e in windows[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    events = []
    for s, e in merged:
        sub = ts.iloc[s : e + 1]
        events.append(
            {
                "start": sub["timestamp"].iloc[0],
                "end": sub["timestamp"].iloc[-1],
                "count": int(e - s + 1),
                "rows": sub,
            }
        )
    events.sort(key=lambda ev: (ev["count"], ev["start"]), reverse=True)
    return events[:MAX_CLUSTERS]


def _longest_gap(df: pd.DataFrame) -> dict | None:
    if "anomaly_score" not in df.columns or len(df) == 0:
        return None
    mask = df["anomaly_score"].isna().to_numpy()
    if not mask.any():
        return None
    best_len, best_s = 0, None
    cur_len, cur_s = 0, None
    for i, m in enumerate(mask):
        if m:
            if cur_s is None:
                cur_s = i
                cur_len = 1
            else:
                cur_len += 1
        else:
            if cur_len > best_len:
                best_len, best_s = cur_len, cur_s
            cur_s, cur_len = None, 0
    if cur_len > best_len:
        best_len, best_s = cur_len, cur_s
    if best_len == 0 or best_s is None:
        return None
    sub = df.iloc[best_s : best_s + best_len]
    return {
        "length": int(best_len),
        "start": sub["timestamp"].iloc[0],
        "end": sub["timestamp"].iloc[-1],
    }


def _single_rec(building_id: str, name: str, row, category: str, rule: str) -> dict:
    z = float(row.robust_z_168)
    sev = _severity_for_z(z)
    ts = _iso(row.timestamp)
    energy = _clean(getattr(row, "energy_kwh", None))
    score = _clean(getattr(row, "anomaly_score", None))
    is_drop = category == "unusual_drop"
    direction = "below" if is_drop else "above"
    title = (
        f"Unusually {'low' if is_drop else 'high'} historical reading — {name}"
    )
    evidence = {
        "timestamp": ts,
        "energy_kwh": energy,
        "robust_z_168": _clean(z),
        "anomaly_score": score,
        "note": (
            f"Flagged hour observed {energy} kWh at {ts}, "
            f"{abs(z):.1f} robust deviations {direction} its own recent history. "
            "Anomaly score is detector output (higher = more anomalous), not a probability."
        ),
    }
    cov = _clean(getattr(row, "coverage", None)) if hasattr(row, "coverage") else None
    if cov is not None:
        evidence["coverage"] = cov
    if is_drop:
        causes = [
            "Planned shutdown, holiday schedule, or unusually low occupancy during that hour.",
            "HVAC/lighting operating on a reduced schedule.",
            "Meter connectivity or supply interruption (only if the drop was unexpected).",
        ]
        actions = [
            "Check whether the building was intentionally shut down or on a holiday schedule.",
            "Review HVAC and lighting operating schedules for that period.",
            "If unexpected, verify meter connectivity and electrical supply records.",
        ]
    else:
        causes = [
            "High-load equipment running outside its normal schedule.",
            "Unusual occupancy or event load during that hour.",
            "Meter or aggregation artefact (verify against an independent source).",
        ]
        actions = [
            "Inspect high-load equipment and operating schedules for that hour.",
            "Check for unexpected equipment running outside normal hours.",
            "Verify the reading against the meter or another available source.",
        ]
    safe_id_ts = str(ts).replace(":", "").replace("+", "p") if ts else "unknown"
    return {
        "id": f"{building_id}-{category}-{safe_id_ts}",
        "building_id": building_id,
        "building_name": name,
        "category": category,
        "severity": sev,
        "title": title,
        "summary": (
            f"One flagged hour ({ts}) with robust z {z:+.1f}: unusually "
            f"{'low' if is_drop else 'high'} vs its own past-{168}h baseline. "
            "Historical pattern only — not a confirmed fault."
        ),
        "evidence": evidence,
        "possible_causes": causes,
        "recommended_actions": actions,
        "timestamp": ts,
        "start": ts,
        "end": ts,
        "confidence": _confidence(sev),
        "status": "needs_review",
        "rule": rule,
    }


def _cluster_rec(building_id: str, name: str, ev: dict, idx: int) -> dict:
    rows = ev["rows"]
    zs = rows["robust_z_168"].dropna() if "robust_z_168" in rows.columns else pd.Series([], dtype=float)
    mean_z = float(zs.mean()) if len(zs) else None
    direction = _cluster_direction(mean_z)
    s, e = _iso(ev["start"]), _iso(ev["end"])
    hours = max(1, int((ev["end"] - ev["start"]).total_seconds() // 3600 + 1))
    density = ev["count"] / (hours / 24)
    sev = _severity_for_cluster(ev["count"], density)
    if direction == "low-use":
        title, phrase = "An unusual spell — mostly low use", "mostly lower-than-usual use"
    elif direction == "high-use":
        title, phrase = "An unusual spell — mostly high use", "mostly higher-than-usual use"
    else:
        title, phrase = "An unusual spell — mixed pattern", "a mix of lower and higher use"
    safe = str(s).replace(":", "").replace("+", "p") if s else f"c{idx}"
    return {
        "id": f"{building_id}-repeated_anomalies-{safe}",
        "building_id": building_id,
        "building_name": name,
        "category": "repeated_anomalies",
        "severity": sev,
        "title": f"{title} — {name} ({ev['count']} hours)",
        "summary": (
            f"{ev['count']} flagged hours over ~{hours}h "
            f"({density:.1f} per day) from {s} to {e}: {phrase}. "
            "Suggests a sustained unusual period rather than an isolated hour. "
            "Historical pattern only — not a confirmed fault."
        ),
        "evidence": {
            "start": s,
            "end": e,
            "affected_hours": ev["count"],
            "span_hours": hours,
            "density_per_24h": _clean(round(float(density), 2)),
            "direction": direction,
            "mean_robust_z_168": _clean(mean_z),
            "max_anomaly_score": _clean(float(rows["anomaly_score"].max())),
            "note": "One recommendation per cluster; individual hours listed under Anomaly Explorer.",
        },
        "possible_causes": [
            "Planned shutdown, unusual occupancy, or operational change spanning the period.",
            "Meter or data-collection issue affecting several consecutive hours.",
        ],
        "recommended_actions": [
            "Review the operating schedule covering the affected period.",
            "Check whether a planned shutdown, unusual occupancy, or meter issue explains the cluster.",
        ],
        "timestamp": s,
        "start": s,
        "end": e,
        "confidence": _confidence(sev),
        "status": "needs_review",
        "rule": f"C: >= {CLUSTER_MIN_COUNT} flags / rolling {CLUSTER_WINDOW_H}h; severity by density",
    }


def _overnight_rec(building_id: str, name: str, window_df: pd.DataFrame, col: str) -> dict | None:
    if col not in window_df.columns or "anomaly_score" not in window_df.columns:
        return None
    scorable = window_df[window_df["anomaly_score"].notna()].copy()
    if len(scorable) == 0:
        return None
    hours, dates = _ist_parts(scorable)
    scorable = scorable.assign(_h_ist=hours.to_numpy(), _d_ist=dates.to_numpy())
    is_night = (scorable["_h_ist"] >= OVERNIGHT_IST_START) & (scorable["_h_ist"] < OVERNIGHT_IST_END)
    night = scorable[is_night]
    day = scorable[~is_night]
    night_flags = night[night[col] == 1]
    if len(night_flags) < OVERNIGHT_MIN_COUNT:
        return None
    n_dates = night_flags["_d_ist"].nunique()
    if n_dates < OVERNIGHT_MIN_DATES:
        return None
    night_rate = float((night[col] == 1).mean()) if len(night) else 0.0
    day_rate = float((day[col] == 1).mean()) if len(day) else 0.0
    if not night_rate > day_rate:
        return None
    enrichment = night_rate / day_rate if day_rate > 0 else float("inf")

    end = window_df["timestamp"].max()
    recent = scorable[scorable["timestamp"] >= end - pd.Timedelta(days=RECENT_DAYS)].copy()
    r_night = recent[(recent["_h_ist"] >= OVERNIGHT_IST_START) & (recent["_h_ist"] < OVERNIGHT_IST_END)]
    r_night_flags = r_night[r_night[col] == 1]
    recent_nights_total = int(recent["_d_ist"].nunique())
    recent_nights_hit = int(r_night_flags["_d_ist"].nunique()) if len(r_night_flags) else 0
    recent_share = recent_nights_hit / recent_nights_total if recent_nights_total else 0.0
    sev = _severity_for_overnight(recent_share)
    recent_30 = r_night_flags[r_night_flags["timestamp"] >= end - pd.Timedelta(days=30)]
    most_recent = _iso(r_night_flags["timestamp"].max()) if len(r_night_flags) else None
    examples = [_iso(t) for t in night_flags.sort_values("anomaly_score", ascending=False)["timestamp"].head(3)]
    all_nights = int(scorable["_d_ist"].nunique())
    return {
        "id": f"{building_id}-overnight_pattern",
        "building_id": building_id,
        "building_name": name,
        "category": "overnight_pattern",
        "severity": sev,
        "title": f"Late nights stand out — {name}",
        "summary": (
            f"Late-night (00:00–06:00 IST) hours flag at {night_rate * 100:.1f}% vs "
            f"{day_rate * 100:.1f}% daytime ({enrichment:.1f}x). {n_dates} of {all_nights} "
            f"nights affected overall; in the last {RECENT_DAYS} days, {recent_nights_hit} of "
            f"{recent_nights_total} nights ({recent_share * 100:.0f}%) were affected"
            + (f", most recently {most_recent}." if most_recent else ".")
            + " Overnight use is not assumed to be zero; this flags recurrence only. "
            "Historical pattern only — not a confirmed fault."
        ),
        "evidence": {
            "overnight_window": "00:00-06:00 IST (UTC+05:30)",
            "overnight_flags": int(len(night_flags)),
            "distinct_dates": int(n_dates),
            "night_rate": _clean(round(night_rate, 4)),
            "day_rate": _clean(round(day_rate, 4)),
            "enrichment_vs_day": _clean(round(enrichment, 2)) if enrichment != float("inf") else None,
            "share_of_nights": _clean(round(n_dates / all_nights, 4)) if all_nights else None,
            "recent_days": RECENT_DAYS,
            "recent_nights_affected": recent_nights_hit,
            "recent_nights_total": recent_nights_total,
            "recent_share": _clean(round(recent_share, 4)),
            "recent_nights_last_30d": int(recent_30["_d_ist"].nunique()) if len(recent_30) else 0,
            "most_recent_night": most_recent,
            "first": _iso(night_flags["timestamp"].min()),
            "last": _iso(night_flags["timestamp"].max()),
            "example_timestamps": examples,
        },
        "possible_causes": [
            "Regular overnight HVAC, lighting, or equipment schedules.",
            "Planned overnight occupancy or shutdown-period baseload.",
        ],
        "recommended_actions": [
            "Review overnight HVAC, lighting, and equipment schedules.",
            "Compare against planned occupancy or shutdown periods.",
        ],
        "timestamp": most_recent,
        "start": _iso(night_flags["timestamp"].min()),
        "end": most_recent,
        "confidence": _confidence(sev),
        "status": "needs_review",
        "rule": (
            f"D: IST 00:00–06:00 (UTC+05:30), >= {OVERNIGHT_MIN_COUNT} night flags on "
            f">= {OVERNIGHT_MIN_DATES} nights with night rate above day rate"
        ),
    }


def _data_quality_rec(building_id: str, name: str, df: pd.DataFrame) -> dict | None:
    n = len(df)
    if n == 0:
        return None
    if "is_masked" in df.columns:
        masked = int((df["is_masked"] == 1).sum())
    else:
        masked = int(df["energy_kwh"].isna().sum()) if "energy_kwh" in df.columns else 0
    unscorable = int(df["anomaly_score"].isna().sum()) if "anomaly_score" in df.columns else 0
    unscorable_share = (unscorable / n) if n else 0.0
    gap = _longest_gap(df)
    gap_len = gap["length"] if gap else 0
    if not (unscorable_share >= UNSCORABLE_RATIO_THRESHOLD or gap_len >= MIN_GAP_H):
        return None
    sev = "medium" if (unscorable_share >= MED_UNSCORABLE_RATIO or gap_len >= MED_GAP_H) else "low"
    s = _iso(gap["start"]) if gap else None
    e = _iso(gap["end"]) if gap else None
    return {
        "id": f"{building_id}-data_quality",
        "building_id": building_id,
        "building_name": name,
        "category": "data_quality",
        "severity": sev,
        "title": f"Some history is missing — {name}",
        "summary": (
            f"{unscorable} of {n} hours ({unscorable_share * 100:.1f}%) could not be scored"
            + (
                f", mostly one blind spell of {gap_len}h ({s} to {e}). "
                if gap
                else ". "
            )
            + f"Of the unscorable hours, {masked} were masked by the low-coverage rule. "
            "Gaps are reported as missing data, not an energy fault. No imputation "
            "was performed; unscorable rows were never treated as normal."
        ),
        "evidence": {
            "total_hours": n,
            "unscorable_hours": unscorable,
            "unscorable_share": _clean(round(float(unscorable_share), 4)),
            "masked_hours": masked,
            "masked_ratio": _clean(round(float(masked / n) if n else 0.0, 4)),
            "longest_gap_hours": gap_len,
            "longest_gap_start": s,
            "longest_gap_end": e,
        },
        "possible_causes": [
            "Meter outage or connectivity loss during the gap period.",
            "Low-coverage hours masked by the preprocessing rule (<50% minute samples).",
        ],
        "recommended_actions": [
            "Treat gap periods as unknown; do not interpolate them for decisions.",
            "If coverage matters, check meter logs for the longest gap window.",
        ],
        "timestamp": s,
        "start": s,
        "end": e,
        "confidence": _confidence(sev),
        "status": "needs_review",
        "rule": f"E: unscorable share >= {UNSCORABLE_RATIO_THRESHOLD} or gap >= {MIN_GAP_H}h",
    }


def get_insights(
    building_id: str,
    severity: str = "p98",
    limit: int = DEFAULT_LIMIT,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    if building_id not in BUILDINGS:
        raise KeyError(f"Unknown building: {building_id}")
    col = _flag_col(severity)
    try:
        limit = max(1, min(int(limit), MAX_LIMIT))
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer")
    name = BUILDINGS[building_id]["name"]

    df = load_building(building_id)
    window = _filter_window(df, start, end)
    flagged = window[window[col] == 1].copy() if col in window.columns else window.iloc[0:0].copy()
    if "anomaly_score" in flagged.columns:
        flagged = flagged[flagged["anomaly_score"].notna()]

    recs: list[dict] = []

    for i, ev in enumerate(_find_clusters(flagged)):
        recs.append(_cluster_rec(building_id, name, ev, i))

    over = _overnight_rec(building_id, name, _filter_window(df, start, end), col)
    if over:
        recs.append(over)

    dq = _data_quality_rec(building_id, name, _filter_window(df, start, end))
    if dq:
        recs.append(dq)

    if "robust_z_168" in flagged.columns and len(flagged):
        drops = flagged[flagged["robust_z_168"] <= DROP_Z]
        spikes = flagged[flagged["robust_z_168"] >= SPIKE_Z]
        if len(drops):
            worst_drop = drops.sort_values("robust_z_168").iloc[0]
            recs.append(_single_rec(building_id, name, worst_drop, "unusual_drop", f"A: flagged && robust_z_168 <= {DROP_Z}"))
        if len(spikes):
            worst_spike = spikes.sort_values("robust_z_168", ascending=False).iloc[0]
            recs.append(_single_rec(building_id, name, worst_spike, "unusual_spike", f"B: flagged && robust_z_168 >= +{SPIKE_Z}"))

    if not recs and len(flagged):
        worst = flagged.sort_values("anomaly_score", ascending=False).iloc[0]
        ts = _iso(worst["timestamp"])
        recs.append(
            {
                "id": f"{building_id}-general_review-{str(ts).replace(':', '').replace('+', 'p') if ts else 'worst'}",
                "building_id": building_id,
                "building_name": name,
                "category": "general_review",
                "severity": "low",
                "title": f"Historical anomalies merit review — {name}",
                "summary": (
                    f"Flagged hours are present but none meets the strong drop/spike "
                    f"(|z|>=3), cluster, or overnight-recurrence rules in this window. "
                    f"Worst historical hour {ts} shown for review. Not a confirmed fault."
                ),
                "evidence": {
                    "timestamp": ts,
                    "energy_kwh": _clean(getattr(worst, "energy_kwh", None)),
                    "robust_z_168": _clean(getattr(worst, "robust_z_168", None)),
                    "anomaly_score": _clean(getattr(worst, "anomaly_score", None)),
                    "flagged_hours_in_window": int(len(flagged)),
                },
                "possible_causes": ["No strong directional pattern; manual review needed."],
                "recommended_actions": [
                    "Open the Anomaly Explorer table for the worst-first flagged hours.",
                    "Compare flagged hours against operating schedules before acting.",
                ],
                "timestamp": ts,
                "start": ts,
                "end": ts,
                "confidence": "limited",
                "status": "needs_review",
                "rule": "fallback: flagged present, no A-D rule fired",
            }
        )

    rank = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: (rank.get(r["severity"], 3), str(r.get("timestamp") or "")))
    total = len(recs)
    recs = recs[:limit]

    return {
        "building": name,
        "available": True,
        "message": (
            f"{total} historical insight(s) for {name} ({severity}). "
            "Patterns only — not confirmed faults."
            if total
            else f"No strong historical pattern for {name} in this window."
        ),
        "building_id": building_id,
        "severity": severity,
        "rules_version": RULES_VERSION,
        "total": total,
        "limit": limit,
        "recommendations": recs,
    }
