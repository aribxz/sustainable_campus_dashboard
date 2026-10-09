from functools import lru_cache

import numpy as np
import pandas as pd
from data.building_registry import BUILDINGS

RECENT_WINDOW = 168
ATTENTION_CUTOFF = 0.05


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
        return _clean(value)
    return value


def _iso(ts) -> str | None:
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).isoformat()


@lru_cache(maxsize=None)
def load_building(building_id: str) -> pd.DataFrame:
    if building_id not in BUILDINGS:
        raise KeyError(f"Unknown building: {building_id}")
    df = pd.read_csv(BUILDINGS[building_id]["dataset"])
    if "hour_start" in df.columns:
        df = df.rename(columns={"hour_start": "timestamp"})
    if "log_energy" in df.columns:
        df = df.rename(columns={"log_energy": "log_meter"})
    if "building" not in df.columns:
        df["building"] = BUILDINGS[building_id]["name"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _status(df: pd.DataFrame) -> tuple[str, float]:
    scorable = df[df["anomaly_score"].notna()].tail(RECENT_WINDOW)
    if len(scorable) == 0:
        return "normal", 0.0
    rate = float((scorable["anomaly_flag_p98"] == 1).mean())
    if rate >= ATTENTION_CUTOFF:
        return "critical", rate
    if rate > 0:
        return "attention", rate
    return "normal", rate


def get_overview(building_id: str) -> dict:
    df = load_building(building_id)
    name = BUILDINGS[building_id]["name"]
    scorable = df[df["anomaly_score"].notna()]
    energy = df["energy_kwh"].dropna()
    latest_idx = df["energy_kwh"].last_valid_index()
    p98 = df[df["anomaly_flag_p98"] == 1]
    p99 = df[df["anomaly_flag_p99"] == 1]
    status, recent_rate = _status(df)
    worst_idx = df["anomaly_score"].idxmax() if len(scorable) else None
    return {
        "id": building_id,
        "name": name,
        "status": status,
        "status_rule": (
            f"p98 rate over last {RECENT_WINDOW} scorable hours: "
            f"0=normal, <{ATTENTION_CUTOFF}=attention, "
            f">={ATTENTION_CUTOFF}=critical (recent={recent_rate:.3f})"
        ),
        "latest": {
            "timestamp": _iso(df.loc[latest_idx, "timestamp"])
            if latest_idx is not None
            else None,
            "energy_kwh": _clean(
                df.loc[latest_idx, "energy_kwh"] if latest_idx is not None else None
            ),
        },
        "energy": {
            "mean_kwh": _clean(energy.mean() if len(energy) else None),
            "max_kwh": _clean(energy.max() if len(energy) else None),
            "scorable_observations": int(len(scorable)),
        },
        "anomalies": {
            "p98_count": int(len(p98)),
            "p98_rate": _clean(len(p98) / len(scorable) if len(scorable) else None),
            "p99_count": int(len(p99)),
            "p99_rate": _clean(len(p99) / len(scorable) if len(scorable) else None),
            "latest_anomaly_timestamp": _iso(p98["timestamp"].max())
            if len(p98)
            else None,
        },
        "worst": {
            "anomaly_score": _clean(
                df.loc[worst_idx, "anomaly_score"] if worst_idx is not None else None
            ),
            "robust_z_168": _clean(
                df.loc[worst_idx, "robust_z_168"] if worst_idx is not None else None
            ),
        },
    }


def get_campus_overview() -> list:
    cards = []
    for building_id in BUILDINGS:
        ov = get_overview(building_id)
        cards.append(
            {
                "id": ov["id"],
                "name": ov["name"],
                "status": ov["status"],
                "anomaly_rate": ov["anomalies"]["p98_rate"],
                "latest_energy_kwh": ov["latest"]["energy_kwh"],
                "latest_timestamp": ov["latest"]["timestamp"],
            }
        )
    return cards
