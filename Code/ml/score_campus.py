from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent.parent
DATASETS_DIR = PROJECT_ROOT / "Datasets"
LEAD_TRAIN = DATASETS_DIR / "LEAD" / "train.csv"
PRE_DIR = DATASETS_DIR / "IITD" / "preprocessed"
SCORED_DIR = DATASETS_DIR / "IITD" / "scored"
CAMPUS_ANOMALIES = DATASETS_DIR / "IITD" / "campus_anomalies.csv"

FEATURES = ["log_meter", "hour_x", "hour_y", "robust_z_168"]
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_samples": 256,
    "max_features": 1.0,
    "contamination": 0.02,
    "random_state": 42,
    "n_jobs": -1,
}

BUILDINGS = {
    "A Block": PRE_DIR / "transformer1_hourly.csv",
    "B Block": PRE_DIR / "transformer2_hourly.csv",
    "C Block": PRE_DIR / "transformer3_hourly.csv",
    "Mess": PRE_DIR / "mess_hourly.csv",
    "Hostels": PRE_DIR / "hostel_hourly.csv",
}
OUTPUTS = {
    "A Block": SCORED_DIR / "transformer1_scored.csv",
    "B Block": SCORED_DIR / "transformer2_scored.csv",
    "C Block": SCORED_DIR / "transformer3_scored.csv",
    "Mess": SCORED_DIR / "mess_scored.csv",
    "Hostels": SCORED_DIR / "hostel_scored.csv",
}


def build_lead_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["log_meter"] = np.log1p(df["meter_reading"].clip(lower=0))
    hour = df["timestamp"].dt.hour
    df["hour_x"] = np.cos(2 * np.pi * hour / 24.0)
    df["hour_y"] = np.sin(2 * np.pi * hour / 24.0)
    order = df.sort_values(["building_id", "timestamp"]).index
    shifted = df.loc[order].groupby("building_id")["log_meter"].shift(1)
    rolled = shifted.groupby(df.loc[order, "building_id"]).rolling(168, min_periods=72)
    med = rolled.median().reset_index(level=0, drop=True)
    iqr = rolled.quantile(0.75).reset_index(level=0, drop=True) - rolled.quantile(
        0.25,
    ).reset_index(level=0, drop=True)
    df.loc[order, "robust_z_168"] = ((shifted - med) / (iqr + 1e-6)).to_numpy()
    return df


def train_frozen_model() -> IsolationForest:
    df = pd.read_csv(
        LEAD_TRAIN, usecols=["building_id", "timestamp", "meter_reading", "anomaly"]
    )
    df = build_lead_features(df)
    df = df.dropna(subset=["meter_reading"])
    df = df[df["meter_reading"] > 0].reset_index(drop=True)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    normal = df[df["anomaly"] == 0]
    print(
        f"LEAD train usable: {len(df)} rows / "
        f"{df['building_id'].nunique()} buildings "
        f"({len(normal)} normal for fitting)"
    )
    model = IsolationForest(**MODEL_PARAMS)
    model.fit(normal[FEATURES])
    return model


def score_building(
    building: str, src: Path, model: IsolationForest
) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(src)
    assert df["building"].nunique() == 1 and df["building"].iloc[0] == building
    x = df.rename(columns={"log_energy": "log_meter"})[FEATURES]
    scorable = (
        df["energy_kwh"].notna() & (df["energy_kwh"] > 0) & x.notna().all(axis=1)
    ).to_numpy()
    scores = np.full(len(df), np.nan)
    scores[scorable] = -model.score_samples(x.loc[scorable])
    df["anomaly_score"] = scores
    thr98 = float(np.nanquantile(scores, 0.98))
    thr99 = float(np.nanquantile(scores, 0.99))
    df["anomaly_flag_p98"] = 0
    df["anomaly_flag_p99"] = 0
    df.loc[scorable & (scores >= thr98), "anomaly_flag_p98"] = 1
    df.loc[scorable & (scores >= thr99), "anomaly_flag_p99"] = 1
    keep = [
        "building",
        "hour_start",
        "energy_kwh",
        "power_mean_W",
        "power_min_W",
        "power_max_W",
        "coverage",
        "is_masked",
        "is_missing",
        "log_energy",
        "hour_x",
        "hour_y",
        "robust_z_168",
        "anomaly_score",
        "anomaly_flag_p98",
        "anomaly_flag_p99",
    ]
    return df[keep], {
        "building": building,
        "scorable": int(scorable.sum()),
        "scores": scores[scorable],
        "thr98": thr98,
        "thr99": thr99,
        "flags98": int(df["anomaly_flag_p98"].sum()),
        "flags99": int(df["anomaly_flag_p99"].sum()),
        "masked_flagged": int(
            ((df["is_masked"] == 1) & (df["anomaly_flag_p98"] == 1)).sum()
        ),
    }


def main() -> None:
    model = train_frozen_model()
    model_id = (
        f"IF({MODEL_PARAMS['n_estimators']},{MODEL_PARAMS['max_samples']},"
        f"{MODEL_PARAMS['max_features']},{MODEL_PARAMS['contamination']},"
        f"{MODEL_PARAMS['random_state']})"
    )
    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    frames = []
    for building, src in BUILDINGS.items():
        scored, info = score_building(building, src, model)
        OUTPUTS[building].parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(OUTPUTS[building], index=False)
        summaries.append(info)
        sub = scored[scored["anomaly_flag_p98"] == 1][
            [
                "building",
                "hour_start",
                "energy_kwh",
                "anomaly_score",
                "anomaly_flag_p98",
                "anomaly_flag_p99",
                "robust_z_168",
                "coverage",
            ]
        ].copy()
        sub["threshold_p98"] = info["thr98"]
        sub["threshold_p99"] = info["thr99"]
        frames.append(sub)
        print(f"Wrote {OUTPUTS[building].name}")
    campus = pd.concat(frames, ignore_index=True)
    campus.to_csv(CAMPUS_ANOMALIES, index=False)
    print(f"Wrote campus_anomalies.csv ({len(campus)} p98 records)")

    print("Building | Scorable | p98 thr | p98 n | p98 % | p99 thr | p99 n | p99 %")
    for info in summaries:
        n = info["scorable"]
        print(
            f"{info['building']} | {n} | {info['thr98']:.4f} | {info['flags98']} | "
            f"{info['flags98'] / n * 100:.2f}% | {info['thr99']:.4f} | "
            f"{info['flags99']} | {info['flags99'] / n * 100:.2f}%"
        )
    print(f"Frozen model for all 5: {model_id} | LEAD thr 0.6418 reused: never")

    for info in summaries:
        s = info["scores"]
        print(f"--- {info['building']} ---")
        print(pd.Series(s).describe().to_string())
        print(pd.Series(s).quantile([0.9, 0.95, 0.98, 0.99]).to_string())


if __name__ == "__main__":
    main()
