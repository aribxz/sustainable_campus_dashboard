"""Score IITD Library with the frozen finalized LEAD model (no retraining/tuning).

Frozen model: FEATURES = [log_meter, hour_x, hour_y, robust_z_168],
  IsolationForest(200, max_samples=256, max_features=1.0,
                  contamination=0.02, random_state=42).
  Trained on LEAD train.csv normal (anomaly==0) rows only.

Frozen feature definitions (Library's own series, past-only):
  log_meter    = log1p(energy_kwh)
  hour_x/y     = cos/sin(2*pi*hour/24) from hour_start
  robust_z_168 = (log_meter - median_168) / IQR_168, rolling 168h,
                 min_periods=72, shift(1) so the current hour is NOT in
                 its own baseline.

Input (already hourly, never resampled): Datasets/IITD/preprocessed/library_hourly.csv
  consumption column = energy_kwh (== power_mean_W/1000, NaN when masked).
The old library_hourly_scored.csv scores/flags are inspected for
diagnostics only and never reused.
Output: Datasets/IITD/scored/library_final_scored.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent.parent
DATASETS_DIR = PROJECT_ROOT / "Datasets"
LEAD_TRAIN = DATASETS_DIR / "LEAD" / "train.csv"
LIB_HOURLY = DATASETS_DIR / "IITD" / "preprocessed" / "library_hourly.csv"
LIB_SCORED_OLD = Path(r"C:\Users\ASUS\Downloads\CCH_archive\library_hourly_scored.csv")
LIB_FINAL = DATASETS_DIR / "IITD" / "scored" / "library_final_scored.csv"

FEATURES = ["log_meter", "hour_x", "hour_y", "robust_z_168"]
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_samples": 256,
    "max_features": 1.0,
    "contamination": 0.02,
    "random_state": 42,
    "n_jobs": -1,
}
ROLL_WINDOW = 168
ROLL_MIN_PERIODS = 72


def build_lead_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Frozen LEAD feature build on raw (building_id, timestamp, meter)."""
    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["log_meter"] = np.log1p(df["meter_reading"].clip(lower=0))
    hour = df["timestamp"].dt.hour
    df["hour_x"] = np.cos(2 * np.pi * hour / 24.0)
    df["hour_y"] = np.sin(2 * np.pi * hour / 24.0)
    order = df.sort_values(["building_id", "timestamp"]).index
    shifted = df.loc[order].groupby("building_id")["log_meter"].shift(1)
    rolled = shifted.groupby(df.loc[order, "building_id"]).rolling(
        ROLL_WINDOW,
        min_periods=ROLL_MIN_PERIODS,
    )
    med = rolled.median().reset_index(level=0, drop=True)
    iqr = rolled.quantile(0.75).reset_index(level=0, drop=True) - rolled.quantile(
        0.25,
    ).reset_index(level=0, drop=True)
    df.loc[order, "robust_z_168"] = ((shifted - med) / (iqr + 1e-6)).to_numpy()
    return df


def train_frozen_model() -> IsolationForest:
    """Instantiate the finalized model: same params, same LEAD normals."""
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


def build_library_features() -> pd.DataFrame:
    """Hourly Library frame + freshly recomputed frozen features."""
    df = pd.read_csv(LIB_HOURLY)
    df["hour_start"] = pd.to_datetime(df["hour_start"], utc=True)
    df = df.sort_values("hour_start").reset_index(drop=True)

    df["log_meter"] = np.log1p(df["energy_kwh"].clip(lower=0))
    hour = df["hour_start"].dt.hour
    df["hour_x"] = np.cos(2 * np.pi * hour / 24.0)
    df["hour_y"] = np.sin(2 * np.pi * hour / 24.0)
    shifted = df["log_meter"].shift(1)
    med = shifted.rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).median()
    iqr = shifted.rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).quantile(
        0.75
    ) - shifted.rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).quantile(0.25)
    df["robust_z_168"] = (df["log_meter"] - med) / (iqr + 1e-6)
    return df


def main() -> None:

    if LIB_SCORED_OLD.is_file():
        old = pd.read_csv(LIB_SCORED_OLD, usecols=["anomaly_score", "anomaly_flag_p98"])
        print(
            f"Old scored file (diagnostic, NOT reused): "
            f"{len(old)} rows, p98 flags={int(old['anomaly_flag_p98'].sum())}"
        )
    model = train_frozen_model()
    lib = build_library_features()
    scorable = (
        lib["energy_kwh"].notna()
        & (lib["energy_kwh"] > 0)
        & lib[FEATURES].notna().all(axis=1)
    ).to_numpy()
    print(
        f"Library rows: {len(lib)} | masked: {int(lib['is_masked'].sum())} | "
        f"scorable (energy + 72h history): {int(scorable.sum())}"
    )
    scores = np.full(len(lib), np.nan)
    scores[scorable] = -model.score_samples(lib.loc[scorable, FEATURES])
    lib["anomaly_score"] = scores
    thr98 = float(np.nanquantile(scores, 0.98))
    thr99 = float(np.nanquantile(scores, 0.99))
    lib["anomaly_flag_p98"] = 0
    lib["anomaly_flag_p99"] = 0
    lib.loc[scorable & (scores >= thr98), "anomaly_flag_p98"] = 1
    lib.loc[scorable & (scores >= thr99), "anomaly_flag_p99"] = 1

    out = lib[
        [
            "hour_start",
            "energy_kwh",
            "power_mean_W",
            "power_min_W",
            "power_max_W",
            "coverage",
            "is_masked",
            "is_missing",
            "log_meter",
            "hour_x",
            "hour_y",
            "robust_z_168",
            "anomaly_score",
            "anomaly_flag_p98",
            "anomaly_flag_p99",
        ]
    ].copy()
    out = out.rename(columns={"hour_start": "timestamp"})
    out.to_csv(LIB_FINAL, index=False)
    print(f"Wrote {LIB_FINAL}")

    s = scores[scorable]
    n98 = int(lib["anomaly_flag_p98"].sum())
    n99 = int(lib["anomaly_flag_p99"].sum())
    print("Library score distribution (scorable):")
    print(pd.Series(s).describe().to_string())
    print(
        f"p98 thr={thr98:.4f} -> {n98} rows ({n98 / scorable.sum() * 100:.2f}%); "
        f"p99 thr={thr99:.4f} -> {n99} rows ({n99 / scorable.sum() * 100:.2f}%)"
    )
    top20 = lib.loc[scorable].nlargest(20, "anomaly_score")[
        ["hour_start", "energy_kwh", "robust_z_168", "anomaly_score"]
    ]
    print("Top-20 most anomalous Library hours:")
    print(top20.to_string(index=False))
    fl = lib[lib["anomaly_flag_p98"] == 1]
    print("Flagged (p98) energy vs scorable energy:")
    print(
        pd.DataFrame(
            {
                "flagged": fl["energy_kwh"].describe(),
                "scorable": lib.loc[scorable, "energy_kwh"].describe(),
            }
        ).to_string()
    )
    print("Flagged (p98) by hour:")
    print(fl["hour_start"].dt.hour.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
