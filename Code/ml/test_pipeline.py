"""Finalized LEAD model -> Kaggle test_features sanity check (unlabeled).

Model (frozen): FEATURES = [log_meter, hour_x, hour_y, robust_z_168],
  IF(200, max_samples=256, max_features=1.0, contamination=0.02, rs=42),
  threshold 0.6418 (p99 from building-disjoint validation).
  Trained on normal (anomaly==0) train rows only.

Key point: test_features.csv has NO robust_z_168 column, and any
Kaggle-side rolling would not match our past-only shift(1)/168/min72
definition anyway. So portable features are recomputed from raw train.csv /
test.csv with one shared function -- train and test stay consistent.
Test set has no labels: only flag counts/rates/distributions are reported,
never precision/recall/AUC.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent.parent
DATASETS_DIR = PROJECT_ROOT / "Datasets"
TRAIN_CSV = DATASETS_DIR / "LEAD" / "train.csv"
TEST_CSV = DATASETS_DIR / "LEAD" / "test.csv"

FEATURES = ["log_meter", "hour_x", "hour_y", "robust_z_168"]
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_samples": 256,
    "max_features": 1.0,
    "contamination": 0.02,
    "random_state": 42,
    "n_jobs": -1,
}
THRESHOLD = 0.6418
ROLL_WINDOW = 168
ROLL_MIN_PERIODS = 72


def add_portable_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build shared portable features: log scale, time, past-only robust z."""
    df = df.copy()
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


def load_train() -> pd.DataFrame:
    """Load raw train CSV and return modeled rows (positive energy + history)."""
    df = pd.read_csv(TRAIN_CSV)
    df = add_portable_features(df)
    df = df.dropna(subset=["meter_reading"])
    df = df[df["meter_reading"] > 0].reset_index(drop=True)
    return df.dropna(subset=FEATURES).reset_index(drop=True)


def load_test() -> pd.DataFrame:
    """Load raw Kaggle test CSV with shared portable features attached."""
    return add_portable_features(pd.read_csv(TEST_CSV))


def train_final_model(train: pd.DataFrame) -> IsolationForest:
    """Fit the frozen Isolation Forest on normal train rows only."""
    normal = train[train["anomaly"] == 0]
    model = IsolationForest(**MODEL_PARAMS)
    model.fit(normal[FEATURES])
    return model


def report_test(test: pd.DataFrame, scores: np.ndarray) -> None:
    """Print the unlabeled generalization report (no labels => no P/R/AUC)."""
    scorable = pd.Series(scores).notna().to_numpy()
    valid_scores = scores[scorable]
    scored = test.loc[scorable].copy()
    flags = valid_scores >= THRESHOLD
    print(f"Test rows: {len(test)} | buildings: {test['building_id'].nunique()}")
    print(
        f"Scorable (energy + 72h history): {int(scorable.sum())} "
        f"({scorable.mean() * 100:.1f}%)",
    )
    print(
        f"Flagged @ thr {THRESHOLD}: {int(flags.sum())} "
        f"({flags.mean() * 100:.2f}% of scorable)",
    )
    print("Score distribution (scorable):")
    print(pd.Series(valid_scores).describe().to_string())
    print("Score percentiles:")
    print(
        pd.Series(valid_scores)
        .quantile([0.5, 0.9, 0.95, 0.98, 0.99, 0.995])
        .to_string(),
    )

    by_building = (
        pd.DataFrame({"building_id": scored["building_id"].to_numpy(), "flag": flags})
        .groupby("building_id")["flag"]
        .agg(["sum", "count"])
    )
    by_building["rate"] = by_building["sum"] / by_building["count"]
    print("Per-building flag rate summary:")
    print(by_building["rate"].describe().to_string())
    print("Top-10 most anomalous buildings (id: flags/rows = rate):")
    top = by_building.sort_values("rate", ascending=False).head(10)
    for building_id, row in top.iterrows():
        print(
            f"  {building_id}: {int(row['sum'])}/{int(row['count'])} "
            f"= {row['rate']:.3f}",
        )
    print(
        "Zero-flag buildings:",
        int((by_building["sum"] == 0).sum()),
        "| >5% flagged:",
        int((by_building["rate"] > 0.05).sum()),
    )

    scored["score"] = valid_scores
    scored["flag"] = flags.astype(int)
    flagged = scored[scored["flag"] == 1]
    print("Flagged by hour:")
    print(flagged["timestamp"].dt.hour.value_counts().sort_index().to_string())
    print("Flagged by month:")
    print(flagged["timestamp"].dt.month.value_counts().sort_index().to_string())
    print("Flagged energy vs scorable energy (kWh/h):")
    print(
        pd.DataFrame(
            {
                "flagged": scored.loc[scored["flag"] == 1, "meter_reading"].describe(),
                "all_scorable": scored["meter_reading"].describe(),
            },
        ).to_string(),
    )


def main() -> None:
    """Train the frozen model and score the Kaggle test set."""
    print(
        "train_features robust_z_168 present: False "
        "(Kaggle table has 57 cols, none ours) -> recompute from raw CSVs.",
    )
    train = load_train()
    print(
        f"Train usable: {len(train)} rows / {train['building_id'].nunique()} buildings",
    )
    test = load_test()
    model = train_final_model(train)
    scorable_mask = (
        test["meter_reading"].notna()
        & (test["meter_reading"] > 0)
        & test[FEATURES].notna().all(axis=1)
    ).to_numpy()
    scores = np.full(len(test), np.nan)
    scores[scorable_mask] = -model.score_samples(test.loc[scorable_mask, FEATURES])
    out = test.copy()
    out["anomaly_score"] = scores
    out["anomaly_flag"] = 0
    out.loc[scorable_mask & (scores >= THRESHOLD), "anomaly_flag"] = 1
    out.to_csv(DATASETS_DIR / "LEAD" / "test_predictions.csv", index=False)
    print("Wrote Datasets/LEAD/test_predictions.csv")
    report_test(test, scores)


if __name__ == "__main__":
    main()
