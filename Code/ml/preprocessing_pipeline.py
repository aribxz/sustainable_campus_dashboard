from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent.parent
DATASETS_DIR = PROJECT_ROOT / "Datasets"
LEAD_TRAIN_FEATURES = DATASETS_DIR / "LEAD" / "train_features.csv"

FEATURES = ["log_meter", "hour_x", "hour_y", "robust_z_168"]
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_samples": 256,
    "max_features": 1.0,
    "contamination": 0.02,
    "random_state": 42,
    "n_jobs": -1,
}
THRESHOLD_PCT = 99
ROLL_WINDOW = 168
ROLL_MIN_PERIODS = 72


def resolve_dataset() -> Path:
    if not LEAD_TRAIN_FEATURES.is_file():
        available = (
            sorted(
                str(p.relative_to(PROJECT_ROOT)) for p in DATASETS_DIR.rglob("*.csv")
            )
            if DATASETS_DIR.is_dir()
            else []
        )
        raise FileNotFoundError(
            f"Dataset not found: {LEAD_TRAIN_FEATURES}\n"
            f"Available CSVs:\n  " + "\n  ".join(available)
        )
    return LEAD_TRAIN_FEATURES


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=[
            "building_id",
            "timestamp",
            "meter_reading",
            "anomaly",
            "hour_x",
            "hour_y",
        ],
    )
    df = df.dropna(subset=["meter_reading"])
    df = df[df["meter_reading"] > 0].reset_index(drop=True)
    return df


def add_portable_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["log_meter"] = np.log1p(df["meter_reading"].clip(lower=0))
    order = df.sort_values(["building_id", "timestamp"]).index
    shifted = df.loc[order].groupby("building_id")["log_meter"].shift(1)
    rolled = shifted.groupby(df.loc[order, "building_id"]).rolling(
        ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS
    )
    med = rolled.median().reset_index(level=0, drop=True)
    iqr = rolled.quantile(0.75).reset_index(level=0, drop=True) - rolled.quantile(
        0.25
    ).reset_index(level=0, drop=True)
    z = (shifted - med) / (iqr + 1e-6)
    df.loc[order, "robust_z_168"] = z.values
    return df


def building_split(df: pd.DataFrame):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    return next(splitter.split(df[FEATURES], df["anomaly"], groups=df["building_id"]))


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> IsolationForest:
    model = IsolationForest(**MODEL_PARAMS)
    model.fit(x_train[y_train == 0])
    return model


def evaluate(model: IsolationForest, x_val: pd.DataFrame, y_val: pd.Series) -> dict:
    scores = -model.score_samples(x_val)
    threshold = np.percentile(scores, THRESHOLD_PCT)
    pred = (scores >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_val, scores),
        "pr_auc": average_precision_score(y_val, scores),
        "precision": precision_score(y_val, pred),
        "recall": recall_score(y_val, pred),
        "f1": f1_score(y_val, pred),
        "threshold": float(threshold),
    }


def main() -> None:
    df = load_data(resolve_dataset())
    df = add_portable_features(df)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    train_idx, val_idx = building_split(df)
    x_train, x_val = df.loc[train_idx, FEATURES], df.loc[val_idx, FEATURES]
    y_train, y_val = df.loc[train_idx, "anomaly"], df.loc[val_idx, "anomaly"]
    print(
        f"Train: {len(x_train)} rows / "
        f"{df.loc[train_idx, 'building_id'].nunique()} buildings | "
        f"Val: {len(x_val)} rows / "
        f"{df.loc[val_idx, 'building_id'].nunique()} buildings"
    )

    model = train_model(x_train, y_train)
    metrics = evaluate(model, x_val, y_val)
    print(f"Features: {FEATURES}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Threshold (p{THRESHOLD_PCT}): {metrics['threshold']:.4f}")


if __name__ == "__main__":
    main()
