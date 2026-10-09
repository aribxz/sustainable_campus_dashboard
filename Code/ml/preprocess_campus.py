from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent.parent
IITD_DIR = PROJECT_ROOT / "Datasets" / "IITD"
OUT_DIR = IITD_DIR / "preprocessed"

BUILDINGS = {
    "A Block": IITD_DIR / "transformer_1.csv",
    "B Block": IITD_DIR / "transformer_2.csv",
    "C Block": IITD_DIR / "transformer_3.csv",
    "Mess": IITD_DIR / "mess_build_mains.csv",
    "Hostels": IITD_DIR / "hostels.csv",
}
OUTPUTS = {
    "A Block": OUT_DIR / "transformer1_hourly.csv",
    "B Block": OUT_DIR / "transformer2_hourly.csv",
    "C Block": OUT_DIR / "transformer3_hourly.csv",
    "Mess": OUT_DIR / "mess_hourly.csv",
    "Hostels": OUT_DIR / "hostel_hourly.csv",
}

N_EXPECTED_PER_HOUR = 60
COVERAGE_THRESHOLD = 0.5
SHORT_WINDOW = 24
SHORT_MIN_PERIODS = 12
LONG_WINDOW = 168
LONG_MIN_PERIODS = 72
EPS = 1e-6


COLUMNS = [
    "building",
    "hour_start",
    "n_valid",
    "n_expected",
    "coverage",
    "gap_fraction",
    "is_masked",
    "is_missing",
    "power_mean_W",
    "power_min_W",
    "power_max_W",
    "power_std_W",
    "energy_kwh",
    "energy_kwh_observed",
    "current_mean",
    "current_min",
    "current_max",
    "current_std",
    "n_valid_current",
    "voltage_mean",
    "voltage_min",
    "voltage_max",
    "voltage_std",
    "n_valid_voltage",
    "freq_mean",
    "freq_min",
    "freq_max",
    "freq_std",
    "pf_mean",
    "pf_min",
    "pf_max",
    "pf_std",
    "hour",
    "weekday",
    "is_weekend",
    "month",
    "hour_x",
    "hour_y",
    "month_x",
    "month_y",
    "roll_med_24h",
    "roll_std_24h",
    "roll_med_168h",
    "roll_iqr_168h",
    "dev_24h",
    "robust_z_24h",
    "dev_168h",
    "robust_z_168h",
    "cv_hour",
    "log_energy",
    "robust_z_168",
]


def aggregate_to_hourly(df_min: pd.DataFrame, building: str) -> pd.DataFrame:
    df = df_min.copy()
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("ts").sort_index()
    agg = {"power": ["count", "mean", "min", "max", "std"]}
    for ch in ["current", "voltage", "frequency", "power_factor"]:
        if ch in df.columns:
            agg[ch] = ["count", "mean", "min", "max", "std"]
    hourly = df.resample("1h").agg(agg)
    hourly.columns = ["_".join(c) for c in hourly.columns.values]
    hourly = hourly.reset_index().rename(columns={"ts": "hour_start"})
    hourly["building"] = building
    hourly["n_valid"] = hourly["power_count"].fillna(0).astype(int)
    hourly["n_expected"] = N_EXPECTED_PER_HOUR
    hourly["coverage"] = hourly["n_valid"] / N_EXPECTED_PER_HOUR
    hourly["gap_fraction"] = 1.0 - hourly["coverage"]
    hourly["is_missing"] = (hourly["n_valid"] == 0).astype(int)
    hourly["is_masked"] = (hourly["coverage"] < COVERAGE_THRESHOLD).astype(int)
    ren = {
        "power_mean": "power_mean_W",
        "power_min": "power_min_W",
        "power_max": "power_max_W",
        "power_std": "power_std_W",
        "current_mean": "current_mean",
        "current_min": "current_min",
        "current_max": "current_max",
        "current_std": "current_std",
        "current_count": "n_valid_current",
        "voltage_mean": "voltage_mean",
        "voltage_min": "voltage_min",
        "voltage_max": "voltage_max",
        "voltage_std": "voltage_std",
        "voltage_count": "n_valid_voltage",
        "frequency_mean": "freq_mean",
        "frequency_min": "freq_min",
        "frequency_max": "freq_max",
        "frequency_std": "freq_std",
        "power_factor_mean": "pf_mean",
        "power_factor_min": "pf_min",
        "power_factor_max": "pf_max",
        "power_factor_std": "pf_std",
    }
    return hourly.rename(columns=ren)


def add_energy(hourly: pd.DataFrame) -> pd.DataFrame:
    df = hourly.copy()
    df["energy_kwh"] = df["power_mean_W"] / 1000.0
    df["energy_kwh_observed"] = df["power_mean_W"] * df["n_valid"] / 60.0 / 1000.0
    df.loc[df["is_masked"] == 1, "energy_kwh"] = np.nan
    return df


def reindex_full_grid(hourly: pd.DataFrame, building: str) -> pd.DataFrame:
    df = hourly.copy()
    df["hour_start"] = pd.to_datetime(df["hour_start"], utc=True)
    full = pd.date_range(
        start=df["hour_start"].min().floor("h"),
        end=df["hour_start"].max().ceil("h"),
        freq="1h",
        tz="UTC",
    )
    df = df.set_index("hour_start").reindex(full)
    df.index.name = "hour_start"
    df = df.reset_index()
    df["building"] = building
    df["n_valid"] = df["n_valid"].fillna(0).astype(int)
    df["n_expected"] = N_EXPECTED_PER_HOUR
    df["coverage"] = df["coverage"].fillna(0.0)
    df["gap_fraction"] = 1.0 - df["coverage"]
    df["is_missing"] = (df["n_valid"] == 0).astype(int)
    df["is_masked"] = (df["coverage"] < COVERAGE_THRESHOLD).astype(int)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["hour_start"], utc=True)
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.weekday
    df["is_weekend"] = (ts.dt.weekday >= 5).astype(int)
    df["month"] = ts.dt.month
    df["hour_x"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["hour_y"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["month_x"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_y"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("hour_start").reset_index(drop=True)
    e = np.log1p(df["energy_kwh"].clip(lower=0))
    df["log_energy"] = e
    df["roll_med_24h"] = e.rolling(SHORT_WINDOW, min_periods=SHORT_MIN_PERIODS).median()
    df["roll_std_24h"] = e.rolling(SHORT_WINDOW, min_periods=SHORT_MIN_PERIODS).std()
    df["roll_med_168h"] = e.rolling(LONG_WINDOW, min_periods=LONG_MIN_PERIODS).median()
    q75 = e.rolling(LONG_WINDOW, min_periods=LONG_MIN_PERIODS).quantile(0.75)
    q25 = e.rolling(LONG_WINDOW, min_periods=LONG_MIN_PERIODS).quantile(0.25)
    df["roll_iqr_168h"] = q75 - q25
    df["dev_24h"] = e - df["roll_med_24h"]
    df["robust_z_24h"] = df["dev_24h"] / (df["roll_std_24h"] + EPS)
    df["dev_168h"] = e - df["roll_med_168h"]
    df["robust_z_168h"] = df["dev_168h"] / (df["roll_iqr_168h"] + EPS)

    shifted = e.shift(1)
    pmed = shifted.rolling(LONG_WINDOW, min_periods=LONG_MIN_PERIODS).median()
    piqr = shifted.rolling(LONG_WINDOW, min_periods=LONG_MIN_PERIODS).quantile(
        0.75
    ) - shifted.rolling(LONG_WINDOW, min_periods=LONG_MIN_PERIODS).quantile(0.25)
    df["robust_z_168"] = (e - pmed) / (piqr + EPS)
    if "power_std_W" in df and "power_mean_W" in df:
        df["cv_hour"] = df["power_std_W"] / (df["power_mean_W"].abs() + EPS)
    else:
        df["cv_hour"] = np.nan
    return df


def preprocess_building(input_path: Path, building: str, output_path: Path) -> Path:
    df_min = pd.read_csv(input_path)
    hourly = aggregate_to_hourly(df_min, building)
    hourly = reindex_full_grid(hourly, building)
    hourly = add_energy(hourly)
    hourly = add_time_features(hourly)
    hourly = add_history_features(hourly)
    for col in COLUMNS:
        if col not in hourly.columns:
            hourly[col] = np.nan
    hourly = hourly[COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output_path, index=False)
    ver = (hourly["energy_kwh"] - hourly["power_mean_W"] / 1000.0).abs().max()
    print(
        f"{building}: raw={len(df_min)} -> hourly={len(hourly)} "
        f"usable={int((hourly['is_masked'] == 0).sum())} "
        f"masked={int(hourly['is_masked'].sum())} "
        f"energy_check_maxdiff={ver:.2e} -> {output_path.name}"
    )
    return output_path


def main() -> None:
    for building, src in BUILDINGS.items():
        if not src.is_file():
            print(f"{building}: MISSING {src} -- skipped")
            continue
        preprocess_building(src, building, OUTPUTS[building])


if __name__ == "__main__":
    main()
