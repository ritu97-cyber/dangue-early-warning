"""Ward-level lag features. Every feature at week t uses data from week t or earlier only."""
import numpy as np
import pandas as pd

DATA_PATH = "data/hackathon_preli_dataset.xlsx"
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_END = pd.Timestamp("2023-12-31")
PURGE_WEEKS = 4  # target looks up to 4 weeks ahead

STATIC = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
RAIN_LAGS = [1, 2, 3, 4, 5, 6]


def load_data(path=DATA_PATH):
    df = pd.read_excel(path)
    return df.sort_values(["ward_id", "date"]).reset_index(drop=True)


def add_features(df):
    df = df.sort_values(["ward_id", "date"]).copy()
    g = df.groupby("ward_id")
    for k in RAIN_LAGS:
        df[f"rain_lag{k}"] = g["rainfall_mm"].shift(k)
    for w in (2, 4, 8):  # rolling sums include current week (past only)
        df[f"rain_sum{w}w"] = g["rainfall_mm"].transform(lambda s: s.rolling(w).sum())
    for k in (2, 4):
        df[f"temp_lag{k}"] = g["avg_temp_c"].shift(k)
        df[f"hum_lag{k}"] = g["relative_humidity_pct"].shift(k)
    df["temp_mean4w"] = g["avg_temp_c"].transform(lambda s: s.rolling(4).mean())
    df["hum_mean4w"] = g["relative_humidity_pct"].transform(lambda s: s.rolling(4).mean())
    week = df["date"].dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * week / 52)
    df["week_cos"] = np.cos(2 * np.pi * week / 52)
    return df


FEATURE_COLS = (
    ["rainfall_mm", "avg_temp_c", "relative_humidity_pct"]
    + [f"rain_lag{k}" for k in RAIN_LAGS]
    + [f"rain_sum{w}w" for w in (2, 4, 8)]
    + ["temp_lag2", "temp_lag4", "hum_lag2", "hum_lag4", "temp_mean4w", "hum_mean4w"]
    + ["week_sin", "week_cos"]
    + STATIC
)


def add_target(df):
    """target_cases = cases in weeks t+2, t+3, t+4 (NaN if window incomplete)."""
    g = df.groupby("ward_id")["dengue_cases"]
    df["target_cases"] = g.shift(-2) + g.shift(-3) + g.shift(-4)
    return df


def build_dataset(path=DATA_PATH):
    df = add_target(add_features(load_data(path)))
    return df.dropna(subset=FEATURE_COLS).reset_index(drop=True)


def time_split(df):
    """Chronological split with purging so no target window crosses a boundary."""
    gap = pd.Timedelta(weeks=PURGE_WEEKS)
    d = df.dropna(subset=["target_cases"])
    train = d[d["date"] <= TRAIN_END - gap]
    val = d[(d["date"] > TRAIN_END) & (d["date"] <= VAL_END - gap)]
    test = d[d["date"] > VAL_END]
    return train, val, test


if __name__ == "__main__":
    df = build_dataset()
    train, val, test = time_split(df)
    print("rows:", len(df), "| train/val/test:", len(train), len(val), len(test))
    # Check 1: lag really equals rainfall from k rows earlier (same ward)
    w = df[df["ward_id"] == df["ward_id"].iloc[0]].reset_index(drop=True)
    assert w.loc[20, "rain_lag3"] == w.loc[17, "rainfall_mm"], "lag is wrong"
    # Check 2: no target window crosses a split boundary
    assert train["date"].max() + pd.Timedelta(weeks=PURGE_WEEKS) <= TRAIN_END
    assert val["date"].max() + pd.Timedelta(weeks=PURGE_WEEKS) <= VAL_END
    assert val["date"].min() > TRAIN_END and test["date"].min() > VAL_END
    print("OK: lags and purged split verified")