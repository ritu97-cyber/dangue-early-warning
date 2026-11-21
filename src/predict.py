"""Prediction helper for the dashboard. Loads the saved model; never retrains."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import STATIC, build_dataset

ROOT = Path(__file__).resolve().parent.parent
TIERS = ["Low", "Medium", "High", "Critical"]


def group_of(col):
    """Map a feature to a human-readable driver group."""
    if col.startswith("rain"):
        return "Recent rainfall"
    if "temp" in col:
        return "Temperature"
    if col.startswith("hum") or col == "relative_humidity_pct":
        return "Humidity"
    if col.startswith("week_"):
        return "Season"
    if col in STATIC:
        return "Ward vulnerability"
    return "Other"


def load_assets():
    """Load model, tier cut-offs and feature table once (cache this in the dashboard)."""
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    thr = json.loads((ROOT / "models" / "thresholds.json").read_text())
    df = build_dataset(ROOT / "data" / "hackathon_preli_dataset.xlsx")
    return {"model": bundle["model"], "features": bundle["features"],
            "cuts": thr["pred_cuts"], "df": df}


def available_dates(assets):
    return sorted(assets["df"]["date"].unique())


def available_wards(assets):
    d = assets["df"][["ward_id", "city"]].drop_duplicates().sort_values("ward_id")
    return d.to_dict("records")


def _tier(pred, cuts):
    return TIERS[int(np.digitize(pred, cuts))]


def _score(assets, rows):
    X = rows[assets["features"]]
    preds = assets["model"].predict(X)
    # per-feature contribution to this prediction (last column = baseline, dropped)
    contrib = assets["model"].booster_.predict(X, pred_contrib=True)[:, :-1]
    return preds, contrib


def _drivers(contrib_row, features):
    totals = {}
    for name, value in zip(features, contrib_row):
        g = group_of(name)
        totals[g] = totals.get(g, 0.0) + float(value)
    total_abs = sum(abs(v) for v in totals.values()) or 1e-9
    out = [{"driver": g, "share_pct": round(100 * abs(v) / total_abs, 1),
            "effect": "raises risk" if v > 0 else "lowers risk"}
           for g, v in totals.items()]
    return sorted(out, key=lambda d: -d["share_pct"])


def predict(ward_id, date, assets):
    """Risk for one ward. Uses the latest data date on or before `date`."""
    df = assets["df"]
    date = pd.Timestamp(date)
    rows = df[(df["ward_id"] == ward_id) & (df["date"] <= date)].tail(1)
    if rows.empty:
        raise ValueError(f"No data for {ward_id} on or before {date.date()}")
    preds, contrib = _score(assets, rows)
    row = rows.iloc[0]
    actual = row["target_cases"]
    return {
        "ward_id": ward_id,
        "city": row["city"],
        "date_used": row["date"].date().isoformat(),
        "window_start": (row["date"] + pd.Timedelta(weeks=2)).date().isoformat(),
        "window_end": (row["date"] + pd.Timedelta(weeks=4)).date().isoformat(),
        "predicted_cases": round(float(preds[0]), 1),
        "tier": _tier(preds[0], assets["cuts"]),
        "drivers": _drivers(contrib[0], assets["features"]),
        "actual_cases": None if pd.isna(actual) else int(actual),
    }


def predict_all(date, assets):
    """Risk for every ward at once (one model call). Used for ranking and heatmap."""
    df = assets["df"]
    date_used = df.loc[df["date"] <= pd.Timestamp(date), "date"].max()
    rows = df[df["date"] == date_used].copy()
    preds, _ = _score(assets, rows)
    rows["predicted_cases"] = np.round(preds, 1)
    rows["tier"] = [_tier(p, assets["cuts"]) for p in preds]
    return rows[["ward_id", "city", "date", "predicted_cases", "tier"]].reset_index(drop=True)


if __name__ == "__main__":
    import time
    t0 = time.time()
    A = load_assets()
    print(f"Load time: {time.time() - t0:.1f}s")
    ward = A["df"]["ward_id"].iloc[0]
    t0 = time.time()
    peak = predict(ward, "2025-08-11", A)
    print(f"One prediction: {(time.time() - t0) * 1000:.0f} ms")
    winter = predict(ward, "2025-01-13", A)
    print("AUG:", peak["tier"], peak["predicted_cases"], "| JAN:", winter["tier"], winter["predicted_cases"])
    print("Drivers (Aug):", peak["drivers"][:3])
    everyone = predict_all("2025-08-11", A)
    print(everyone["tier"].value_counts().to_dict())
    assert peak["tier"] in TIERS and len(everyone) == 60
    assert peak["predicted_cases"] > winter["predicted_cases"], "August should be riskier than January"
    print("OK: predict.py works")