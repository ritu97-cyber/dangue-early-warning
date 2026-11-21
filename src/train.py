"""Train LightGBM, tune risk tiers on validation only. Test is used once, with --test."""
import itertools
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, mean_squared_error

from features import FEATURE_COLS, build_dataset, time_split

TIERS = ["Low", "Medium", "High", "Critical"]
LABELS = [0, 1, 2, 3]


def tier_of(values, cuts):
    """0=Low, 1=Medium, 2=High, 3=Critical."""
    return np.digitize(values, cuts)


def tune_cuts(pred, true_tier):
    """Pick 3 cut-offs on VALIDATION predictions that best match the true tiers (macro-F1)."""
    cands = np.unique(np.quantile(pred, np.linspace(0.5, 0.99, 30)))
    best_score, best_cuts = -1.0, None
    for cuts in itertools.combinations(cands, 3):
        score = f1_score(true_tier, tier_of(pred, cuts), labels=LABELS,
                         average="macro", zero_division=0)
        if score > best_score:
            best_score, best_cuts = score, cuts
    return [float(c) for c in best_cuts], float(best_score)


def main(run_test):
    df = build_dataset()
    train, val, test = time_split(df)
    X_tr, y_tr = train[FEATURE_COLS], train["target_cases"]
    X_va, y_va = val[FEATURE_COLS], val["target_cases"]

    model = lgb.LGBMRegressor(
        objective="poisson", n_estimators=1000, learning_rate=0.03,
        num_leaves=15, min_child_samples=50, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False)])

    # True tiers come from TRAIN targets only
    true_cuts = np.quantile(y_tr, [0.70, 0.90, 0.97])
    assert np.all(np.diff(true_cuts) > 0), "true cut-offs not increasing"

    # Tune predicted cut-offs on VALIDATION only
    pred_va = model.predict(X_va)
    pred_cuts, val_f1 = tune_cuts(pred_va, tier_of(y_va, true_cuts))

    val_mae = float(mean_absolute_error(y_va, pred_va))
    base_mae = float(mean_absolute_error(y_va, np.full(len(y_va), y_tr.mean())))
    gain = model.booster_.feature_importance("gain")
    imp = sorted(zip(FEATURE_COLS, gain / gain.sum()), key=lambda x: -x[1])

    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLS}, "models/model.joblib")
    thresholds = {"tiers": TIERS, "true_cuts": [float(c) for c in true_cuts],
                  "pred_cuts": pred_cuts}
    Path("models/thresholds.json").write_text(json.dumps(thresholds, indent=2))

    metrics = {"best_iteration": int(model.best_iteration_),
               "val": {"mae": val_mae, "baseline_mae": base_mae, "macro_f1": val_f1},
               "top_features": [(n, round(float(v), 4)) for n, v in imp[:10]]}

    print("Rows train/val:", len(train), len(val))
    print("True tier cut-offs (train target):", np.round(true_cuts, 1))
    print("Tuned prediction cut-offs (val):", np.round(pred_cuts, 2))
    print(f"VAL  MAE={val_mae:.3f}  (baseline mean-guess MAE={base_mae:.3f})  macro-F1={val_f1:.3f}")
    print("Top features:")
    for n, v in imp[:8]:
        print(f"  {n}: {v:.3f}")

    if run_test:
        X_te, y_te = test[FEATURE_COLS], test["target_cases"]
        pred_te = model.predict(X_te)
        true_te = tier_of(y_te, true_cuts)
        got_te = tier_of(pred_te, pred_cuts)
        cm = confusion_matrix(true_te, got_te, labels=LABELS)
        recall = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
        crit_prec = cm[3, 3] / max(cm[:, 3].sum(), 1)
        metrics["test"] = {
            "mae": float(mean_absolute_error(y_te, pred_te)),
            "rmse": float(np.sqrt(mean_squared_error(y_te, pred_te))),
            "baseline_mae": float(mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean()))),
            "macro_f1": float(f1_score(true_te, got_te, labels=LABELS, average="macro", zero_division=0)),
            "confusion_matrix": cm.tolist(),
            "recall_per_tier": [float(r) for r in recall],
            "critical_precision": float(crit_prec),
        }
        print("\nTEST (final, run once):", json.dumps(metrics["test"], indent=1))
        Path("models/metrics.json").write_text(json.dumps(metrics, indent=2))
    else:
        Path("models/metrics_val.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(run_test="--test" in sys.argv)