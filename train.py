"""Leakage-aware training benchmark for the ULB credit-card dataset.

Run from the repository root:
  python train.py --data data/raw/creditcard.csv
"""
import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "raw" / "creditcard.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def evaluate_budget(y, scores, name, budgets):
    order = np.argsort(-scores, kind="stable")
    n = len(y)
    fraud_total = int((y == 1).sum())
    ap = float(average_precision_score(y, scores))
    auc = float(roc_auc_score(y, scores))
    rows = []
    for fraction in budgets:
        k = min(n, max(1, math.ceil(n * fraction)))
        selected = order[:k]
        caught = int((y[selected] == 1).sum())
        false_alerts = k - caught
        missed_fraud = fraud_total - caught
        true_negatives = int((y == 0).sum()) - false_alerts
        accuracy = (caught + true_negatives) / n
        f1 = (2 * caught / (2 * caught + false_alerts + missed_fraud)
              if (2 * caught + false_alerts + missed_fraud) else 0.0)
        rows.append({
            "model": name, "alert_budget_pct": 100 * fraction,
            "alerts_reviewed": k, "holdout_transactions": n,
            "holdout_fraud": fraud_total, "fraud_caught": caught,
            "recall_at_budget_pct": 100 * caught / fraud_total if fraud_total else 0,
            "precision_at_budget_pct": 100 * caught / k,
            "accuracy_pct": 100 * accuracy, "f1_at_budget_pct": 100 * f1,
            "false_alerts": false_alerts, "missed_fraud": missed_fraud,
            "average_precision": ap, "roc_auc": auc,
            "score_cutoff": float(scores[order[k - 1]]),
        })
    return rows


def make_models(y_train, smote_ratio):
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    smote = SMOTE(sampling_strategy=smote_ratio, k_neighbors=5, random_state=42)
    def rf(use_smote):
        steps = [("scale", StandardScaler())]
        if use_smote:
            steps.append(("smote", SMOTE(sampling_strategy=smote_ratio, k_neighbors=5, random_state=42)))
        steps.append(("model", RandomForestClassifier(
            n_estimators=350, max_features="sqrt", min_samples_leaf=2,
            class_weight=None if use_smote else "balanced_subsample",
            n_jobs=-1, random_state=42,
        )))
        return Pipeline(steps)
    def xgb(use_smote):
        steps = [("scale", StandardScaler())]
        if use_smote:
            steps.append(("smote", SMOTE(sampling_strategy=smote_ratio, k_neighbors=5, random_state=42)))
        steps.append(("model", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
            objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
            scale_pos_weight=1.0 if use_smote else neg / pos,
            n_jobs=-1, random_state=42,
        )))
        return Pipeline(steps)
    return {
        "Random Forest + SMOTE (bagging)": rf(True),
        "Random Forest + class weights (bagging)": rf(False),
        "XGBoost + SMOTE (boosting)": xgb(True),
        "XGBoost + class weights (boosting)": xgb(False),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare bagging and boosting with SMOTE or class weights.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--smote-ratio", type=float, default=0.10)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.005, 0.01])
    args = parser.parse_args()
    if not 0 < args.test_fraction < 0.5 or not 0 < args.smote_ratio <= 1 or any(not 0 < b <= 1 for b in args.budgets):
        parser.error("Check fractions: test (0,.5), SMOTE ratio (0,1], budgets (0,1].")
    path = Path(args.data)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        parser.error(f"Dataset not found: {path}. Put creditcard.csv in data/raw/.")

    raw = pd.read_csv(path)
    needed = {"Time", "Amount", "Class"}
    if not needed.issubset(raw.columns):
        parser.error("CSV needs Time, Amount, Class and preferably the V1-V28 columns.")
    if raw.isna().any().any():
        parser.error("Missing values found. Inspect them in the EDA notebook before training; no silent imputation is applied.")
    if not set(raw["Class"].unique()).issubset({0, 1, 0.0, 1.0}):
        parser.error("Class must be encoded as 0/1.")
    features = [c for c in raw.columns if c != "Class"]
    if not all(pd.api.types.is_numeric_dtype(raw[c]) for c in features):
        parser.error("This benchmark expects numeric predictors.")

    n_duplicates = int(raw.duplicated().sum())
    data = raw.drop_duplicates().sort_values("Time", kind="stable").reset_index(drop=True)
    cut_position = int(len(data) * (1 - args.test_fraction))
    boundary_time = float(data.iloc[cut_position]["Time"])
    train = data[data["Time"] < boundary_time].copy()
    test = data[data["Time"] >= boundary_time].copy()
    if train.empty or test.empty or train["Class"].nunique() != 2 or test["Class"].nunique() != 2:
        parser.error("Temporal split must contain both classes on each side; inspect Time/class counts.")

    X_train, y_train = train[features], train["Class"].astype(int).to_numpy()
    X_test, y_test = test[features], test["Class"].astype(int).to_numpy()
    if int((y_train == 1).sum()) <= 5:
        parser.error("SMOTE k_neighbors=5 requires more than five frauds in training.")
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    result_rows = []
    score_columns = {}
    model_files = {}
    for i, (name, model) in enumerate(make_models(y_train, args.smote_ratio).items()):
        print(f"Fitting {name} ...", flush=True)
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        result_rows.extend(evaluate_budget(y_test, scores, name, args.budgets))
        score_key = ["rf_smote", "rf_weighted", "xgb_smote", "xgb_weighted"][i]
        score_columns[score_key] = scores
        filename = f"{score_key}.joblib"
        joblib.dump({"model": model, "features": features, "name": name}, MODELS_DIR / filename)
        model_files[name] = filename
        if score_key == "rf_smote":
            fi = model.named_steps["model"].feature_importances_
            pd.DataFrame({"feature": features, "importance": fi}).sort_values("importance", ascending=False).to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    metrics = pd.DataFrame(result_rows)
    metrics.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    prediction_frame = test.reset_index(names="holdout_id").copy()
    for key, scores in score_columns.items():
        prediction_frame[f"score_{key}"] = scores
    prediction_frame.to_csv(REPORTS_DIR / "holdout_transactions.csv", index=False)
    class_counts = {str(int(k)): int(v) for k, v in data["Class"].value_counts().sort_index().items()}
    metadata = {
        "source_file": path.name, "raw_rows": int(len(raw),),
        "exact_duplicates_removed_before_split": n_duplicates,
        "rows_after_deduplication": int(len(data)), "class_counts_after_deduplication": class_counts,
        "train_rows_earlier_time": int(len(train)), "holdout_rows_later_time": int(len(test)),
        "train_class_counts_before_resampling": {str(int(k)): int(v) for k, v in train["Class"].value_counts().sort_index().items()},
        "holdout_class_counts_untouched": {str(int(k)): int(v) for k, v in test["Class"].value_counts().sort_index().items()},
        "time_boundary": boundary_time, "smote_ratio_in_training": args.smote_ratio,
        "alert_budgets": args.budgets, "features": features, "model_files": model_files,
        "package_versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost", "joblib"]},
        "note": "SMOTE pipelines resample training rows only. The later-time holdout remains untouched. Time is elapsed time, not calendar time. No account ID exists, so no account velocity features are created.",
    }
    (REPORTS_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nTraining finished. Results: reports/model_comparison.csv")
    print(f"Raw rows={len(raw):,}; duplicates removed={n_duplicates:,}; used={len(data):,}")
    print("Train labels before resampling:", metadata["train_class_counts_before_resampling"])
    print("Untouched holdout labels:", metadata["holdout_class_counts_untouched"])
    print(metrics[["model", "alert_budget_pct", "fraud_caught", "accuracy_pct", "precision_at_budget_pct", "recall_at_budget_pct", "f1_at_budget_pct", "false_alerts", "average_precision", "roc_auc"]].to_string(index=False))

if __name__ == "__main__":
    main()
