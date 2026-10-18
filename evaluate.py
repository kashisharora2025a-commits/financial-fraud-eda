"""Reprint metrics from the saved untouched holdout predictions."""
import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent
metrics_path = ROOT / "reports" / "model_comparison.csv"
metadata_path = ROOT / "reports" / "run_metadata.json"
if not metrics_path.exists():
    raise SystemExit("Run python train.py first; no holdout metrics found.")
metrics = pd.read_csv(metrics_path)
meta = json.loads(metadata_path.read_text(encoding="utf-8"))
print("Temporal holdout evaluation (not training-set performance)")
print("Holdout counts:", meta["holdout_class_counts_untouched"])
print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nFor charts and detailed analysis, open notebooks/02_model_results.ipynb")
