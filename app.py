import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from geolocation import detect_impossible_travel, CITY_COORDINATES

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
DATA = ROOT / "data" / "raw" / "creditcard.csv"
st.set_page_config(page_title="Fraud Risk Analyzer", page_icon="🛡️", layout="wide")
st.title("🛡️ Financial Transaction Fraud Risk Analyzer")
st.caption("Model scoring uses real held-out benchmark rows; impossible-travel is a separate geolocation demo.")

risk_tab, geo_tab = st.tabs(["Transaction risk scoring", "Impossible travel"])
with risk_tab:
    st.header("Score a held-out transaction")
    st.info("The ULB CSV anonymizes 28 principal components (V1-V28). Amount and time alone cannot reproduce a model input. This demo therefore scores genuine later-time holdout rows from the CSV instead of inventing missing feature values.")
    pred_path = REPORTS / "holdout_transactions.csv"
    meta_path = REPORTS / "run_metadata.json"
    if not pred_path.exists() or not meta_path.exists():
        st.warning("No trained holdout is available yet. Put creditcard.csv in data/raw/ and run python train.py from the repository root.")
    else:
        preds = pd.read_csv(pred_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        choices = {
            "Random Forest + SMOTE (bagging)": ("rf_smote", "rf_smote.joblib"),
            "Random Forest + class weights (bagging)": ("rf_weighted", "rf_weighted.joblib"),
            "XGBoost + SMOTE (boosting)": ("xgb_smote", "xgb_smote.joblib"),
            "XGBoost + class weights (boosting)": ("xgb_weighted", "xgb_weighted.joblib"),
        }
        model_name = st.selectbox("Model", list(choices))
        score_key, model_file = choices[model_name]
        model_path = MODELS / model_file
        score_col = f"score_{score_key}"
        if not model_path.exists() or score_col not in preds:
            st.warning("Model artifact is missing. Rerun python train.py.")
        else:
            budget_options = sorted(meta["alert_budgets"])
            budget = st.selectbox("Investigator review budget", budget_options, format_func=lambda x: f"{100*x:g}% of holdout")
            budget_row = pd.read_csv(REPORTS / "model_comparison.csv")
            budget_row = budget_row[(budget_row.model == model_name) & (budget_row.alert_budget_pct == 100*budget)]
            cutoff = float(budget_row.iloc[0].score_cutoff) if len(budget_row) else float("inf")
            choice = int(st.number_input("Holdout row number", min_value=0, max_value=len(preds)-1, value=0, step=1))
            row = preds.loc[choice]
            bundle = joblib.load(model_path)
            sample = pd.DataFrame([row[meta["features"]].to_dict()], columns=meta["features"])
            score = float(bundle["model"].predict_proba(sample)[:, 1][0])
            threshold = st.slider("Alert threshold (set from selected review budget)", 0.0, 1.0, min(max(cutoff, 0.0), 1.0), 0.001)
            c1, c2, c3 = st.columns(3)
            c1.metric("Model risk score", f"{score:.2%}")
            c2.metric("Amount", f"{row['Amount']:.2f}")
            c3.metric("Elapsed time", f"{row['Time']:.0f} sec")
            if score >= threshold:
                st.error("Flag for review at this threshold")
            else:
                st.success("Below this review threshold")
            if st.checkbox("Reveal benchmark label for this example"):
                st.write("Dataset label:", "Fraud" if int(row["Class"]) == 1 else "Legitimate")
            with st.expander("Show anonymized feature values"):
                st.dataframe(pd.DataFrame([row[meta["features"]]]), use_container_width=True)
            st.caption("This is a retrospective learning demo. A displayed holdout transaction has a known benchmark label; the model has not seen it during training.")

with geo_tab:
    st.header("Geolocation velocity & anomaly analyzer")
    st.write("This is a separate rule-based demonstration. The ULB CSV contains no city or account fields, so geolocation is not used by the trained model.")
    c1, c2, c3 = st.columns(3)
    with c1:
        last_city = st.selectbox("Previous transaction city", list(CITY_COORDINATES), index=0)
    with c2:
        current_city = st.selectbox("Current transaction city", list(CITY_COORDINATES), index=1)
    with c3:
        hours = st.number_input("Hours since previous transaction", min_value=0.1, max_value=48.0, value=1.0, step=0.5)
    if st.button("Check travel feasibility"):
        result = detect_impossible_travel(last_city, current_city, hours)
        st.write(f"Distance: {result['distance_miles']} miles")
        st.write(f"Required speed: {result['required_speed_mph']} mph")
        if result["is_impossible"]:
            st.error("Impossible travel under the configured speed limit")
        else:
            st.success("Travel is feasible under the configured speed limit")

