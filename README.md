# Financial Transaction Fraud and Anomaly Detection

An educational project that explores the ULB credit-card transaction dataset and compares four imbalance-aware classifiers. Exploratory and model-result charts live in Jupyter notebooks.

## Start here

1. Open this project folder in VS Code: **File → Open Folder**. Open the folder that contains `app.py`, `train.py`, and `requirements.txt`.
2. Put your own copy of `creditcard.csv` at `data/raw/creditcard.csv`. The CSV is intentionally excluded from GitHub.
3. In VS Code choose **Terminal → New Terminal** and run these commands from the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py --data data/raw/creditcard.csv
```

If PowerShell blocks activation, use the venv interpreter directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe train.py --data data/raw/creditcard.csv
```

The training command creates the four model files in `models/` and evaluation outputs in `reports/`. It does not change the source CSV. In VS Code, open `notebooks/01_exploratory_data_analysis.ipynb`, select the `.venv` kernel, then run cells from top to bottom. Do the same for `notebooks/02_model_results.ipynb` after training.

To start the interactive demo after training:

```powershell
streamlit run app.py
```

## What the project compares

The dataset has `Time`, `Amount`, anonymized principal components `V1`–`V28`, and the binary target `Class` (0 = legitimate, 1 = fraud). The project compares:

| Model | Imbalance method | Ensemble type |
|---|---|---|
| Random Forest | SMOTE | Bagging |
| Random Forest | Class weights | Bagging |
| XGBoost | SMOTE | Boosting |
| XGBoost | Positive-class weighting | Boosting |

SMOTE is applied inside each model pipeline to training data only. The later-time holdout is never oversampled. A chronological split makes the task more realistic than randomly mixing earlier and later records. Exact duplicate records are removed before splitting; the run summary records the effect of that choice.

## Results from the supplied dataset

The actual run used 283,726 unique rows after removing 1,081 exact duplicates. It trained on 226,980 earlier-time rows (399 fraud) and evaluated on 56,746 later-time rows (74 fraud). At a review budget, the model ranks holdout transactions by score and the top 0.5% or 1% are treated as alerts.

| Model | Review budget | Accuracy | F1 | Recall | Precision | Fraud caught | False alerts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest + SMOTE | 0.5% | 99.59% | 35.75% | 86.49% | 22.54% | 64/74 | 220 |
| Random Forest + SMOTE | 1.0% | 99.10% | 20.25% | 87.84% | 11.44% | 65/74 | 503 |
| Random Forest + class weights | 0.5% | 99.59% | 35.75% | 86.49% | 22.54% | 64/74 | 220 |
| Random Forest + class weights | 1.0% | 99.10% | 20.56% | 89.19% | 11.62% | 66/74 | 502 |
| XGBoost + SMOTE | 0.5% | 99.58% | 33.52% | 81.08% | 21.13% | 60/74 | 224 |
| XGBoost + SMOTE | 1.0% | 99.09% | 19.94% | 86.49% | 11.27% | 64/74 | 504 |
| XGBoost + class weights | 0.5% | 99.58% | 32.96% | 79.73% | 20.77% | 59/74 | 225 |
| XGBoost + class weights | 1.0% | 99.09% | 19.63% | 85.14% | 11.09% | 63/74 | 505 |

**Which performed best?** Random Forest ranked fraud best overall on this one holdout (average precision: 0.8155 with SMOTE and 0.8150 with class weights). At a 1% review budget, Random Forest with class weights caught the most fraud: 66 of 74. At 0.5%, both Random Forest variants caught 64.

**explaining accuracy:** the holdout is highly imbalanced. Predicting every transaction as legitimate would be about 99.87% accurate while detecting zero fraud. Therefore accuracy alone is misleading; we discuss recall, precision, F1, average precision, and the number of alerts that a reviewer can handle.

These results are from one later-time holdout containing 74 fraud transactions. They are a benchmark result, not proof of production readiness or guaranteed future performance. Rolling temporal validation would provide stronger evidence.

## Project files

| Path | Purpose |
|---|---|
| `train.py` | Duplicate handling, chronological split, four model training runs, and saved evaluation outputs |
| `notebooks/01_exploratory_data_analysis.ipynb` | Dataset checks, class imbalance, distributions, and EDA charts |
| `notebooks/02_model_results.ipynb` | Metrics, model comparison, and result charts |
`reports/project_report.md` | Project write-up, methods, results, interpretation, limitations, and viva notes |
| `reports/training_summary.md` | Summary of the supplied-data training run |
| `reports/smote_training_data.csv` | Optional inspection export of resampled training rows; not needed to train or run the app |
| `models/*.joblib` | Four fitted model bundles created by training; regenerate with `train.py` if absent |

## About the SMOTE inspection CSV

`reports/smote_training_data.csv` was exported separately for inspection. It contains the earlier-time training partition only, after SMOTE at a 0.10 fraud-to-legitimate ratio. It has 249,239 rows: 226,581 legitimate and 22,658 fraud, of which 22,259 fraud rows are synthetic. `is_synthetic` marks generated rows. It is not the original dataset, not the holdout, and should not be fed back as the input to `train.py`.

SMOTE in the actual training workflow runs on scaled training features inside the pipeline. The export inverse-transforms those rows back to the dataset's feature scales for readability. Small floating-point differences can appear in the CSV.

## Dataset limits and responsible interpretation

- The ULB CSV contains no customer/account ID, city, merchant, or calendar timestamp. `Time` is elapsed seconds. The project does not invent account velocity features or claim geography is part of model training.
- The geolocation feature is a separate rule-based UI demonstration using manually selected cities; it is not a trained model feature and is not linked to CSV records.
- This is an educational retrospective benchmark. We do not use it to make real financial decisions.
