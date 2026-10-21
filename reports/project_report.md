# Exploratory Analysis and Fraud Detection for Financial Transactions

**Project report**  
**Dataset:** ULB credit-card transactions (`creditcard.csv`)  
**Task:** Explore highly imbalanced transaction data and compare fraud-ranking models  
**Evaluation:** Chronological holdout; results below were generated on 18 October 2026

## Abstract

This project studies transaction-level fraud detection in a severely imbalanced dataset. It combines exploratory data analysis (EDA) with a leakage-aware model comparison. A chronological split separates earlier transactions for training from later transactions for evaluation. Four approaches are compared: Random Forest with SMOTE, Random Forest with class weights, XGBoost with SMOTE, and XGBoost with positive-class weighting. SMOTE is applied only to training data. Models are evaluated by ranking transactions under 0.5% and 1% review budgets, with accuracy, precision, recall, F1, average precision, and ROC-AUC reported. Random Forest produced the strongest ranking on this single holdout. The results also show why accuracy is not sufficient for a fraud task: an all-legitimate baseline is about 99.87% accurate while catching no fraud.

## 1. Problem statement

Financial fraud detection is a rare-event classification problem. A model must identify a small number of fraudulent transactions among many legitimate ones. A model that predicts “legitimate” for every transaction can achieve deceptively high accuracy while providing no fraud detection. The practical question is therefore not just “How many predictions are correct?” but also “How many fraud cases are found, how many false alerts are created, and can an investigation team review them?”

The project asks:

1. What patterns and class imbalance are present in the available transaction data?
2. How do bagging and boosting compare when imbalance is handled using SMOTE or class weighting?
3. How do the models perform when only a small share of transactions can be reviewed?

## 2. Dataset and initial checks

The supplied ULB dataset has 284,807 rows and 31 columns: `Time`, `Amount`, `V1`–`V28`, and `Class`. `Class=1` marks fraud and `Class=0` marks legitimate transactions. `V1`–`V28` are anonymized principal components, so their original business meaning is unavailable. `Time` is elapsed seconds in the dataset, not a calendar date or time-of-day. The file contains no customer ID, account ID, merchant, city, or location.

The raw file contains 284,315 legitimate labels and 492 fraud labels. The fraud share is about 0.173%. The EDA and training pipeline found no missing values. It identified 1,081 exact duplicate rows; the reported training run removes these before splitting, leaving 283,726 rows with 283,253 legitimate and 473 fraud transactions. Exact duplicates can represent repeated records or genuine identical transactions; removing them is a documented preprocessing choice, not proof they are erroneous.

The EDA notebook contains code and charts for class balance, transaction amount, elapsed time, distributions, and correlations. The anonymized feature columns must be interpreted statistically rather than assigned invented business meanings.

## 3. Data preparation and leakage controls

After duplicate removal, data is ordered by `Time`. The first 80% by row position (subject to the split boundary) is used as an earlier-time training set and the later segment as a holdout:

| Partition | Rows | Legitimate | Fraud |
|---|---:|---:|---:|
| Earlier-time training data, before resampling | 226,980 | 226,581 | 399 |
| Later-time holdout, unchanged | 56,746 | 56,672 | 74 |

The boundary is `Time = 145234` seconds. The holdout preserves the original class imbalance and is never oversampled. Feature scaling and SMOTE are placed inside the training pipeline, so they are fitted only using the training partition. The SMOTE target ratio is 0.10, meaning the generated training data has approximately one fraud example for every ten legitimate examples; it does not make the training classes equal.

The workflow also avoids constructing rolling “customer velocity” features because no account identifier exists. Creating one shared fake account would produce meaningless features and could leak information across rows. The app's geolocation demonstration is separate from model training.

## 4. Methods

Two ensemble families are evaluated:

- **Random Forest (bagging):** an ensemble of decision trees. One version uses SMOTE; the other uses balanced class weights.
- **XGBoost (boosting):** trees are added sequentially to improve errors. One version uses SMOTE; the other uses positive-class weighting.

All four models use the same training partition, feature set, random seed, and holdout. Models produce fraud scores. For the alert-budget analysis, transactions are ranked by score and the top 0.5% or 1% are considered alerts. Metrics at these budgets describe that operational ranking decision, not predictions at a universal probability threshold of 0.5.

### Metrics

- **Accuracy:** share of all holdout rows classified correctly under the selected alert list.
- **Precision:** share of reviewed alerts that are fraud.
- **Recall:** share of all holdout fraud caught in the alert list.
- **F1:** harmonic mean of precision and recall.
- **Average precision (AP):** summary of the precision-recall ranking across score thresholds; useful for rare positive classes.
- **ROC-AUC:** ranking discrimination across thresholds; useful as a secondary metric, but can look strong under heavy imbalance.

At each alert budget, `accuracy`, `precision`, `recall`, and `F1` are computed from the top-ranked alert set. AP and ROC-AUC use all continuous holdout scores.

## 5. Results

| Model | Review budget | Accuracy | F1 | Recall | Precision | Fraud caught | False alerts | Missed fraud |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest + SMOTE | 0.5% | 99.59% | 35.75% | 86.49% | 22.54% | 64/74 | 220 | 10 |
| Random Forest + SMOTE | 1.0% | 99.10% | 20.25% | 87.84% | 11.44% | 65/74 | 503 | 9 |
| Random Forest + class weights | 0.5% | 99.59% | 35.75% | 86.49% | 22.54% | 64/74 | 220 | 10 |
| Random Forest + class weights | 1.0% | 99.10% | 20.56% | 89.19% | 11.62% | 66/74 | 502 | 8 |
| XGBoost + SMOTE | 0.5% | 99.58% | 33.52% | 81.08% | 21.13% | 60/74 | 224 | 14 |
| XGBoost + SMOTE | 1.0% | 99.09% | 19.94% | 86.49% | 11.27% | 64/74 | 504 | 10 |
| XGBoost + class weights | 0.5% | 99.58% | 32.96% | 79.73% | 20.77% | 59/74 | 225 | 15 |
| XGBoost + class weights | 1.0% | 99.09% | 19.63% | 85.14% | 11.09% | 63/74 | 505 | 11 |

Average precision was 0.8155 for Random Forest + SMOTE, 0.8150 for Random Forest + class weights, 0.7927 for XGBoost + SMOTE, and 0.7839 for XGBoost + class weights. ROC-AUC was 0.9880, 0.9751, 0.9848, and 0.9869, respectively. The complete run output is saved in `reports/model_comparison.csv` when training is run locally.

## 6. Interpretation and model selection

Random Forest had the strongest average precision in this run, with SMOTE and class weighting producing nearly identical AP. At the 0.5% budget, both Random Forest versions caught 64 of 74 fraud cases with 220 false alerts. At the 1% budget, Random Forest with class weights caught the most fraud, 66 of 74, while creating 502 false alerts. The XGBoost versions caught fewer fraud cases at both budgets in this particular holdout.

There is no single “best” model independent of the review constraint. If the priority is the most detected fraud among a 1% review list, the weighted Random Forest leads in this run. If comparing ranking quality over all score thresholds, Random Forest + SMOTE has the highest AP, by a very small margin over weighted Random Forest.

Accuracy is especially easy to misread. Predicting every holdout transaction as legitimate yields approximately 99.87% accuracy because only 74 of 56,746 holdout rows are fraud. That baseline would catch zero fraud. The models’ lower accuracy at the selected review budgets is partly the consequence of deliberately flagging transactions to catch fraud. Therefore present accuracy together with recall, precision, F1, average precision, and false-alert counts.

## 7. What was newly created

The project contribution is the corrected, reproducible end-to-end workflow:

1. An EDA notebook that checks data quality and visualizes the severe class imbalance and feature distributions.
2. A chronological training/holdout design that keeps future data out of training and preserves the holdout's natural class distribution.
3. Four controlled comparisons across Random Forest and XGBoost, with SMOTE or class weighting; SMOTE is confined to training pipelines.
4. Fraud metrics evaluated at explicit review budgets, so the tradeoff between fraud caught and false alerts is visible.
5. A results notebook, CSV reports, saved model bundles, and an interactive Streamlit demonstration that scores real holdout rows using the actual anonymized feature set.
6. A separately labeled geolocation rule demonstration. The dataset itself has no location fields, so this feature is not presented as a trained-model input.

## 8. Limitations and next steps

- The final holdout contains only 74 fraud examples. One or two changed outcomes can noticeably move the percentages.
- This is one time split and one run configuration. Use rolling temporal validation and tune thresholds only on training/validation data before making broad claims.
- Duplicate removal may discard repeated but genuine transactions. Compare results with and without deduplication as a sensitivity analysis.
- The available features are anonymized and omit important real-world information such as account history, merchants, and location.
- The benchmark does not estimate deployment costs, investigator capacity beyond the chosen alert budgets, model drift, or fairness across customer groups.
- The UI is an educational retrospective demo. It is not a real-time payment service and should not make real financial decisions.

Recommended next step: add rolling time-window validation and report the mean and variability of AP and recall across several future-like splits. Keep a final holdout untouched while choosing models and alert thresholds.

## 9. Short presentation / viva guide

**What is your project?**  
“We explored an imbalanced credit-card transaction dataset and built a leakage-aware comparison of Random Forest and XGBoost, with SMOTE and class-weight approaches. We evaluate how many frauds each model catches at fixed investigation budgets.”

**Why is accuracy not enough?**  
“Fraud is rare. On our holdout, predicting every transaction as legitimate would already be about 99.87% accurate but would detect no fraud. Recall, precision, F1, average precision, and false alerts show the useful tradeoffs.”

**Why did you use SMOTE?**  
“The training set has 399 fraud cases versus 226,581 legitimate cases. SMOTE creates synthetic minority examples inside the training pipeline only. We leave the later-time holdout untouched so the evaluation keeps its natural class balance.”

**Which model did better?**  
“Random Forest ranked fraud better in this run. At a 1% review budget, the class-weighted Random Forest caught 66 of the 74 holdout frauds. This is a result on one split, not a guarantee of future performance.”

**What is the new contribution?**  
“The contribution is a corrected end-to-end project: EDA, chronological evaluation, train-only imbalance handling, a controlled four-model comparison, alert-budget metrics, result notebooks, and a UI that uses the real anonymized feature inputs. The geolocation demo is clearly separate because the CSV has no location data.”

## 10. Reproducibility

See the root `README.md` for Windows/VS Code setup, training commands, notebook instructions, app launch, and safe GitHub upload guidance. The training script writes models to `models/` and machine-readable outputs to `reports/`.
