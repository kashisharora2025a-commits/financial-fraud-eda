# Training run summary

Trained from the supplied `creditcard.csv` on 2026-10-18. Evaluation is one chronological holdout, not a random split.

- Raw rows: 284,807
- Exact duplicates removed before split: 1,081
- Rows used: 283,726 (283,253 legitimate; 473 fraud)
- Earlier-time training segment: 226,980 rows (226,581 legitimate; 399 fraud)
- Later-time holdout: 56,746 rows (56,672 legitimate; 74 fraud)
- Split boundary: elapsed Time 145,234 seconds; the holdout was never resampled.
- SMOTE was performed only in the training pipeline, at a fraud-to-legitimate ratio of 0.10.

## Results at fixed alert budgets

| Model | Budget | Accuracy | F1 | Recall | Precision | Fraud caught | False alerts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest + SMOTE (bagging) | 0.5% | 99.59% | 35.75% | 86.49% | 22.54% | 64/74 | 220 |
| Random Forest + SMOTE (bagging) | 1.0% | 99.10% | 20.25% | 87.84% | 11.44% | 65/74 | 503 |
| Random Forest + class weights (bagging) | 0.5% | 99.59% | 35.75% | 86.49% | 22.54% | 64/74 | 220 |
| Random Forest + class weights (bagging) | 1.0% | 99.10% | 20.56% | 89.19% | 11.62% | 66/74 | 502 |
| XGBoost + SMOTE (boosting) | 0.5% | 99.58% | 33.52% | 81.08% | 21.13% | 60/74 | 224 |
| XGBoost + SMOTE (boosting) | 1.0% | 99.09% | 19.94% | 86.49% | 11.27% | 64/74 | 504 |
| XGBoost + class weights (boosting) | 0.5% | 99.58% | 32.96% | 79.73% | 20.77% | 59/74 | 225 |
| XGBoost + class weights (boosting) | 1.0% | 99.09% | 19.63% | 85.14% | 11.09% | 63/74 | 505 |

## Interpretation

At the 0.5% budget, both Random Forest variants caught 64 of 74 holdout frauds. At 1%, Random Forest with class weights caught the most (66), but precision fell to 11.62% because investigators review 568 alerts. The SMOTE and class-weight Random Forests have nearly identical average precision (0.8155 and 0.8150). In this run, XGBoost ranked below the Random Forest variants by average precision.

Accuracy is not useful alone here: predicting every holdout row as legitimate would be about 99.87% accurate while catching no fraud. The F1/precision tradeoff also shows that most flagged alerts are false positives at these review budgets.

These results cover only 74 fraud examples in one time-based holdout. They are evidence for this benchmark split, not a proof of generalization, not proof that a model is overfit, and not a deployment recommendation. Use rolling temporal validation before making a strong model-selection claim. Exact duplicate removal is documented as a choice; repeated identical events may sometimes be genuine.
