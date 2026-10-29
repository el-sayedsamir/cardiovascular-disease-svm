# Cardiovascular Disease Prediction with SVM

End-to-end machine-learning pipeline that predicts the presence of cardiovascular disease from 11 routine patient measurements, using a Support Vector Machine (scikit-learn).

The goal is **not** just a high accuracy number: the project focuses on reproducible results, leakage-free evaluation, and documenting every decision.

> **Disclaimer:** educational project. The model is not a medical device and must not be used for diagnosis.

## Results at a glance

| Metric (held-out test set, 13,713 patients) | Value |
|---|---|
| Accuracy | **0.725** |
| Precision | 0.716 |
| Recall (sensitivity) | **0.735** |
| F1-score | 0.726 |
| ROC-AUC | 0.793 |

- Decision threshold set to **-0.3** (instead of 0) to raise recall from 0.668 to 0.735, at a cost of only 0.5 points of accuracy. Missing a sick patient is costlier than a false alarm.
- Test metrics match the 5-fold cross-validation estimate on the training set to within 0.005, which indicates no leakage and no overfitting.

## Dataset

[Cardiovascular Disease dataset (Kaggle, Svetlana Ulianova)](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset): 70,000 patients, 11 features, binary target `cardio`. Classes are balanced (about 50 / 50).

| Feature | Description |
|---|---|
| `age` | age in days |
| `gender` | 1 = female, 2 = male |
| `height` (cm), `weight` (kg) | body measurements |
| `ap_hi`, `ap_lo` | systolic / diastolic blood pressure |
| `cholesterol`, `gluc` | 1 normal, 2 above normal, 3 well above normal |
| `smoke`, `alco`, `active` | lifestyle (0/1) |
| `cardio` | **target**: cardiovascular disease (0/1) |

## Pipeline and key decisions

1. **Data inspection**: no missing values, no fully duplicated rows. Found impossible values (negative blood pressure, `ap_hi` up to 16,020, height 55 cm, weight 10 kg).
2. **Data cleaning** (70,000 → 68,561 rows, 2.06% removed). Rows were dropped, not "fixed", because the loss is small and fixing would require guessing the intended value.

   | Rule | Rows dropped |
   |---|---|
   | Duplicates (ignoring `id`) | 24 |
   | Non-positive blood pressure | 29 |
   | `ap_hi` outside 70-250 | 212 |
   | `ap_lo` outside 40-150 | 1,005 |
   | `ap_lo` >= `ap_hi` | 88 |
   | Height outside 120-220 cm | 50 |
   | Weight < 30 kg | 6 |
   | BMI outside 12-60 | 25 |

3. **EDA**: distributions by target, cardio rate per category, correlation matrix, IQR outliers.
4. **Feature engineering** (deterministic, row-wise, no leakage): `age_years`, `gender_bin`, `bmi`, `bmi_cat`, `pulse_pressure`, `map` (mean arterial pressure), `bp_cat` (AHA categories), `risk_habits`.
5. **Train/test split**: 80/20, stratified, `random_state=42`. The test set is untouched until the final evaluation.
6. **Preprocessing**: `StandardScaler` on continuous + ordinal features, binary features passed through, all inside a scikit-learn `Pipeline` so it is fitted on training folds only.
7. **Feature selection** by group ablation (SVM, 5-fold CV on a 20k stratified subsample, paired comparison on the same folds):

   | Dropped group | Change in ROC-AUC |
   |---|---|
   | demographics (age, gender) | -0.0189 |
   | labs (cholesterol, glucose) | -0.0098 |
   | body (height, weight, BMI) | -0.0040 |
   | lifestyle | -0.0030 |
   | raw blood pressure | -0.0001 |
   | **derived blood pressure** (`pulse_pressure`, `map`, `bp_cat`) | **+0.0045** |
   | all blood pressure (control) | -0.0962 |

   The derived blood-pressure features are redundant with `ap_hi`/`ap_lo`; removing them improved AUC consistently. **Final model uses 14 features.** The control experiment (no blood pressure at all) confirms the method works: blood pressure is the most informative signal.
8. **SVM baseline**: RBF kernel, default `C=1`, `gamma='scale'`.
9. **Tuning**: `RandomizedSearchCV` (20 candidates, 3-fold, ROC-AUC, 15k subsample) over RBF and linear kernels. Selected **RBF, C = 1.84, gamma = 0.0083** (top 5 candidates were statistically tied; the simpler one was chosen).
10. **Threshold selection** on out-of-fold training scores (not on the test set).

## Before vs after tuning (5-fold CV on the training set)

| Metric | Baseline | Tuned | Change |
|---|---|---|---|
| ROC-AUC | 0.7901 | 0.7935 | **+0.0034** (consistent across folds) |
| Accuracy | 0.7341 | 0.7320 | -0.0020 (within noise) |
| Precision | 0.7621 | 0.7599 | -0.0023 (within noise) |
| Recall | 0.6724 | 0.6702 | -0.0022 (within noise) |
| F1 | 0.7144 | 0.7122 | -0.0022 (within noise) |

## What worked and what did not

**Worked**
- Leakage-free design: split first, all fitted steps inside a `Pipeline`, test set used once.
- Documented cleaning rules removed clearly impossible values with a 2% loss.
- Feature-group ablation identified redundant features and confirmed blood pressure as the key signal.
- Threshold tuning gave the only meaningful practical gain (recall 0.67 → 0.74).

**Did not work / limits**
- Hyper-parameter tuning gave only a marginal ROC-AUC gain and no gain in accuracy. Baseline and tuned models show almost no train/validation gap, so the model is limited by the information in the features, not by its settings. A linear kernel scored almost the same as RBF, so the data is nearly linearly separable.
- Accuracy plateaus around 73%, a known ceiling for this dataset. Several values (e.g., blood pressure) are likely entry errors, and the features do not include strong clinical markers.

**Possible next steps**
- Try to correct blood-pressure entries with an extra zero (e.g., 1100 → 110) instead of dropping them, and compare.
- Compare against gradient boosting and logistic regression baselines.
- Calibrate probabilities (e.g., `CalibratedClassifierCV`).

## Repository structure

```
├── notebook.ipynb              # full workflow, cell by cell
├── requirements.txt            # pinned library versions
├── data/
│   └── Cardiovascular_Disease.csv   # raw dataset
├── cardio_clean.csv            # cleaned data + engineered features
├── cardio_train.csv            # training split (14 selected features + target)
├── cardio_test.csv             # test split
├── cardio_svm_final.joblib     # trained pipeline + threshold + feature list
├── LICENSE
└── README.md
```

## How to run

```bash
git clone https://github.com/el-sayedsamir/cardiovascular-disease-svm.git
cd cardiovascular-disease-svm
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

- Tested with Python 3.13.7 and the library versions in `requirements.txt`.
- All randomness is fixed with `random_state=42`.
- Running the full notebook takes roughly 20-25 minutes (the SVM cross-validation steps are the slow ones).
- The last notebook section loads `cardio_svm_final.joblib` and predicts for a new patient. The joblib file requires the same scikit-learn version as in `requirements.txt`.

## License

MIT
