# DATA_LEAKAGE_CHECK.md — Data Leakage Prevention Verification Log

This document details the checks and safeguards applied during the development of the ML Anemia Detection model for the **AI-Based Smart Blood Report Analyzer** project.

---

## 1. Dataset Verification & Quality Findings

- **Source**: Kaggle Anemia Dataset (by Biswa Ranjan Rao / open repository mirror).
- **Total Records**: 1,421
- **Features (5)**: `Gender` (0=Female, 1=Male), `Hemoglobin` (g/dL), `MCH` (pg), `MCHC` (g/dL), `MCV` (fL).
- **Target (1)**: `Result` (0 = Normal / Non-anemic, 1 = Anemic).
- **Missing Values**: 0 (0.0%).
- **Duplicate Records in Raw Data**: 887 duplicates (62.42%) exist in the raw public Kaggle table.
- **Target Label Derivation Check**: Empirical verification shows a **95.00% concordance** between the `Result` target column and standard WHO Hemoglobin threshold rules (`Female: Hb < 12.0 g/dL`, `Male: Hb < 13.0 g/dL`).
  - *Medical Limitation Note*: The ground-truth `Result` in this dataset is strongly threshold-derived from Hemoglobin and gender. It is presented as a screening prototype model, not as an independent clinical laboratory gold standard.

---

## 2. Data Leakage Prevention Checklist

| Safeguard Item | Status | Method / Implementation Details |
|---|---|---|
| **Patient-Level / Stratified Split** | ✅ Passed | 70% Train (994), 15% Val (213), 15% Test (214) split using `stratify=y` and fixed `random_state=42`. Class distributions preserved across all splits (43.6%–43.7% anemic). |
| **No Oversampling Leakage** | ✅ Passed | Synthetic oversampling (e.g. SMOTE) was **NOT** used prior to splitting. Class weighting (`class_weight='balanced'`) was applied directly inside model estimators on the training set. |
| **Scaler Fit-Only-on-Train** | ✅ Passed | `StandardScaler` was fitted strictly on `X_train` (`scaler.fit_transform(X_train)`). Validation and Test sets were transformed using the pre-fitted scaler (`scaler.transform()`). |
| **Model Selection via Validation Set** | ✅ Passed | Candidate models (Logistic Regression, Random Forest, Gradient Boosting, SVM, Baseline) were evaluated exclusively on the Validation Set. The final model (**Random Forest**) was selected based on validation Recall & F1-score. |
| **Test Set Isolation** | ✅ Passed | The held-out Test Set was kept completely untouched during model training and hyperparameter tuning, and evaluated **ONCE** at the end. |
| **No Target Leakage in Features** | ✅ Passed | Input features consist solely of raw lab parameters (`Gender`, `Hemoglobin`, `MCH`, `MCHC`, `MCV`). The target column `Result` was strictly isolated. |

---

## 3. Summary of Split Class Distributions

- **Full Dataset (1,421 samples)**:
  - Normal (0): 801 (56.4%)
  - Anemic (1): 620 (43.6%)

- **Training Set (994 samples - 70%)**:
  - Normal (0): 560 (56.3%)
  - Anemic (1): 434 (43.7%)

- **Validation Set (213 samples - 15%)**:
  - Normal (0): 120 (56.3%)
  - Anemic (1): 93 (43.7%)

- **Test Set (214 samples - 15%)**:
  - Normal (0): 121 (56.5%)
  - Anemic (1): 93 (43.5%)
