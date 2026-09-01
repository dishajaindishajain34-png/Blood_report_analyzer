# ANEMIA_MODEL.md — Machine Learning Anemia Detection Model Documentation

## 1. Executive Summary

This document describes the secondary machine-learning Anemia Detection stream integrated into the **AI-Based Smart Blood Report Analyzer**. The model is designed to process laboratory Complete Blood Count (CBC) parameters entered by the user and classify whether the patient exhibits Anemia.

---

## 2. Dataset Description & Verification

- **Dataset Name**: Kaggle Anemia Dataset (Biswa Ranjan Rao / open repository mirror)
- **Download Source**: `data/anemia/raw/anemia_dataset.csv`
- **Sample Count**: 1,421 records
- **Feature Count**: 5 features (`Gender`, `Hemoglobin`, `MCH`, `MCHC`, `MCV`)
- **Target Label**: `Result` (0 = Normal / Non-anemic, 1 = Anemic)
- **Data Quality**:
  - Missing Values: 0 (0.0%)
  - Duplicate Records in Raw File: 887 duplicates (62.42%) in raw source table.
  - Label Derivation Check: Empirical analysis demonstrates a **95.00% match** with clinical WHO Hemoglobin thresholds (`Female < 12.0 g/dL`, `Male < 13.0 g/dL`).
- **Limitation Note**: The target label is strongly derived from Hemoglobin thresholds and gender. The model serves as a prototype screening module and is **not** an independent clinical laboratory gold standard.

---

## 3. Data Preprocessing & Splitting

- **Split Ratio**: 70% Training (994), 15% Validation (213), 15% Test (214) with `stratify=y` (`random_state=42`).
- **Class Distributions**:
  - **Full Dataset (1,421)**: Normal = 801 (56.4%), Anemic = 620 (43.6%)
  - **Train Set (994)**: Normal = 560 (56.3%), Anemic = 434 (43.7%)
  - **Validation Set (213)**: Normal = 120 (56.3%), Anemic = 93 (43.7%)
  - **Test Set (214)**: Normal = 121 (56.5%), Anemic = 93 (43.5%)
- **Scaling**: `StandardScaler` fitted strictly on training data (`X_train`) to prevent data leakage.

---

## 4. Model Benchmarking & Validation Selection

Candidate algorithms were evaluated on the Validation Set:

| Model | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Baseline (Reference Range)** | 0.9718 | 1.0000 | 0.9355 | 1.0000 | 0.9667 | 0.5000 |
| **Logistic Regression** | 0.9906 | 0.9789 | 1.0000 | 0.9833 | 0.9894 | 1.0000 |
| **Random Forest (Selected)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Gradient Boosting** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **SVM (RBF Kernel)** | 0.9765 | 0.9490 | 1.0000 | 0.9583 | 0.9738 | 0.9996 |

**Selected Model**: **Random Forest Classifier** (`n_estimators=100`, `class_weight='balanced'`), chosen based on top validation Recall and F1-score.

---

## 5. Held-Out Test Set Performance (Evaluated ONCE)

Evaluation on the untouched 214-sample held-out Test Set:

- **Accuracy**: 1.0000 (100.0%)
- **Precision**: 1.0000 (100.0%)
- **Recall / Sensitivity**: 1.0000 (100.0%)
- **Specificity**: 1.0000 (100.0%)
- **F1-Score**: 1.0000
- **ROC-AUC**: 1.0000
- **Confusion Matrix**: `TN=121, FP=0, FN=0, TP=93`

---

## 6. Feature Importance

Feature contributions extracted from the Random Forest model:

1. **Hemoglobin**: 0.8332 (83.3%)
2. **Gender**: 0.0982 (9.8%)
3. **MCV**: 0.0277 (2.8%)
4. **MCH**: 0.0223 (2.2%)
5. **MCHC**: 0.0186 (1.9%)

*Plot exported to*: `models/anemia/feature_importance.png`

---

## 7. Model Artifacts

Model files stored in `models/anemia/`:
- `anemia_model.joblib`: Serialized Random Forest model
- `scaler.joblib`: Pre-fitted StandardScaler instance
- `feature_columns.json`: `["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]`
- `metrics.json`: Empirical validation & test evaluation metrics
- `feature_importance.png`: Feature importance visualization bar plot

---

## 8. Reproducibility & Commands

To re-verify dataset quality, re-split data, retrain models, and generate all metrics:

```bash
python scripts/train_anemia_model.py
```

To run the unit and integration test suite:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

To launch the complete dual-stream Streamlit application:

```bash
streamlit run frontend/app.py
```

---

## 9. Medical & Ethical Limitations

- **Research Prototype Only**: This model is built for academic and assistive research purposes.
- **Not a Clinical Diagnosis**: Outputs must be reviewed and confirmed by a certified pathologist or physician.
- **Separate Stream Notice**: Blood smear microscope images do not directly yield MCV, MCH, or MCHC values. The ML Anemia stream processes explicit laboratory CBC inputs provided by the user.
