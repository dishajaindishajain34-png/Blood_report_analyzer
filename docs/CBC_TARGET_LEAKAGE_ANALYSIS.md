# CBC Model Target Leakage Audit & Screening Feasibility Analysis

*HemaVision AI — Machine Learning Medical Integrity Audit*

---

## 1. Executive Summary & Problem Definition

In clinical hematology, **anemia is defined by low Hemoglobin concentration** (WHO criteria: $<12.0\text{ g/dL}$ in non-pregnant females, $<13.0\text{ g/dL}$ in males).

In public machine-learning benchmarks (including Kaggle's Anemia Dataset and combined clinical CBC datasets), the ground-truth binary label (`Result` or `Diagnosis`) is derived mathematically from the `Hemoglobin` and `Gender` values. 

When a machine learning classifier is given `Hemoglobin` as an input feature to predict an Hb-derived label:
1. The model achieves **~100% accuracy** almost trivially.
2. The tree feature importance for `Hemoglobin` exceeds **80–90%**.
3. **Target Leakage**: The model has not learned physiological patterns of disease; it has merely memorized the deterministic threshold rule used to label the rows.

To address this, we conducted a rigorous comparative experiment evaluating whether red blood cell indices (**MCV, MCH, MCHC**) and **Gender** can predict anemia *without* access to raw Hemoglobin.

---

## 2. Experimental Setup

- **Dataset**: Consolidated Master CBC Dataset (`master_cbc_dataset.csv`, 2,695 patient records)
- **Class Balance**: 1,429 Anemic (53.0%), 1,266 Normal (47.0%)
- **Split Strategy**: Stratified 70% Train (1,886 samples) / 30% Test (809 samples held-out)
- **Algorithms Compared**: Random Forest (RF), Gradient Boosting (GB), Logistic Regression (LR)
- **Feature Sets**:
  - **Pipeline A (Standard / Leaked)**: `[Gender, Hemoglobin, MCV, MCH, MCHC]`
  - **Pipeline B (Leakage-Aware Screening)**: `[Gender, MCV, MCH, MCHC]` (strictly **excluding `Hemoglobin`**)

---

## 3. Comparative Results (Held-Out Test Set — 809 Patients)

| Pipeline | Model | Accuracy | Sensitivity | Specificity | F1-Score | ROC-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Standard (With Hb)** | Random Forest | 0.9951 | 0.9953 | 0.9947 | 0.9953 | 0.9999 |
| | Gradient Boosting | 0.9988 | 0.9977 | 1.0000 | 0.9988 | 1.0000 |
| | Logistic Regression | 0.9481 | 0.9464 | 0.9500 | 0.9508 | 0.9905 |
| **Leakage-Aware (Without Hb)** | Random Forest | **0.8356** | **0.8928** | **0.7711** | **0.8521** | **0.8974** |
| | Gradient Boosting | 0.8072 | 0.8392 | 0.7711 | 0.8219 | 0.8774 |
| | Logistic Regression | 0.6452 | 0.7483 | 0.5289 | 0.6911 | 0.6646 |

---

## 4. Feature Importance Comparison

### Pipeline A (With Hemoglobin) — Random Forest
```
{
  "Gender": 0.0543,
  "Hemoglobin": 0.7998,
  "MCV": 0.07,
  "MCH": 0.0452,
  "MCHC": 0.0308
}
```
> Hemoglobin dominates with 80.0% of total decision split importance.

### Pipeline B (Without Hemoglobin) — Random Forest
```
{
  "Gender": 0.0952,
  "MCV": 0.3863,
  "MCH": 0.2777,
  "MCHC": 0.2408
}
```
> When Hemoglobin is removed, the model redistributes weight across erythrocyte morphological markers: MCV (38.6%), MCH (27.8%), and MCHC (24.1%).

---

## 5. Medical & Clinical Discussion

1. **Why does the leakage-aware model perform above chance (0.8974 AUC)?**
   - In most clinical cohorts, the leading etiologies of anemia are **Iron Deficiency Anemia (IDA)** and **Thalassemia trait**, both of which cause **microcytosis** (low MCV) and **hypochromia** (low MCH and low MCHC).
   - The leakage-free model detects these cellular morphological abnormalities as surrogate markers of anemia.

2. **Crucial Medical Limitations**:
   - **Normocytic Anemia**: Patients with normocytic normochromic anemia (such as anemia of chronic disease, early aplastic anemia, or acute blood loss) will have normal MCV, MCH, and MCHC, and will be missed by an Hb-free model.
   - **Macrocytic Anemia**: B12 or folate deficiency produces macrocytosis (high MCV), which may follow different non-linear decision boundaries.
   - **Diagnostic vs. Screening**: This experiment establishes that red cell morphology contains non-trivial predictive signal for anemia screening, but **it does not replace direct hemoglobin quantitation for clinical diagnostic confirmation**.
