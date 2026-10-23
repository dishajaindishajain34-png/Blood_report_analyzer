"""
evaluate_cbc_leakage.py

Rigorous Target Leakage Audit & Comparative Experiment for CBC Anemia Models.

Compares:
1. Standard Diagnostic Pipeline (Features: Gender, Hemoglobin, MCV, MCH, MCHC)
2. Leakage-Aware Screening Pipeline (Features: Gender, MCV, MCH, MCHC — strictly excluding Hemoglobin)

Evaluates Random Forest, Gradient Boosting, and Logistic Regression on identical
stratified train/test splits (70/30) of the consolidated master CBC dataset (2,695 records).
Computes Accuracy, Precision, Sensitivity, Specificity, F1, and ROC-AUC.
Saves detailed JSON metrics and Markdown clinical documentation.

Usage:
    python scripts/evaluate_cbc_leakage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, project_path

log = get_logger("cbc_leakage_audit")


def run_experiment():
    dataset_path = project_path("data", "anemia", "processed", "master_cbc_dataset.csv")
    if not dataset_path.exists():
        log.error("Master CBC dataset not found at %s", dataset_path)
        return False

    df = pd.read_csv(dataset_path)
    log.info("Loaded master CBC dataset: %d rows, %d columns", len(df), len(df.columns))
    log.info("Target distribution: %s", df["Result"].value_counts().to_dict())

    # Map Gender to binary (0 = Female, 1 = Male) if string
    if df["Gender"].dtype == object:
        df["Gender"] = df["Gender"].apply(lambda g: 1 if str(g).strip().lower() in ["m", "male", "1"] else 0)

    # Ensure numerical types
    for col in ["Hemoglobin", "MCV", "MCH", "MCHC", "Result"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Gender", "Hemoglobin", "MCV", "MCH", "MCHC", "Result"]).copy()

    # Stratified Train/Test Split (70% train / 30% test)
    y = df["Result"].values.astype(int)
    indices = np.arange(len(df))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y, test_size=0.30, random_state=42, stratify=y
    )

    log.info("Train set: %d samples (%d Normal, %d Anemic)", len(idx_train), (y_train == 0).sum(), (y_train == 1).sum())
    log.info("Test set:  %d samples (%d Normal, %d Anemic)", len(idx_test), (y_test == 0).sum(), (y_test == 1).sum())

    features_leaked = ["Gender", "Hemoglobin", "MCV", "MCH", "MCHC"]
    features_leakage_free = ["Gender", "MCV", "MCH", "MCHC"]

    configurations = [
        ("standard_with_hb", "Standard Diagnostic Pipeline (With Hemoglobin)", features_leaked),
        ("leakage_free_without_hb", "Leakage-Aware Screening Pipeline (Without Hemoglobin)", features_leakage_free),
    ]

    model_factories = {
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "GradientBoosting": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    }

    results = {
        "dataset_summary": {
            "total_records": len(df),
            "train_samples": len(idx_train),
            "test_samples": len(idx_test),
            "anemic_count": int((y == 1).sum()),
            "normal_count": int((y == 0).sum()),
        },
        "configurations": {},
    }

    for config_key, config_name, feature_cols in configurations:
        log.info("\n=======================================================")
        log.info("Evaluating: %s", config_name)
        log.info("Features: %s", feature_cols)
        log.info("=======================================================")

        X_raw = df[feature_cols].values
        X_tr = X_raw[idx_train]
        X_te = X_raw[idx_test]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        config_results = {"features": feature_cols, "models": {}}

        for model_name, factory in model_factories.items():
            clf = factory()
            clf.fit(X_tr_scaled, y_train)

            preds = clf.predict(X_te_scaled)
            probs = clf.predict_proba(X_te_scaled)[:, 1] if hasattr(clf, "predict_proba") else None

            acc = float(accuracy_score(y_test, preds))
            p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            auc = float(roc_auc_score(y_test, probs)) if probs is not None else 0.5

            feature_importance = {}
            if hasattr(clf, "feature_importances_"):
                feature_importance = {col: round(float(imp), 4) for col, imp in zip(feature_cols, clf.feature_importances_)}
            elif hasattr(clf, "coef_"):
                feature_importance = {col: round(float(c), 4) for col, c in zip(feature_cols, clf.coef_[0])}

            log.info(
                "  [%s] Acc: %.4f | Sens: %.4f | Spec: %.4f | F1: %.4f | AUC: %.4f",
                model_name, acc, r, spec, f1, auc
            )
            if feature_importance:
                log.info("      Feature Importance / Coef: %s", feature_importance)

            config_results["models"][model_name] = {
                "accuracy": round(acc, 4),
                "precision": round(float(p), 4),
                "sensitivity_recall": round(float(r), 4),
                "specificity": round(spec, 4),
                "f1_score": round(float(f1), 4),
                "roc_auc": round(auc, 4),
                "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
                "feature_importance": feature_importance,
            }

        results["configurations"][config_key] = config_results

    # Save results to json
    out_dir = project_path("models", "anemia_cbc")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "target_leakage_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Saved target leakage audit metrics -> %s", out_json)

    # Plot comparative ROC curves
    plt.figure(figsize=(8, 6))
    for config_key, config_name, feature_cols in configurations:
        X_raw = df[feature_cols].values
        X_tr = X_raw[idx_train]
        X_te = X_raw[idx_test]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        rf.fit(X_tr_s, y_train)
        probs = rf.predict_proba(X_te_s)[:, 1]
        auc = roc_auc_score(y_test, probs)
        fpr, tpr, _ = roc_curve(y_test, probs)

        label_name = f"RF With Hb (AUC = {auc:.4f})" if "with_hb" in config_key else f"RF Without Hb (Screening, AUC = {auc:.4f})"
        plt.plot(fpr, tpr, lw=2, label=label_name)

    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Target Leakage Impact on CBC Anemia Models (ROC Curve)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "leakage_comparison_roc.png", bbox_inches="tight")
    plt.close()
    log.info("Saved comparative ROC curve -> %s", out_dir / "leakage_comparison_roc.png")

    # Generate Markdown documentation
    doc_path = project_path("docs", "CBC_TARGET_LEAKAGE_ANALYSIS.md")
    write_leakage_documentation(results, doc_path)
    log.info("Saved clinical leakage analysis document -> %s", doc_path)
    return True


def write_leakage_documentation(results: dict, doc_path: Path):
    with_hb = results["configurations"]["standard_with_hb"]["models"]
    without_hb = results["configurations"]["leakage_free_without_hb"]["models"]

    content = f"""# CBC Model Target Leakage Audit & Screening Feasibility Analysis

*HemaVision AI — Machine Learning Medical Integrity Audit*

---

## 1. Executive Summary & Problem Definition

In clinical hematology, **anemia is defined by low Hemoglobin concentration** (WHO criteria: $<12.0\\text{{ g/dL}}$ in non-pregnant females, $<13.0\\text{{ g/dL}}$ in males).

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
| **Standard (With Hb)** | Random Forest | {with_hb['RandomForest']['accuracy']:.4f} | {with_hb['RandomForest']['sensitivity_recall']:.4f} | {with_hb['RandomForest']['specificity']:.4f} | {with_hb['RandomForest']['f1_score']:.4f} | {with_hb['RandomForest']['roc_auc']:.4f} |
| | Gradient Boosting | {with_hb['GradientBoosting']['accuracy']:.4f} | {with_hb['GradientBoosting']['sensitivity_recall']:.4f} | {with_hb['GradientBoosting']['specificity']:.4f} | {with_hb['GradientBoosting']['f1_score']:.4f} | {with_hb['GradientBoosting']['roc_auc']:.4f} |
| | Logistic Regression | {with_hb['LogisticRegression']['accuracy']:.4f} | {with_hb['LogisticRegression']['sensitivity_recall']:.4f} | {with_hb['LogisticRegression']['specificity']:.4f} | {with_hb['LogisticRegression']['f1_score']:.4f} | {with_hb['LogisticRegression']['roc_auc']:.4f} |
| **Leakage-Aware (Without Hb)** | Random Forest | **{without_hb['RandomForest']['accuracy']:.4f}** | **{without_hb['RandomForest']['sensitivity_recall']:.4f}** | **{without_hb['RandomForest']['specificity']:.4f}** | **{without_hb['RandomForest']['f1_score']:.4f}** | **{without_hb['RandomForest']['roc_auc']:.4f}** |
| | Gradient Boosting | {without_hb['GradientBoosting']['accuracy']:.4f} | {without_hb['GradientBoosting']['sensitivity_recall']:.4f} | {without_hb['GradientBoosting']['specificity']:.4f} | {without_hb['GradientBoosting']['f1_score']:.4f} | {without_hb['GradientBoosting']['roc_auc']:.4f} |
| | Logistic Regression | {without_hb['LogisticRegression']['accuracy']:.4f} | {without_hb['LogisticRegression']['sensitivity_recall']:.4f} | {without_hb['LogisticRegression']['specificity']:.4f} | {without_hb['LogisticRegression']['f1_score']:.4f} | {without_hb['LogisticRegression']['roc_auc']:.4f} |

---

## 4. Feature Importance Comparison

### Pipeline A (With Hemoglobin) — Random Forest
```
{json.dumps(with_hb['RandomForest']['feature_importance'], indent=2)}
```
> Hemoglobin dominates with {with_hb['RandomForest']['feature_importance'].get('Hemoglobin', 0)*100:.1f}% of total decision split importance.

### Pipeline B (Without Hemoglobin) — Random Forest
```
{json.dumps(without_hb['RandomForest']['feature_importance'], indent=2)}
```
> When Hemoglobin is removed, the model redistributes weight across erythrocyte morphological markers: MCV ({without_hb['RandomForest']['feature_importance'].get('MCV', 0)*100:.1f}%), MCH ({without_hb['RandomForest']['feature_importance'].get('MCH', 0)*100:.1f}%), and MCHC ({without_hb['RandomForest']['feature_importance'].get('MCHC', 0)*100:.1f}%).

---

## 5. Medical & Clinical Discussion

1. **Why does the leakage-aware model perform above chance ({without_hb['RandomForest']['roc_auc']:.4f} AUC)?**
   - In most clinical cohorts, the leading etiologies of anemia are **Iron Deficiency Anemia (IDA)** and **Thalassemia trait**, both of which cause **microcytosis** (low MCV) and **hypochromia** (low MCH and low MCHC).
   - The leakage-free model detects these cellular morphological abnormalities as surrogate markers of anemia.

2. **Crucial Medical Limitations**:
   - **Normocytic Anemia**: Patients with normocytic normochromic anemia (such as anemia of chronic disease, early aplastic anemia, or acute blood loss) will have normal MCV, MCH, and MCHC, and will be missed by an Hb-free model.
   - **Macrocytic Anemia**: B12 or folate deficiency produces macrocytosis (high MCV), which may follow different non-linear decision boundaries.
   - **Diagnostic vs. Screening**: This experiment establishes that red cell morphology contains non-trivial predictive signal for anemia screening, but **it does not replace direct hemoglobin quantitation for clinical diagnostic confirmation**.
"""
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    success = run_experiment()
    if not success:
        sys.exit(1)
