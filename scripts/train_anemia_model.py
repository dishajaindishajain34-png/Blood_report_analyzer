"""
train_anemia_model.py — Dataset Verification, Preprocessing, Baseline Comparison,
Candidate Model Training, Validation Selection, Test Set Evaluation, and Artifact Export.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Directories
RAW_DIR = PROJECT_ROOT / "data" / "anemia" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "anemia" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "anemia"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATASET_URL = (
    "https://raw.githubusercontent.com/maladeep/"
    "anemia-detection-with-machine-learning/master/" +
    urllib.parse.quote("anemia data from Kaggle.csv")
)
RAW_CSV_PATH = RAW_DIR / "anemia_dataset.csv"

def download_dataset():
    """Download raw Kaggle Anemia dataset if not already present."""
    if not RAW_CSV_PATH.exists():
        print(f"Downloading dataset from: {DATASET_URL}")
        req = urllib.request.urlopen(DATASET_URL)
        content = req.read().decode("utf-8")
        with open(RAW_CSV_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[OK] Saved raw dataset to {RAW_CSV_PATH}")
    else:
        print(f"[OK] Raw dataset already exists at {RAW_CSV_PATH}")

def inspect_dataset(df: pd.DataFrame):
    """Inspect dataset properties, missing values, duplicates, and label characteristics."""
    print("\n" + "=" * 80)
    print("STEP 1: DATASET VERIFICATION & QUALITY CHECKS")
    print("=" * 80)
    print(f"Number of rows:    {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Column Names:      {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}\n")

    missing = df.isnull().sum()
    print(f"Missing Values:\n{missing}")
    print(f"Missing percentage: {missing.sum() / (df.shape[0] * df.shape[1]):.2%}")

    duplicates = df.duplicated().sum()
    print(f"Duplicate records: {duplicates} ({duplicates / len(df):.2%})")

    print("\nClass Distribution (Result):")
    class_counts = df["Result"].value_counts().to_dict()
    print(f"  0 (Non-anemic): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(df):.2%})")
    print(f"  1 (Anemic):     {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(df):.2%})")

    # Threshold Check: Check if Result label strictly correlates with WHO Hemoglobin thresholds
    # WHO Hemoglobin Thresholds: Female < 12.0 g/dL, Male < 13.0 g/dL (Gender=0 is Female, Gender=1 is Male in dataset)
    # Let's verify how Result relates to Hemoglobin and Gender in this dataset.
    hb_thresh_derived = []
    for idx, row in df.iterrows():
        gender = row["Gender"] # 0 = female, 1 = male
        hb = row["Hemoglobin"]
        expected_anemic = 1 if ((gender == 0 and hb < 12.0) or (gender == 1 and hb < 13.0)) else 0
        hb_thresh_derived.append(expected_anemic == row["Result"])
    match_pct = np.mean(hb_thresh_derived)
    print(f"\nLabel Origin Check (Match against WHO Hb Threshold Rule): {match_pct:.2%}")
    if match_pct > 0.95:
        print("-> NOTE: The 'Result' label in this dataset aligns strongly with clinical WHO Hemoglobin thresholds.")
        print("   This label derivation will be documented in docs/ANEMIA_MODEL.md.")

def evaluate_metrics(y_true, y_pred, y_prob=None):
    """Compute comprehensive medical evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity
    
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5
    
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall_sensitivity": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
        }
    }

def rule_based_baseline_predict(X_df):
    """Existing rule/threshold-based reference-range assessment as baseline."""
    preds = []
    for idx, row in X_df.iterrows():
        gender = row["Gender"]
        hb = row["Hemoglobin"]
        # WHO Reference: Female < 12.0 g/dL -> Anemic (1), Male < 13.0 g/dL -> Anemic (1)
        if (gender == 0 and hb < 12.0) or (gender == 1 and hb < 13.0):
            preds.append(1)
        else:
            preds.append(0)
    return np.array(preds)

def main():
    download_dataset()
    df = pd.read_csv(RAW_CSV_PATH)
    inspect_dataset(df)

    feature_cols = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
    target_col = "Result"

    X = df[feature_cols]
    y = df[target_col]

    print("\n" + "=" * 80)
    print("STEP 2: STRATIFIED DATA SPLITTING (70% Train / 15% Val / 15% Test)")
    print("=" * 80)
    
    # 70% Train, 30% Temp (15% Val, 15% Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Save processed CSV splits
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"[OK] Saved splits to {PROCESSED_DIR}")

    print("\nClass Distributions across Splits:")
    def get_dist(series):
        vc = series.value_counts().to_dict()
        return f"Total={len(series)}, Normal={vc.get(0,0)} ({vc.get(0,0)/len(series):.1%}), Anemic={vc.get(1,0)} ({vc.get(1,0)/len(series):.1%})"
    
    print(f"  Full Dataset: {get_dist(y)}")
    print(f"  Train Set:    {get_dist(y_train)}")
    print(f"  Val Set:      {get_dist(y_val)}")
    print(f"  Test Set:     {get_dist(y_test)}")

    print("\n" + "=" * 80)
    print("STEP 3: PREPROCESSING & SCALER FITTING")
    print("=" * 80)
    # Scaler is fitted ONLY on training data to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("[OK] StandardScaler fitted strictly on X_train.")

    print("\n" + "=" * 80)
    print("STEP 4: CANDIDATE MODEL TRAINING & VALIDATION SELECTION")
    print("=" * 80)

    # Candidate Models (handling class imbalance via class_weight='balanced')
    candidates = {
        "Baseline (Rule-Based Ref Range)": None,
        "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM (RBF Kernel)": SVC(probability=True, class_weight="balanced", random_state=42)
    }

    val_results = {}
    fitted_models = {}

    for name, model in candidates.items():
        if name == "Baseline (Rule-Based Ref Range)":
            y_val_pred = rule_based_baseline_predict(X_val)
            y_val_prob = None
        else:
            model.fit(X_train_scaled, y_train)
            fitted_models[name] = model
            y_val_pred = model.predict(X_val_scaled)
            y_val_prob = model.predict_proba(X_val_scaled)[:, 1]

        metrics = evaluate_metrics(y_val, y_val_pred, y_val_prob)
        val_results[name] = metrics

        print(f"\n--- Model: {name} (Validation Set) ---")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Precision:         {metrics['precision']:.4f}")
        print(f"  Recall/Sensitivity:{metrics['recall_sensitivity']:.4f}")
        print(f"  Specificity:       {metrics['specificity']:.4f}")
        print(f"  F1-Score:          {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
        print(f"  Confusion Matrix:  {metrics['confusion_matrix']}")

    # Select Best Model based on Validation Recall & F1-Score
    best_name = max(
        fitted_models.keys(),
        key=lambda k: (val_results[k]["recall_sensitivity"], val_results[k]["f1_score"])
    )
    best_model = fitted_models[best_name]
    print(f"\n[SELECTED] Final Model based on Validation Set: {best_name}")

    print("\n" + "=" * 80)
    print("STEP 5: FINAL EVALUATION ON HELD-OUT TEST SET (UNTOUCHED UNTIL NOW)")
    print("=" * 80)
    y_test_pred = best_model.predict(X_test_scaled)
    y_test_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    test_metrics = evaluate_metrics(y_test, y_test_pred, y_test_prob)

    print(f"\nFinal Selected Model: {best_name} (Test Set Performance)")
    print(f"  Test Accuracy:           {test_metrics['accuracy']:.4f}")
    print(f"  Test Precision:          {test_metrics['precision']:.4f}")
    print(f"  Test Recall/Sensitivity: {test_metrics['recall_sensitivity']:.4f}")
    print(f"  Test Specificity:        {test_metrics['specificity']:.4f}")
    print(f"  Test F1-Score:           {test_metrics['f1_score']:.4f}")
    print(f"  Test ROC-AUC:            {test_metrics['roc_auc']:.4f}")
    print(f"  Test Confusion Matrix:   {test_metrics['confusion_matrix']}")

    print("\n" + "=" * 80)
    print("STEP 6: FEATURE IMPORTANCE & EXPLAINABILITY")
    print("=" * 80)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.ones(len(feature_cols)) / len(feature_cols)

    feature_imp = dict(zip(feature_cols, [round(float(i), 4) for i in importances]))
    print(f"Feature Importances for {best_name}:")
    for feat, imp in sorted(feature_imp.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {feat:<15}: {imp:.4f}")

    # Save Plot
    plt.figure(figsize=(8, 4.5))
    sorted_feats = sorted(feature_imp.items(), key=lambda x: x[1], reverse=False)
    plt.barh([x[0] for x in sorted_feats], [x[1] for x in sorted_feats], color="#0d9488")
    plt.xlabel("Importance Score")
    plt.title(f"Feature Importance - {best_name}")
    plt.tight_layout()
    plot_path = MODEL_DIR / "feature_importance.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[OK] Saved feature importance plot to {plot_path}")

    print("\n" + "=" * 80)
    print("STEP 7: EXPORTING MODEL ARTIFACTS")
    print("=" * 80)
    
    # Save Model, Scaler, Feature Names, and Metadata
    joblib.dump(best_model, MODEL_DIR / "anemia_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    
    with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    export_metadata = {
        "model_name": best_name,
        "model_version": "1.0.0",
        "dataset_source": "Kaggle Anemia Dataset (Biswa Ranjan Rao)",
        "features": feature_cols,
        "class_mapping": {"0": "Normal", "1": "Anemic"},
        "sample_counts": {
            "total": len(df),
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test)
        },
        "class_distributions": {
            "full": df["Result"].value_counts().to_dict(),
            "train": y_train.value_counts().to_dict(),
            "val": y_val.value_counts().to_dict(),
            "test": y_test.value_counts().to_dict()
        },
        "validation_results": val_results,
        "test_results": test_metrics,
        "feature_importances": feature_imp
    }

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(export_metadata, f, indent=2)

    print(f"[OK] Saved anemia_model.joblib, scaler.joblib, feature_columns.json, metrics.json to {MODEL_DIR}")
    print("\n[OK] Training & Evaluation Complete!")

if __name__ == "__main__":
    main()
