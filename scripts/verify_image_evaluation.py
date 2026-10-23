"""
verify_image_evaluation.py

Independent, reproducible verification of image classifiers on the exact held-out test split:
1. Baseline Checkpoint (mobilenetv3_anemia_original_baseline.pth)
   - Evaluated at default threshold (0.50)
   - Evaluated at validation-tuned frozen threshold (0.7390)
2. Weighted Retrained Checkpoint (mobilenetv3_anemia.pth)
   - Evaluated at default threshold (0.50)
   - Evaluated at validation-tuned frozen threshold (0.8500)

Prints exact checkpoints, sample counts, confusion matrices, and metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, project_path
from scripts.train_anemia_image import load_dataset_metadata, split_dataset

log = get_logger("verify_image")


def run_verification():
    root = project_path("data", "anerbc_raw")
    df = load_dataset_metadata(root)
    
    # Cross-class duplicate check (purging hash e98d48f1d84419386a71aa3dc9df1ae7)
    cross_class_hashes = set(df.groupby("md5")["label"].nunique().loc[lambda s: s > 1].index)
    if cross_class_hashes:
        log.info("Purging %d cross-class duplicate file(s) with conflicting labels: %s",
                 len(cross_class_hashes), cross_class_hashes)
        df = df[~df["md5"].isin(cross_class_hashes)].copy()

    df = df.drop_duplicates(subset=["md5"]).copy()
    log.info("Total clean unique dataset size: %d images", len(df))

    # Identical reproducible split with fixed seed=42
    train_df, val_df, test_df = split_dataset(df, seed=42)
    log.info("Test split size: %d images (Normal: %d, Anemic: %d)",
             len(test_df), (test_df["label"] == 0).sum(), (test_df["label"] == 1).sum())

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def load_model(weights_path: Path):
        m = models.mobilenet_v3_small(weights=None)
        in_feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feat, 2)
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        m.load_state_dict(sd)
        m.eval()
        return m

    def infer(m, split_df):
        probs, labels = [], []
        for _, row in split_df.iterrows():
            img = Image.open(row["filepath"]).convert("RGB")
            t = transform(img).unsqueeze(0)
            with torch.no_grad():
                out = m(t)
                p = torch.softmax(out, dim=1).squeeze().numpy()[1]
                probs.append(float(p))
                labels.append(int(row["label"]))
        return np.array(probs), np.array(labels)

    def metrics_at_thresh(probs, labels, thresh):
        preds = (probs >= thresh).astype(int)
        acc = float(accuracy_score(labels, preds))
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        rep = classification_report(labels, preds, output_dict=True, zero_division=0)
        prec = float(rep["1"]["precision"]) if "1" in rep else 0.0
        f1 = float(rep["1"]["f1-score"]) if "1" in rep else 0.0
        return {
            "threshold": round(thresh, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
        }

    # 1. Baseline Model
    baseline_path = project_path("models", "anemia_image", "mobilenetv3_anemia_original_baseline.pth")
    m_base = load_model(baseline_path)
    val_probs_base, val_labels_base = infer(m_base, val_df)
    test_probs_base, test_labels_base = infer(m_base, test_df)

    # Compute validation-only threshold for baseline
    fpr_vb, tpr_vb, th_vb = roc_curve(val_labels_base, val_probs_base)
    j_base = tpr_vb - fpr_vb
    tau_base = float(th_vb[int(np.argmax(j_base))])
    tau_base = max(0.15, min(0.85, tau_base))
    auc_base = float(roc_auc_score(test_labels_base, test_probs_base))

    base_def = metrics_at_thresh(test_probs_base, test_labels_base, 0.50)
    base_cal = metrics_at_thresh(test_probs_base, test_labels_base, tau_base)

    # 2. Weighted Retrained Model
    weighted_path = project_path("models", "anemia_image", "mobilenetv3_anemia.pth")
    m_weight = load_model(weighted_path)
    val_probs_w, val_labels_w = infer(m_weight, val_df)
    test_probs_w, test_labels_w = infer(m_weight, test_df)

    # Compute validation-only threshold for weighted model
    fpr_vw, tpr_vw, th_vw = roc_curve(val_labels_w, val_probs_w)
    j_w = tpr_vw - fpr_vw
    tau_weight = float(th_vw[int(np.argmax(j_w))])
    tau_weight = max(0.15, min(0.85, tau_weight))
    auc_weight = float(roc_auc_score(test_labels_w, test_probs_w))

    weight_def = metrics_at_thresh(test_probs_w, test_labels_w, 0.50)
    weight_cal = metrics_at_thresh(test_probs_w, test_labels_w, tau_weight)

    print("\n" + "="*80)
    print("INDEPENDENT IMAGE CLASSIFIER VERIFICATION RESULTS")
    print("="*80)
    print(f"Test Set Size: {len(test_df)} images ({int((test_df['label']==0).sum())} Normal, {int((test_df['label']==1).sum())} Anemic)")
    print(f"\n1. BASELINE MODEL: {baseline_path.name}")
    print(f"   Validation Youden's J: {float(np.max(j_base)):.4f} -> Frozen Threshold: {tau_base:.4f}")
    print(f"   Test ROC-AUC: {auc_base:.4f}")
    print(f"   - At Threshold 0.50 : Acc={base_def['accuracy']:.4f}, Sens={base_def['sensitivity']:.4f}, Spec={base_def['specificity']:.4f}, F1={base_def['f1_score']:.4f}")
    print(f"     Confusion Matrix   : {base_def['confusion_matrix']}")
    print(f"   - At Calibrated ({tau_base:.4f}): Acc={base_cal['accuracy']:.4f}, Sens={base_cal['sensitivity']:.4f}, Spec={base_cal['specificity']:.4f}, F1={base_cal['f1_score']:.4f}")
    print(f"     Confusion Matrix   : {base_cal['confusion_matrix']}")

    print(f"\n2. WEIGHTED RETRAINED MODEL: {weighted_path.name}")
    print(f"   Validation Youden's J: {float(np.max(j_w)):.4f} -> Frozen Threshold: {tau_weight:.4f}")
    print(f"   Test ROC-AUC: {auc_weight:.4f}")
    print(f"   - At Threshold 0.50 : Acc={weight_def['accuracy']:.4f}, Sens={weight_def['sensitivity']:.4f}, Spec={weight_def['specificity']:.4f}, F1={weight_def['f1_score']:.4f}")
    print(f"     Confusion Matrix   : {weight_def['confusion_matrix']}")
    print(f"   - At Calibrated ({tau_weight:.4f}): Acc={weight_cal['accuracy']:.4f}, Sens={weight_cal['sensitivity']:.4f}, Spec={weight_cal['specificity']:.4f}, F1={weight_cal['f1_score']:.4f}")
    print(f"     Confusion Matrix   : {weight_cal['confusion_matrix']}")
    print("="*80 + "\n")

    out_file = project_path("models", "anemia_image", "independent_verification.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_sample_count": len(test_df),
            "test_normal": int((test_df["label"] == 0).sum()),
            "test_anemic": int((test_df["label"] == 1).sum()),
            "baseline_model": {
                "checkpoint": str(baseline_path),
                "frozen_threshold": tau_base,
                "roc_auc": auc_base,
                "default_metrics": base_def,
                "calibrated_metrics": base_cal
            },
            "weighted_retrained_model": {
                "checkpoint": str(weighted_path),
                "frozen_threshold": tau_weight,
                "roc_auc": auc_weight,
                "default_metrics": weight_def,
                "calibrated_metrics": weight_cal
            }
        }, f, indent=2)
    log.info("Saved independent verification JSON -> %s", out_file)
    return True


if __name__ == "__main__":
    run_verification()
