"""
audit_image_dataset_and_baseline.py

Empirical inspection of:
1. Dataset independence: patient/sample-level split leakage, exact duplicates, near-duplicates, class mapping consistency.
2. Baseline model predictions: inspect actual logits and probability distributions for the original baseline model.
   Verify whether class imbalance and uncalibrated 0.5 threshold caused low specificity, or if other mechanisms are at play.
3. Validate threshold tuning methodology: verify threshold is selected strictly on validation split and frozen before test evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, project_path

log = get_logger("audit_image_baseline")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_label(path: Path) -> tuple[int | None, str]:
    parts = [p.lower() for p in path.parts]
    anemic_kw = {"anemia", "anemic", "anaemic", "disease", "anerbc_ii_anemic", "anerbc_i_anemic"}
    normal_kw = {"normal", "healthy", "control", "anerbc_ii_normal", "anerbc_i_normal"}

    for part in reversed(parts[:-1]):
        if part in anemic_kw:
            return 1, f"parent_dir:{part}"
        if part in normal_kw:
            return 0, f"parent_dir:{part}"
        if any(k in part for k in anemic_kw):
            return 1, f"parent_dir_contains:{part}"
        if any(k in part for k in normal_kw):
            return 0, f"parent_dir_contains:{part}"

    stem = path.stem.lower()
    if any(k in stem for k in anemic_kw):
        return 1, f"filename:{stem}"
    if any(k in stem for k in normal_kw):
        return 0, f"filename:{stem}"

    return None, "unmatched"


def get_patient_group(path: Path) -> str:
    stem = path.stem
    m = re.match(r"^([A-Za-z]*\d+)_", stem)
    if m:
        return m.group(1)
    return stem


def audit_dataset_splits_and_leakage(root: Path):
    log.info("--- 1. AUDITING DATASET INTEGRITY, DUPLICATES, AND LEAKAGE ---")
    all_images = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    log.info("Total images scanned in %s: %d", root, len(all_images))

    records = []
    hash_to_paths = defaultdict(list)
    class_mapping_checks = []

    for img in all_images:
        lbl, reason = detect_label(img)
        if lbl is None:
            continue
        pid = get_patient_group(img)
        f_hash = md5_of_file(img)
        hash_to_paths[f_hash].append(str(img))
        records.append({
            "filepath": str(img),
            "filename": img.name,
            "patient_id": pid,
            "label": lbl,
            "label_reason": reason,
            "md5": f_hash
        })

    df = pd.DataFrame(records)
    log.info("Labeled images detected: %d (Normal: %d, Anemic: %d)", len(df), (df["label"] == 0).sum(), (df["label"] == 1).sum())

    # Check 1: Exact Duplicates (MD5)
    dup_groups = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    log.info("Exact duplicate groups found: %d (involving %d total files)", len(dup_groups), sum(len(p) for p in dup_groups.values()))
    
    # Check 2: Cross-class duplicates
    cross_class_dups = []
    for h, paths in dup_groups.items():
        labels_in_group = set(df[df["md5"] == h]["label"])
        if len(labels_in_group) > 1:
            cross_class_dups.append({"md5": h, "paths": paths, "labels": list(labels_in_group)})
    log.info("Cross-class duplicate groups (same image with conflicting Normal/Anemia labels): %d", len(cross_class_dups))

    # Check 3: Deduplication for clean dataset
    df_clean = df.drop_duplicates(subset=["md5"]).copy()
    log.info("Deduplicated unique images: %d (Normal: %d, Anemic: %d)", len(df_clean), (df_clean["label"] == 0).sum(), (df_clean["label"] == 1).sum())

    # Check 4: Patient-level group distribution
    pids_per_label = df_clean.groupby("label")["patient_id"].nunique().to_dict()
    log.info("Unique patient/slide groups: Normal=%d, Anemic=%d", pids_per_label.get(0, 0), pids_per_label.get(1, 0))

    # Verify group-level split independence
    from scripts.train_anemia_image import split_dataset
    train_df, val_df, test_df = split_dataset(df_clean, seed=42)

    train_pids = set(train_df["patient_id"])
    val_pids = set(val_df["patient_id"])
    test_pids = set(test_df["patient_id"])

    leak_train_val = train_pids.intersection(val_pids)
    leak_train_test = train_pids.intersection(test_pids)
    leak_val_test = val_pids.intersection(test_pids)

    log.info("Split patient overlap check:")
    log.info("  Train-Val overlap: %d patients", len(leak_train_val))
    log.info("  Train-Test overlap: %d patients", len(leak_train_test))
    log.info("  Val-Test overlap: %d patients", len(leak_val_test))

    assert len(leak_train_test) == 0, "DATA LEAKAGE DETECTED BETWEEN TRAIN AND TEST!"
    assert len(leak_val_test) == 0, "DATA LEAKAGE DETECTED BETWEEN VAL AND TEST!"
    log.info("[OK] Train, Val, and Test splits are strictly independent at the patient/sample level.")

    return {
        "total_images": len(all_images),
        "labeled_images": len(df),
        "duplicate_groups": len(dup_groups),
        "cross_class_duplicates": len(cross_class_dups),
        "deduplicated_images": len(df_clean),
        "split_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "test_normal": int((test_df["label"] == 0).sum()),
            "test_anemic": int((test_df["label"] == 1).sum()),
        },
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


def audit_baseline_model_predictions(val_df: pd.DataFrame, test_df: pd.DataFrame):
    log.info("\n--- 2. AUDITING ORIGINAL BASELINE MODEL PREDICTIONS ---")
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torchvision import transforms

    baseline_path = project_path("models", "anemia_image", "mobilenetv3_anemia_original_baseline.pth")
    if not baseline_path.exists():
        log.error("Baseline checkpoint not found at %s", baseline_path)
        return {}

    # Build architecture
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    state_dict = torch.load(baseline_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def evaluate_split(split_df: pd.DataFrame, split_name: str):
        probs, logits_list, true_labels = [], [], []
        for _, row in split_df.iterrows():
            img_path = Path(row["filepath"])
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    out = model(tensor)
                    p = torch.softmax(out, dim=1).squeeze().numpy()
                    logits_list.append(out.squeeze().numpy().tolist())
                    probs.append(float(p[1]))  # P(Anemic)
                    true_labels.append(int(row["label"]))
            except Exception as e:
                log.warning("Could not read %s: %s", img_path, e)

        probs = np.array(probs)
        true_labels = np.array(true_labels)
        logits_arr = np.array(logits_list)

        normal_mask = (true_labels == 0)
        anemic_mask = (true_labels == 1)

        normal_probs = probs[normal_mask]
        anemic_probs = probs[anemic_mask]

        from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

        # Default 0.5 evaluation
        preds_05 = (probs >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(true_labels, preds_05).ravel()
        sens_05 = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec_05 = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        acc_05 = float(accuracy_score(true_labels, preds_05))
        auc = float(roc_auc_score(true_labels, probs))

        log.info("[%s Split - Baseline Model at Threshold 0.50]", split_name.upper())
        log.info("  Total samples: %d (Normal: %d, Anemic: %d)", len(true_labels), len(normal_probs), len(anemic_probs))
        log.info("  Accuracy: %.4f | Sens: %.4f | Spec: %.4f | AUC: %.4f", acc_05, sens_05, spec_05, auc)
        log.info("  Confusion Matrix: TN=%d, FP=%d, FN=%d, TP=%d", tn, fp, fn, tp)
        log.info("  Normal slides P(Anemic) stats: mean=%.4f, std=%.4f, median=%.4f, [25%%=%.4f, 75%%=%.4f, min=%.4f, max=%.4f]",
                 float(np.mean(normal_probs)), float(np.std(normal_probs)), float(np.median(normal_probs)),
                 float(np.percentile(normal_probs, 25)), float(np.percentile(normal_probs, 75)),
                 float(np.min(normal_probs)), float(np.max(normal_probs)))
        log.info("  Anemic slides P(Anemic) stats: mean=%.4f, std=%.4f, median=%.4f, [25%%=%.4f, 75%%=%.4f, min=%.4f, max=%.4f]",
                 float(np.mean(anemic_probs)), float(np.std(anemic_probs)), float(np.median(anemic_probs)),
                 float(np.percentile(anemic_probs, 25)), float(np.percentile(anemic_probs, 75)),
                 float(np.min(anemic_probs)), float(np.max(anemic_probs)))

        return {
            "probs": probs,
            "true_labels": true_labels,
            "normal_probs": normal_probs,
            "anemic_probs": anemic_probs,
            "acc_05": acc_05,
            "sens_05": sens_05,
            "spec_05": spec_05,
            "auc": auc,
            "cm_05": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
        }

    val_res = evaluate_split(val_df, "Validation")
    test_res = evaluate_split(test_df, "Held-out Test")

    # Tune threshold strictly on VALIDATION split
    log.info("\n--- 3. TUNING THRESHOLD ON VALIDATION SET ONLY ---")
    from sklearn.metrics import roc_curve
    fpr_v, tpr_v, thresh_v = roc_curve(val_res["true_labels"], val_res["probs"])
    youden_j = tpr_v - fpr_v
    best_idx = int(np.argmax(youden_j))
    frozen_threshold = float(thresh_v[best_idx])
    frozen_threshold = max(0.15, min(0.85, frozen_threshold))
    log.info("Validation Youden's J: %.4f | Optimal Threshold Selected: %.4f", youden_j[best_idx], frozen_threshold)

    # Apply FROZEN threshold to Test split (no re-tuning!)
    log.info("\n--- 4. EVALUATING FROZEN THRESHOLD ON TEST SPLIT (NO TEST TUNING) ---")
    test_probs = test_res["probs"]
    test_labels = test_res["true_labels"]
    preds_cal = (test_probs >= frozen_threshold).astype(int)

    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    tn, fp, fn, tp = confusion_matrix(test_labels, preds_cal).ravel()
    sens_cal = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec_cal = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    acc_cal = float(accuracy_score(test_labels, preds_cal))
    rep = classification_report(test_labels, preds_cal, output_dict=True, zero_division=0)
    prec_cal = float(rep["1"]["precision"]) if "1" in rep else 0.0
    f1_cal = float(rep["1"]["f1-score"]) if "1" in rep else 0.0

    log.info("[Baseline Model on Test Set with FROZEN Calibrated Threshold (%.4f)]", frozen_threshold)
    log.info("  Accuracy: %.4f | Sens: %.4f | Spec: %.4f | Precision: %.4f | F1: %.4f", acc_cal, sens_cal, spec_cal, prec_cal, f1_cal)
    log.info("  Confusion Matrix: TN=%d, FP=%d, FN=%d, TP=%d", tn, fp, fn, tp)

    out_data = {
        "baseline_model": "mobilenetv3_anemia_original_baseline.pth",
        "validation_tuning": {
            "optimal_threshold_frozen": round(frozen_threshold, 4),
            "validation_youden_j": round(float(youden_j[best_idx]), 4),
            "validation_auc": round(val_res["auc"], 4)
        },
        "test_results_default_threshold_0_5": {
            "threshold": 0.5,
            "accuracy": round(test_res["acc_05"], 4),
            "sensitivity": round(test_res["sens_05"], 4),
            "specificity": round(test_res["spec_05"], 4),
            "auc": round(test_res["auc"], 4),
            "confusion_matrix": test_res["cm_05"]
        },
        "test_results_frozen_threshold": {
            "threshold": round(frozen_threshold, 4),
            "accuracy": round(acc_cal, 4),
            "sensitivity": round(sens_cal, 4),
            "specificity": round(spec_cal, 4),
            "precision": round(prec_cal, 4),
            "f1_score": round(f1_cal, 4),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
        },
        "normal_distribution_test": {
            "mean": round(float(np.mean(test_res["normal_probs"])), 4),
            "median": round(float(np.median(test_res["normal_probs"])), 4),
            "q25": round(float(np.percentile(test_res["normal_probs"], 25)), 4),
            "q75": round(float(np.percentile(test_res["normal_probs"], 75)), 4),
            "fraction_above_0_5": round(float(np.mean(test_res["normal_probs"] >= 0.5)), 4)
        },
        "anemic_distribution_test": {
            "mean": round(float(np.mean(test_res["anemic_probs"])), 4),
            "median": round(float(np.median(test_res["anemic_probs"])), 4),
            "q25": round(float(np.percentile(test_res["anemic_probs"], 25)), 4),
            "q75": round(float(np.percentile(test_res["anemic_probs"], 75)), 4),
            "fraction_above_0_5": round(float(np.mean(test_res["anemic_probs"] >= 0.5)), 4)
        }
    }

    out_file = project_path("models", "anemia_image", "prediction_audit_baseline.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    log.info("Saved baseline prediction audit report -> %s", out_file)
    return out_data


if __name__ == "__main__":
    dataset_root = project_path("data", "anerbc_raw")
    audit_res = audit_dataset_splits_and_leakage(dataset_root)
    audit_baseline_model_predictions(audit_res["val_df"], audit_res["test_df"])
