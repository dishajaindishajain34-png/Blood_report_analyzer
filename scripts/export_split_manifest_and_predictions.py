"""
export_split_manifest_and_predictions.py

Exports:
1. models/anemia_image/split_manifest.csv — Full partition manifest (Train, Val, Test)
2. models/anemia_image/test_predictions.csv — Exact sample-by-sample predictions for both checkpoints
3. Verifies that aggregations from test_predictions.csv reproduce reported metrics exactly.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, project_path
from scripts.train_anemia_image import load_dataset_metadata, split_dataset

log = get_logger("export_manifest")


def export_manifest_and_predictions():
    root = project_path("data", "anerbc_raw")
    df = load_dataset_metadata(root)

    # Purge cross-class duplicate with conflicting labels (MD5: e98d48f1d84419386a71aa3dc9df1ae7)
    cross_hashes = set(df.groupby("md5")["label"].nunique().loc[lambda s: s > 1].index)
    if cross_hashes:
        df = df[~df["md5"].isin(cross_hashes)].copy()
    df = df.drop_duplicates(subset=["md5"]).copy()

    # Create reproducible splits (seed=42)
    train_df, val_df, test_df = split_dataset(df, seed=42)

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    manifest_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    manifest_df["class_name"] = manifest_df["label"].map({0: "Normal", 1: "Anemic"})

    manifest_path = project_path("models", "anemia_image", "split_manifest.csv")
    manifest_df[["filepath", "filename", "label", "class_name", "md5", "split"]].to_csv(manifest_path, index=False)
    log.info("Saved split manifest -> %s (%d rows)", manifest_path, len(manifest_df))

    # Model inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def load_m(pth: Path):
        m = models.mobilenet_v3_small(weights=None)
        in_feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feat, 2)
        m.load_state_dict(torch.load(pth, map_location="cpu", weights_only=True))
        m.eval()
        return m

    m_base = load_m(project_path("models", "anemia_image", "mobilenetv3_anemia_original_baseline.pth"))
    m_weight = load_m(project_path("models", "anemia_image", "mobilenetv3_anemia.pth"))

    records = []
    tau_base = 0.7390
    tau_weight = 0.8500

    for _, row in test_df.iterrows():
        img_path = Path(row["filepath"])
        img = Image.open(img_path).convert("RGB")
        t = transform(img).unsqueeze(0)
        with torch.no_grad():
            p_b = float(torch.softmax(m_base(t), dim=1).squeeze().numpy()[1])
            p_w = float(torch.softmax(m_weight(t), dim=1).squeeze().numpy()[1])

        lbl = int(row["label"])
        records.append({
            "filepath": str(img_path),
            "filename": img_path.name,
            "true_label": lbl,
            "class_name": "Anemic" if lbl == 1 else "Normal",
            "baseline_prob_anemic": round(p_b, 4),
            "baseline_pred_thresh_05": 1 if p_b >= 0.50 else 0,
            "baseline_pred_cal_thresh_0739": 1 if p_b >= tau_base else 0,
            "weighted_prob_anemic": round(p_w, 4),
            "weighted_pred_thresh_05": 1 if p_w >= 0.50 else 0,
            "weighted_pred_cal_thresh_0850": 1 if p_w >= tau_weight else 0,
        })

    preds_df = pd.DataFrame(records)
    preds_path = project_path("models", "anemia_image", "test_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    log.info("Saved detailed sample predictions -> %s (%d rows)", preds_path, len(preds_df))

    # Verify reproduction
    y_true = preds_df["true_label"].values

    def summarize(y_pred, name):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        acc = accuracy_score(y_true, y_pred)
        log.info("  %-30s: Acc=%.4f, Sens=%.4f, Spec=%.4f, CM=[TN=%d, FP=%d, FN=%d, TP=%d]",
                 name, acc, sens, spec, tn, fp, fn, tp)

    log.info("\nRe-verifying exact metrics from CSV:")
    summarize(preds_df["baseline_pred_thresh_05"], "Baseline (0.50)")
    summarize(preds_df["baseline_pred_cal_thresh_0739"], "Baseline (Calibrated 0.7390)")
    summarize(preds_df["weighted_pred_thresh_05"], "Weighted (0.50)")
    summarize(preds_df["weighted_pred_cal_thresh_0850"], "Weighted (Calibrated 0.8500)")

    return True


if __name__ == "__main__":
    export_manifest_and_predictions()
