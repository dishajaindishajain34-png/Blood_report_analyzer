"""
train_anemia_image_cluster_split.py

Trains and evaluates MobileNetV3-Small on the remediated cluster-level split:
1. Manifest: models/anemia_image_cluster_split/cluster_split_manifest.csv
2. Checkpoint selection: based strictly on validation split loss.
3. Threshold calibration: based strictly on validation split Youden's J statistic.
4. Final evaluation: evaluated once on the new held-out test set (129 images).
5. Computes exact metrics, confusion matrix, and 95% bootstrap confidence intervals (B=1000).
6. Preserves existing models and artifacts; writes exclusively to models/anemia_image_cluster_split/.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, project_path

log = get_logger("train_cluster_split")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)


class ClusterSmearDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row["filepath"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=0)

        label = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, label


def bootstrap_ci(y_true, y_probs, threshold, n_bootstraps=1000, seed=42):
    """Compute 95% empirical bootstrap confidence intervals."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    accs, senss, specs, precs, f1s, aucs = [], [], [], [], [], []

    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        bt_true = y_true[boot_idx]
        bt_probs = y_probs[boot_idx]

        # Must have at least 1 positive and 1 negative sample
        if len(np.unique(bt_true)) < 2:
            continue

        bt_preds = (bt_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(bt_true, bt_preds, labels=[0, 1]).ravel()

        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        sens = tp / max(1, (tp + fn))
        spec = tn / max(1, (tn + fp))
        prec = tp / max(1, (tp + fp))
        f1 = 2 * prec * sens / max(1e-6, (prec + sens))

        try:
            auc = roc_auc_score(bt_true, bt_probs)
        except ValueError:
            auc = 0.5

        accs.append(acc)
        senss.append(sens)
        specs.append(spec)
        precs.append(prec)
        f1s.append(f1)
        aucs.append(auc)

    def ci(arr):
        return [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]

    return {
        "accuracy_ci_95": ci(accs),
        "sensitivity_ci_95": ci(senss),
        "specificity_ci_95": ci(specs),
        "precision_ci_95": ci(precs),
        "f1_ci_95": ci(f1s),
        "roc_auc_ci_95": ci(aucs),
    }


def train_and_evaluate(epochs=5, batch_size=16, lr=0.0005):
    manifest_path = project_path("models", "anemia_image_cluster_split", "cluster_split_manifest.csv")
    if not manifest_path.exists():
        log.error("Cluster split manifest not found at %s", manifest_path)
        return False

    df = pd.read_csv(manifest_path)
    train_df = df[df["cluster_split"] == "train"].copy()
    val_df = df[df["cluster_split"] == "val"].copy()
    test_df = df[df["cluster_split"] == "test"].copy()

    log.info("Loaded Cluster Split Data:")
    log.info("  Train : %d images (%d Normal, %d Anemic)", len(train_df), (train_df["label"] == 0).sum(), (train_df["label"] == 1).sum())
    log.info("  Val   : %d images (%d Normal, %d Anemic)", len(val_df), (val_df["label"] == 0).sum(), (val_df["label"] == 1).sum())
    log.info("  Test  : %d images (%d Normal, %d Anemic)", len(test_df), (test_df["label"] == 0).sum(), (test_df["label"] == 1).sum())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_ds = ClusterSmearDataset(train_df, transform=train_tf)
    val_ds = ClusterSmearDataset(val_df, transform=eval_tf)
    test_ds = ClusterSmearDataset(test_df, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Compute inverse class frequency weights
    n_0 = int((train_df["label"] == 0).sum())
    n_1 = int((train_df["label"] == 1).sum())
    n_tot = len(train_df)
    w_0 = float(n_tot / (2.0 * max(n_0, 1)))
    w_1 = float(n_tot / (2.0 * max(n_1, 1)))
    class_weights = torch.tensor([w_0, w_1], dtype=torch.float).to(device)
    log.info("Training class weights: Normal(0)=%.4f, Anemic(1)=%.4f", w_0, w_1)

    # Initialize MobileNetV3-Small
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_weights = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    out_dir = project_path("models", "anemia_image_cluster_split")
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting training loop (%d epochs)...", epochs)
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == lbls).sum().item()
            train_total += imgs.size(0)

        ep_tr_loss = train_loss / train_total
        ep_tr_acc = train_correct / train_total

        # Validation pass
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == lbls).sum().item()
                val_total += imgs.size(0)

        ep_val_loss = val_loss / val_total
        ep_val_acc = val_correct / val_total
        elapsed = time.time() - t0

        log.info("Epoch %d/%d (%.1fs) — Train Loss: %.4f, Acc: %.2f%% | Val Loss: %.4f, Acc: %.2f%%",
                 epoch + 1, epochs, elapsed, ep_tr_loss, ep_tr_acc * 100, ep_val_loss, ep_val_acc * 100)

        history["train_loss"].append(ep_tr_loss)
        history["train_acc"].append(ep_tr_acc)
        history["val_loss"].append(ep_val_loss)
        history["val_acc"].append(ep_val_acc)

        scheduler.step(ep_val_loss)

        if ep_val_loss < best_val_loss:
            best_val_loss = ep_val_loss
            best_weights = model.state_dict().copy()
            torch.save(best_weights, out_dir / "mobilenetv3_anemia_cluster_split.pth")
            log.info("Saved new best model weight (Val Loss: %.4f).", best_val_loss)

    if best_weights:
        model.load_state_dict(best_weights)
    model.eval()

    # Step 1: Calibrate threshold strictly on VALIDATION split
    log.info("Calibrating operating threshold strictly on validation split...")
    val_probs_list, val_labels_list = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=1)[:, 1].cpu().numpy()
            val_probs_list.extend(probs)
            val_labels_list.extend(lbls.numpy())

    val_probs_arr = np.array(val_probs_list)
    val_labels_arr = np.array(val_labels_list)

    fpr_v, tpr_v, th_v = roc_curve(val_labels_arr, val_probs_arr)
    youden_j = tpr_v - fpr_v
    best_idx = int(np.argmax(youden_j))
    optimal_threshold = float(th_v[best_idx])
    optimal_threshold = max(0.15, min(0.85, optimal_threshold))
    val_auc = float(roc_auc_score(val_labels_arr, val_probs_arr))

    log.info("Validation Youden's J: %.4f | Optimal Frozen Threshold: %.4f | Val AUC: %.4f",
             youden_j[best_idx], optimal_threshold, val_auc)

    # Step 2: Evaluate frozen model ONCE on the held-out TEST split
    log.info("Evaluating frozen model on held-out test split (129 images)...")
    test_probs_list, test_labels_list, test_fnames = [], [], []
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(test_loader):
            imgs = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=1)[:, 1].cpu().numpy()
            test_probs_list.extend(probs)
            test_labels_list.extend(lbls.numpy())

    test_probs = np.array(test_probs_list)
    test_labels = np.array(test_labels_list)
    test_auc = float(roc_auc_score(test_labels, test_probs))

    # Evaluate at default 0.50
    preds_05 = (test_probs >= 0.50).astype(int)
    tn_05, fp_05, fn_05, tp_05 = confusion_matrix(test_labels, preds_05).ravel()
    sens_05 = tp_05 / (tp_05 + fn_05)
    spec_05 = tn_05 / (tn_05 + fp_05)
    acc_05 = accuracy_score(test_labels, preds_05)
    f1_05 = precision_recall_fscore_support(test_labels, preds_05, average="binary", zero_division=0)[2]

    # Evaluate at calibrated threshold
    preds_cal = (test_probs >= optimal_threshold).astype(int)
    tn_c, fp_c, fn_c, tp_c = confusion_matrix(test_labels, preds_cal).ravel()
    sens_c = tp_c / (tp_c + fn_c)
    spec_c = tn_c / (tn_c + fp_c)
    acc_c = accuracy_score(test_labels, preds_cal)
    prec_c, _, f1_c, _ = precision_recall_fscore_support(test_labels, preds_cal, average="binary", zero_division=0)

    # Compute 95% Bootstrap Confidence Intervals
    log.info("Computing 95% bootstrap confidence intervals (B=1000)...")
    ci_dict = bootstrap_ci(test_labels, test_probs, optimal_threshold, n_bootstraps=1000, seed=42)

    log.info("\n" + "="*80)
    log.info("REMEDIATED CLUSTER-SPLIT TEST EVALUATION RESULTS")
    log.info("="*80)
    log.info("Test set: %d images (%d Normal, %d Anemic)", len(test_labels), (test_labels == 0).sum(), (test_labels == 1).sum())
    log.info("Default Cutoff (0.50)  : Acc=%.4f, Sens=%.4f, Spec=%.4f, F1=%.4f, CM=[TN=%d, FP=%d, FN=%d, TP=%d]",
             acc_05, sens_05, spec_05, f1_05, tn_05, fp_05, fn_05, tp_05)
    log.info("Calibrated Cutoff (%.4f): Acc=%.4f, Sens=%.4f, Spec=%.4f, Prec=%.4f, F1=%.4f",
             optimal_threshold, acc_c, sens_c, spec_c, prec_c, f1_c)
    log.info("Calibrated CM           : TN=%d, FP=%d, FN=%d, TP=%d", tn_c, fp_c, fn_c, tp_c)
    log.info("ROC-AUC                 : %.4f", test_auc)
    log.info("95%% Bootstrap CIs      : Sens=%s, Spec=%s, Acc=%s, AUC=%s",
             ci_dict["sensitivity_ci_95"], ci_dict["specificity_ci_95"], ci_dict["accuracy_ci_95"], ci_dict["roc_auc_ci_95"])
    log.info("="*80 + "\n")

    # Save test predictions CSV
    test_pred_records = []
    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        pb = float(test_probs[idx])
        test_pred_records.append({
            "filename": row["filename"],
            "cluster_id": row["cluster_id"],
            "true_label": int(row["label"]),
            "class_name": row["class_name"],
            "predicted_prob_anemic": round(pb, 4),
            "pred_default_05": int(pb >= 0.50),
            "pred_calibrated": int(pb >= optimal_threshold),
        })
    test_preds_df = pd.DataFrame(test_pred_records)
    test_preds_path = out_dir / "test_predictions.csv"
    test_preds_df.to_csv(test_preds_path, index=False)
    log.info("Saved test predictions -> %s", test_preds_path)

    # Save metrics JSON
    metrics_data = {
        "experiment_name": "Cluster-Level Leakage-Aware Remediation",
        "split_manifest": "models/anemia_image_cluster_split/cluster_split_manifest.csv",
        "checkpoint": "models/anemia_image_cluster_split/mobilenetv3_anemia_cluster_split.pth",
        "optimal_threshold": round(optimal_threshold, 4),
        "validation_metrics": {
            "val_youden_j": round(float(youden_j[best_idx]), 4),
            "val_roc_auc": round(val_auc, 4),
        },
        "test_counts": {
            "total": len(test_labels),
            "normal": int((test_labels == 0).sum()),
            "anemic": int((test_labels == 1).sum()),
        },
        "test_calibrated_metrics": {
            "threshold": round(optimal_threshold, 4),
            "accuracy": round(acc_c, 4),
            "accuracy_ci_95": ci_dict["accuracy_ci_95"],
            "sensitivity": round(sens_c, 4),
            "sensitivity_ci_95": ci_dict["sensitivity_ci_95"],
            "specificity": round(spec_c, 4),
            "specificity_ci_95": ci_dict["specificity_ci_95"],
            "precision": round(prec_c, 4),
            "precision_ci_95": ci_dict["precision_ci_95"],
            "f1_score": round(f1_c, 4),
            "f1_ci_95": ci_dict["f1_ci_95"],
            "roc_auc": round(test_auc, 4),
            "roc_auc_ci_95": ci_dict["roc_auc_ci_95"],
            "confusion_matrix": {"TN": int(tn_c), "FP": int(fp_c), "FN": int(fn_c), "TP": int(tp_c)},
        },
        "test_default_metrics": {
            "threshold": 0.50,
            "accuracy": round(acc_05, 4),
            "sensitivity": round(sens_05, 4),
            "specificity": round(spec_05, 4),
            "f1_score": round(f1_05, 4),
            "confusion_matrix": {"TN": int(tn_05), "FP": int(fp_05), "FN": int(fn_05), "TP": int(tp_05)},
        },
        "training_history": history,
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    log.info("Saved metrics -> %s", metrics_path)

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array([[tn_c, fp_c], [fn_c, tp_c]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anemic"])
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title(f"Remediated Cluster Split Test Confusion Matrix\nSens: {sens_c:.2f}, Spec: {spec_c:.2f} (Thresh: {optimal_threshold:.2f})")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # Compare with existing results
    compare_with_existing(metrics_data, out_dir)
    return True


def compare_with_existing(new_metrics: dict, out_dir: Path):
    old_metrics_path = project_path("models", "anemia_image", "independent_verification.json")
    if not old_metrics_path.exists():
        log.warning("Original verification metrics not found at %s", old_metrics_path)
        return

    with open(old_metrics_path, "r", encoding="utf-8") as f:
        old_data = json.load(f)

    old_weighted = old_data.get("weighted_retrained_model", {}).get("calibrated_metrics", {})
    old_baseline = old_data.get("baseline_model", {}).get("calibrated_metrics", {})
    new_cal = new_metrics["test_calibrated_metrics"]

    comparison = {
        "description": "Comparison between old unclustered split and remediated cluster-level split",
        "caveat": "Differences reflect the elimination of cross-split duplicate/same-slide frames and altered split composition. Performance should be viewed as a more honest, leakage-resistant benchmark rather than a direct algorithmic gain.",
        "metrics_table": {
            "unclustered_baseline_calibrated": {
                "threshold": old_baseline.get("threshold"),
                "accuracy": old_baseline.get("accuracy"),
                "sensitivity": old_baseline.get("sensitivity"),
                "specificity": old_baseline.get("specificity"),
                "f1_score": old_baseline.get("f1_score"),
                "confusion_matrix": old_baseline.get("confusion_matrix")
            },
            "unclustered_weighted_calibrated": {
                "threshold": old_weighted.get("threshold"),
                "accuracy": old_weighted.get("accuracy"),
                "sensitivity": old_weighted.get("sensitivity"),
                "specificity": old_weighted.get("specificity"),
                "f1_score": old_weighted.get("f1_score"),
                "confusion_matrix": old_weighted.get("confusion_matrix")
            },
            "remediated_cluster_split_calibrated": {
                "threshold": new_cal.get("threshold"),
                "accuracy": new_cal.get("accuracy"),
                "accuracy_ci_95": new_cal.get("accuracy_ci_95"),
                "sensitivity": new_cal.get("sensitivity"),
                "sensitivity_ci_95": new_cal.get("sensitivity_ci_95"),
                "specificity": new_cal.get("specificity"),
                "specificity_ci_95": new_cal.get("specificity_ci_95"),
                "precision": new_cal.get("precision"),
                "precision_ci_95": new_cal.get("precision_ci_95"),
                "f1_score": new_cal.get("f1_score"),
                "f1_ci_95": new_cal.get("f1_ci_95"),
                "roc_auc": new_cal.get("roc_auc"),
                "roc_auc_ci_95": new_cal.get("roc_auc_ci_95"),
                "confusion_matrix": new_cal.get("confusion_matrix")
            }
        }
    }

    comp_path = out_dir / "evaluation_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    log.info("Saved comparative evaluation report -> %s", comp_path)


if __name__ == "__main__":
    train_and_evaluate(epochs=5, batch_size=16, lr=0.0005)
