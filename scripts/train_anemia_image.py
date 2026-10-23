"""
train_anemia_image.py

Trains a deep learning image classifier (MobileNetV3-Small) on the AneRBC dataset.
Supports both CPU and GPU (if CUDA is available).
Applies stratified group-level splitting by patient/sample ID to prevent data leakage.
Applies inverse class frequency weighting to address class imbalance.
Calibrates operating decision threshold using Youden's J statistic on the validation split.
Generates metrics, training history, ROC curves, and confusion matrices.

Usage:
    python scripts/train_anemia_image.py
    python scripts/train_anemia_image.py --root data/anerbc_raw --epochs 5 --batch 16
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, project_path

log = get_logger("train_anemia_image")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


seed_everything(42)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_all_images(root: Path) -> list[Path]:
    images = []
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            images.append(p)
    return sorted(images)


def detect_label_from_path(path: Path) -> int | None:
    """Infer label: 0=Normal, 1=Anemic, None if unknown."""
    parts = [p.lower() for p in path.parts]
    anemic_keywords = {"anemia", "anemic", "anaemic", "disease", "anerbc_ii_anemic", "anerbc_i_anemic"}
    normal_keywords = {"normal", "healthy", "control", "anerbc_ii_normal", "anerbc_i_normal"}

    for part in reversed(parts[:-1]):
        if part in anemic_keywords:
            return 1
        if part in normal_keywords:
            return 0
        if any(k in part for k in anemic_keywords):
            return 1
        if any(k in part for k in normal_keywords):
            return 0

    stem = path.stem.lower()
    if any(k in stem for k in anemic_keywords):
        return 1
    if any(k in stem for k in normal_keywords):
        return 0

    return None


def get_patient_id(path: Path) -> str:
    """Extract patient/sample ID from filename if possible, otherwise return filename stem."""
    stem = path.stem
    m = re.match(r'^([A-Za-z]*\d+)_', stem)
    if m:
        return m.group(1)
    return stem


def load_dataset_metadata(root: Path) -> pd.DataFrame:
    """Scan root, detect files, labels, patient IDs and return a DataFrame with duplicate removal."""
    images = find_all_images(root)
    records = []
    for img in images:
        label = detect_label_from_path(img)
        if label is None:
            continue
        pid = get_patient_id(img)
        records.append({
            "filepath": str(img),
            "filename": img.name,
            "patient_id": pid,
            "label": label
        })
    df = pd.DataFrame(records)
    if not df.empty:
        log.info("Checking for duplicate images across %d files...", len(df))
        import hashlib
        hashes = []
        for fp in df["filepath"]:
            with open(fp, "rb") as f:
                hashes.append(hashlib.md5(f.read()).hexdigest())
        df["md5"] = hashes
        before = len(df)
        df = df.drop_duplicates(subset=["md5"]).copy()
        after = len(df)
        if before > after:
            log.info("Removed %d duplicate images. Total unique images: %d", before - after, after)
    return df


def split_dataset(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified group-level split by patient_id and class label: 70% train / 15% val / 15% test."""
    rng = random.Random(seed)

    patient_label = df.groupby("patient_id")["label"].agg(lambda s: s.mode()[0]).to_dict()

    train_pids, val_pids, test_pids = set(), set(), set()

    for lbl in sorted(df["label"].unique()):
        pids_for_lbl = [pid for pid, l in patient_label.items() if l == lbl]
        rng.shuffle(pids_for_lbl)
        n = len(pids_for_lbl)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        train_pids.update(pids_for_lbl[:n_train])
        val_pids.update(pids_for_lbl[n_train:n_train + n_val])
        test_pids.update(pids_for_lbl[n_train + n_val:])

    train_df = df[df["patient_id"].isin(train_pids)].copy()
    val_df = df[df["patient_id"].isin(val_pids)].copy()
    test_df = df[df["patient_id"].isin(test_pids)].copy()

    log.info("Split summary (Stratified by Patient Group & Label):")
    log.info(
        "  Train: %d images (%d Normal, %d Anemic | %d patients)",
        len(train_df), (train_df['label'] == 0).sum(), (train_df['label'] == 1).sum(), len(train_pids)
    )
    log.info(
        "  Val:   %d images (%d Normal, %d Anemic | %d patients)",
        len(val_df), (val_df['label'] == 0).sum(), (val_df['label'] == 1).sum(), len(val_pids)
    )
    log.info(
        "  Test:  %d images (%d Normal, %d Anemic | %d patients)",
        len(test_df), (test_df['label'] == 0).sum(), (test_df['label'] == 1).sum(), len(test_pids)
    )

    return train_df, val_df, test_df


try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms

    class AneRBCDataset(Dataset):
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
except ImportError:
    pass


def train_model(args):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as models
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, roc_curve, ConfusionMatrixDisplay

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    root_path = Path(args.root)
    df = load_dataset_metadata(root_path)
    if df.empty:
        log.error("No valid labeled images found in %s.", root_path)
        return False

    train_df, val_df, test_df = split_dataset(df)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_size = args.imgsz

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = AneRBCDataset(train_df, transform=train_transform)
    val_dataset = AneRBCDataset(val_df, transform=val_transform)
    test_dataset = AneRBCDataset(test_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    log.info("Loading pretrained MobileNetV3-Small backbone...")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    model.to(device)

    # Compute inverse class frequency weights on training set to handle class imbalance
    train_labels = train_df["label"].values
    n_0 = int(np.sum(train_labels == 0))
    n_1 = int(np.sum(train_labels == 1))
    n_total = len(train_labels)
    w_0 = float(n_total / (2.0 * max(n_0, 1)))
    w_1 = float(n_total / (2.0 * max(n_1, 1)))
    class_weights = torch.tensor([w_0, w_1], dtype=torch.float).to(device)
    log.info("Class distribution in Train — Normal(0): %d, Anemic(1): %d | Weights: [%.4f, %.4f]", n_0, n_1, w_0, w_1)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    epochs = args.epochs
    best_val_loss = float("inf")
    best_weights = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    models_dir = project_path("models", "anemia_image")
    models_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting training loop (%d epochs)...", epochs)
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss, train_correct = 0.0, 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            total_train += images.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_correct / total_train

        model.eval()
        val_loss, val_correct = 0.0, 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                total_val += images.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = val_correct / total_val
        elapsed = time.time() - t0

        log.info(
            "Epoch %d/%d (%.1fs) — Train Loss: %.4f, Acc: %.2f%% | Val Loss: %.4f, Acc: %.2f%%",
            epoch + 1, epochs, elapsed, epoch_train_loss, epoch_train_acc * 100,
            epoch_val_loss, epoch_val_acc * 100
        )

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = model.state_dict().copy()
            torch.save(best_weights, models_dir / "mobilenetv3_anemia.pth")
            log.info("Saved new best model weight (Val Loss: %.4f).", best_val_loss)

    if best_weights:
        model.load_state_dict(best_weights)
    model.eval()

    # Step 1: Calibrate optimal threshold on VALIDATION set using Youden's J statistic
    log.info("Calibrating operating threshold on validation split via Youden's Index...")
    val_probs_list, val_labels_list = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            val_probs_list.extend(probs)
            val_labels_list.extend(labels.numpy())

    val_probs_arr = np.array(val_probs_list)
    val_labels_arr = np.array(val_labels_list)

    fpr_val, tpr_val, thresh_val = roc_curve(val_labels_arr, val_probs_arr)
    # Youden's J = Sensitivity + Specificity - 1 = TPR - FPR
    j_scores = tpr_val - fpr_val
    best_idx = int(np.argmax(j_scores))
    optimal_threshold = float(thresh_val[best_idx])
    optimal_threshold = max(0.15, min(0.85, optimal_threshold))
    log.info(
        "Validation Youden's J: %.4f | Optimal Operating Threshold: %.4f (Val TPR: %.4f, FPR: %.4f)",
        j_scores[best_idx], optimal_threshold, tpr_val[best_idx], fpr_val[best_idx]
    )

    # Step 2: Evaluate on HELD-OUT TEST SPLIT
    log.info("Evaluating best model on held-out test split (15%%)...")
    test_preds_raw, test_labels_list, test_probs_list = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            _, preds = torch.max(outputs, 1)

            test_preds_raw.extend(preds.cpu().numpy())
            test_labels_list.extend(labels.numpy())
            test_probs_list.extend(probs)

    test_labels = np.array(test_labels_list)
    test_probs = np.array(test_probs_list)

    try:
        auc = float(roc_auc_score(test_labels, test_probs))
    except Exception:
        auc = 0.5

    def eval_at_threshold(threshold_value: float):
        preds = (test_probs >= threshold_value).astype(int)
        acc = float(accuracy_score(test_labels, preds))
        rep = classification_report(test_labels, preds, output_dict=True, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(test_labels, preds).ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(rep["1"]["precision"]) if "1" in rep else 0.0
        f1 = float(rep["1"]["f1-score"]) if "1" in rep else 0.0
        return {
            "threshold": round(threshold_value, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall_sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp)
            }
        }

    default_eval = eval_at_threshold(0.5)
    calibrated_eval = eval_at_threshold(optimal_threshold)

    log.info("--- HELD-OUT TEST RESULTS ---")
    log.info(
        "Default Threshold (0.50): Acc=%.4f, Sens=%.4f, Spec=%.4f, F1=%.4f | CM=[TN=%d, FP=%d, FN=%d, TP=%d]",
        default_eval["accuracy"], default_eval["recall_sensitivity"], default_eval["specificity"],
        default_eval["f1_score"], default_eval["confusion_matrix"]["TN"], default_eval["confusion_matrix"]["FP"],
        default_eval["confusion_matrix"]["FN"], default_eval["confusion_matrix"]["TP"]
    )
    log.info(
        "Calibrated Threshold (%.4f): Acc=%.4f, Sens=%.4f, Spec=%.4f, F1=%.4f | CM=[TN=%d, FP=%d, FN=%d, TP=%d]",
        optimal_threshold, calibrated_eval["accuracy"], calibrated_eval["recall_sensitivity"],
        calibrated_eval["specificity"], calibrated_eval["f1_score"],
        calibrated_eval["confusion_matrix"]["TN"], calibrated_eval["confusion_matrix"]["FP"],
        calibrated_eval["confusion_matrix"]["FN"], calibrated_eval["confusion_matrix"]["TP"]
    )
    log.info("Held-out Test ROC-AUC: %.4f", auc)

    metrics = {
        "model_name": "MobileNetV3-Small (AneRBC)",
        "model_version": "1.1.0",
        "optimal_threshold": round(optimal_threshold, 4),
        "roc_auc": round(auc, 4),
        "accuracy": calibrated_eval["accuracy"],
        "precision": calibrated_eval["precision"],
        "recall_sensitivity": calibrated_eval["recall_sensitivity"],
        "specificity": calibrated_eval["specificity"],
        "f1_score": calibrated_eval["f1_score"],
        "confusion_matrix": calibrated_eval["confusion_matrix"],
        "calibrated_metrics": calibrated_eval,
        "default_metrics": default_eval,
        "class_weights": {"0_Normal": round(w_0, 4), "1_Anemic": round(w_1, 4)},
        "sample_counts": {
            "total": len(df),
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "test_normal": int((test_df['label'] == 0).sum()),
            "test_anemic": int((test_df['label'] == 1).sum())
        }
    }

    with open(models_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(models_dir / "preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "img_size": img_size,
            "mean": mean,
            "std": std
        }, f, indent=2)

    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "MobileNetV3-Small (AneRBC)",
            "model_version": "1.1.0",
            "num_classes": 2,
            "optimal_threshold": round(optimal_threshold, 4),
            "training_history": history
        }, f, indent=2)

    with open(models_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"0": "Normal", "1": "Anemic"}, f, indent=2)

    # Plot Loss & Acc history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss History")

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy History")
    plt.savefig(models_dir / "training_history.png", bbox_inches='tight')
    plt.close()

    # Plot Confusion Matrix (Calibrated) using sklearn's ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array([
        [calibrated_eval["confusion_matrix"]["TN"], calibrated_eval["confusion_matrix"]["FP"]],
        [calibrated_eval["confusion_matrix"]["FN"], calibrated_eval["confusion_matrix"]["TP"]]
    ])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anemic"])
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title(f"Calibrated Confusion Matrix (Test Split)\nSens: {calibrated_eval['recall_sensitivity']:.2f}, Spec: {calibrated_eval['specificity']:.2f}")
    plt.tight_layout()
    plt.savefig(models_dir / "confusion_matrix.png", bbox_inches='tight')
    plt.close()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend()
    plt.savefig(models_dir / "roc_curve.png", bbox_inches='tight')
    plt.close()

    log.info("Training complete! Model and calibrated metrics saved.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/anerbc_raw")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.0005)
    args = parser.parse_args()

    success = train_model(args)
    if not success:
        sys.exit(1)
