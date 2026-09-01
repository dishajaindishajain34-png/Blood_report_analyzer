"""
train_anemia_image.py

Trains a deep learning image classifier (MobileNetV3-Small) on the AneRBC dataset.
Supports both CPU and GPU (if CUDA is available).
Applies group-level splitting by patient/sample ID if available to prevent data leakage.
Generates metrics, training history, ROC curves, and confusion matrices.

Usage:
    python scripts/train_anemia_image.py
    python scripts/train_anemia_image.py --root data/anerbc_raw --epochs 10 --batch 32
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

# Setup random seeds for reproducibility
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
    anemic_keywords  = {"anemia", "anemic", "anaemic", "disease", "anerbc_ii_anemic", "anerbc_i_anemic"}
    normal_keywords  = {"normal", "healthy", "control", "anerbc_ii_normal", "anerbc_i_normal"}

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
    """Scan root, detect files, labels, patient IDs and return a DataFrame."""
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
        print("Checking for duplicate images...")
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
            log.info("Removed %d duplicate images. Total images: %d", before - after, after)
    return df


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group-level split by patient_id: 70% train / 15% val / 15% test."""
    pids = sorted(df["patient_id"].unique())
    random.shuffle(pids)

    n_pids = len(pids)
    n_train = int(0.7 * n_pids)
    n_val = int(0.15 * n_pids)

    train_pids = set(pids[:n_train])
    val_pids = set(pids[n_train:n_train + n_val])
    test_pids = set(pids[n_train + n_val:])

    train_df = df[df["patient_id"].isin(train_pids)].copy()
    val_df = df[df["patient_id"].isin(val_pids)].copy()
    test_df = df[df["patient_id"].isin(test_pids)].copy()

    log.info("Split summary (by patient group):")
    log.info("  Train: %d images (%d patients)", len(train_df), len(train_pids))
    log.info("  Val:   %d images (%d patients)", len(val_df), len(val_pids))
    log.info("  Test:  %d images (%d patients)", len(test_df), len(test_pids))

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
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score

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
        transforms.RandomRotation(10),
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

    log.info("Loading pretrained MobileNetV3-Small...")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
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
            log.info("Saved new best model weight.")

    log.info("Evaluating best model on held-out test split...")
    if best_weights:
        model.load_state_dict(best_weights)
    model.eval()

    test_preds, test_labels, test_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            _, preds = torch.max(outputs, 1)

            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.numpy())
            test_probs.extend(probs)

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_probs = np.array(test_probs)

    acc = accuracy_score(test_labels, test_preds)
    report = classification_report(test_labels, test_preds, output_dict=True)
    try:
        auc = roc_auc_score(test_labels, test_probs)
    except Exception:
        auc = 0.5

    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "model_name": "MobileNetV3-Small (AneRBC)",
        "model_version": "1.0.0",
        "accuracy": round(acc, 4),
        "precision": round(report["1"]["precision"], 4) if "1" in report else 0.0,
        "recall_sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(report["1"]["f1-score"], 4) if "1" in report else 0.0,
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },
        "sample_counts": {
            "total": len(df),
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
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
            "model_version": "1.0.0",
            "num_classes": 2,
            "training_history": history
        }, f, indent=2)

    with open(models_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"0": "Normal", "1": "Anemic"}, f, indent=2)

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

    import seaborn as sns
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(test_labels, test_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anemic"], yticklabels=["Normal", "Anemic"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix (Test Split)")
    plt.savefig(models_dir / "confusion_matrix.png", bbox_inches='tight')
    plt.close()

    from sklearn.metrics import roc_curve
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

    log.info("Training complete! Model saved.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/anerbc_raw")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    success = train_model(args)
    if not success:
        sys.exit(1)
