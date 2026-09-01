"""
train_resnet_phase2.py — ResNet50 WBC Subtype Classifier Training (Phase 2).

Uses the dataset prepared by download_wbc_dataset.py.
Technique:
  - Start from ImageNet-pretrained ResNet50
  - Phase A: Freeze backbone, train head only (5 epochs) — fast convergence
  - Phase B: Unfreeze all layers, fine-tune with low LR (remaining epochs)
  - Cosine annealing LR schedule
  - Class-weighted cross-entropy loss (handles class imbalance in pseudo-labels)
  - Saves best model to models/resnet50_wbc/wbc_classifier_tinycnn.pth

Usage:
    python training/train_resnet_phase2.py [--epochs 30] [--batch 16]
"""

import sys, argparse, json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
import numpy as np

from src.utils import get_logger, ensure_dir, project_path

log = get_logger("train_resnet_phase2")

WBC_CLASSES  = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]
DATA_ROOT    = project_path("data", "wbc_classifier")
MODEL_OUT    = project_path("models", "resnet50_wbc", "wbc_classifier_tinycnn.pth")
HISTORY_OUT  = project_path("models", "resnet50_wbc", "training_history.json")
IMG_SIZE     = 64   # TinyCNN input size — small enough for fast CPU training
MEAN         = [0.485, 0.456, 0.406]
STD          = [0.229, 0.224, 0.225]


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, val_tf


def make_weighted_sampler(dataset):
    """Return a WeightedRandomSampler to handle class imbalance."""
    class_counts = np.array([0] * len(WBC_CLASSES))
    for _, label in dataset.samples:
        class_counts[label] += 1
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


class TinyCNN(nn.Module):
    """
    Lightweight 4-block CNN for WBC subtype classification.
    ~220K parameters — designed for fast CPU training.
    Input: (B, 3, 64, 64)  Output: (B, n_classes)
    """
    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4: 8x8 -> 4x4
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(n_classes: int = 5) -> nn.Module:
    model = TinyCNN(n_classes=n_classes)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("TinyCNN: %s parameters", f"{n_params:,}")
    return model


def freeze_backbone(model: nn.Module):
    """Freeze CNN feature blocks, keep classifier trainable."""
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def train_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs     = model(images)
            loss        = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


def main(epochs: int = 30, batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_tf, val_tf = build_transforms()
    train_ds = datasets.ImageFolder(str(DATA_ROOT / "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(DATA_ROOT / "val"),   transform=val_tf)
    log.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    sampler  = make_weighted_sampler(train_ds)
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                          num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(n_classes=len(WBC_CLASSES)).to(device)

    # Class-weighted loss
    class_counts = np.array([0] * len(WBC_CLASSES))
    for _, lbl in train_ds.samples:
        class_counts[lbl] += 1
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Phase A: Train head only ───────────────────────────────────────────────
    phase_a_epochs = min(5, epochs // 4)
    log.info("Phase A: Training head only for %d epochs", phase_a_epochs)
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase_a_epochs, eta_min=1e-5
    )

    history   = []
    best_acc  = 0.0
    ensure_dir(MODEL_OUT.parent)

    for ep in range(phase_a_epochs):
        t_loss, t_acc = train_epoch(model, train_dl, criterion, optimizer, device)
        v_loss, v_acc = eval_epoch(model,  val_dl,   criterion, device)
        scheduler.step()
        log.info("[A] Epoch %02d/%02d | train acc=%.3f loss=%.4f | val acc=%.3f loss=%.4f",
                 ep+1, phase_a_epochs, t_acc, t_loss, v_acc, v_loss)
        history.append({"phase": "A", "epoch": ep+1, "t_acc": t_acc, "v_acc": v_acc,
                        "t_loss": t_loss, "v_loss": v_loss})
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), str(MODEL_OUT))
            log.info("  * Best val acc: %.3f — saved model", best_acc)

    # ── Phase B: Fine-tune all layers ─────────────────────────────────────────
    phase_b_epochs = epochs - phase_a_epochs
    log.info("Phase B: Fine-tuning all layers for %d epochs", phase_b_epochs)
    unfreeze_all(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase_b_epochs, eta_min=1e-6
    )

    for ep in range(phase_b_epochs):
        t_loss, t_acc = train_epoch(model, train_dl, criterion, optimizer, device)
        v_loss, v_acc = eval_epoch(model,  val_dl,   criterion, device)
        scheduler.step()
        log.info("[B] Epoch %02d/%02d | train acc=%.3f loss=%.4f | val acc=%.3f loss=%.4f",
                 ep+1, phase_b_epochs, t_acc, t_loss, v_acc, v_loss)
        history.append({"phase": "B", "epoch": ep+1, "t_acc": t_acc, "v_acc": v_acc,
                        "t_loss": t_loss, "v_loss": v_loss})
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), str(MODEL_OUT))
            log.info("  * Best val acc: %.3f — saved model", best_acc)

    # ── Save history ─────────────────────────────────────────────────────────
    HISTORY_OUT.write_text(json.dumps({
        "best_val_acc": best_acc,
        "epochs":       epochs,
        "classes":      WBC_CLASSES,
        "trained_at":   datetime.now().isoformat(),
        "history":      history,
    }, indent=2))

    log.info("Training complete. Best val acc: %.3f", best_acc)
    log.info("Model saved: %s", MODEL_OUT)
    print(f"\nBest validation accuracy: {best_acc:.1%}")
    print(f"Model: {MODEL_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet50 WBC Classifier (Phase 2)")
    parser.add_argument("--epochs",    type=int, default=30, help="Total training epochs")
    parser.add_argument("--batch",     type=int, default=16, help="Batch size")
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch)
