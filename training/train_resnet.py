"""
train_resnet.py — Fine-tune ResNet50 for WBC subtype classification (Phase 2).

Classes (5):
  0: Neutrophil
  1: Lymphocyte
  2: Monocyte
  3: Eosinophil
  4: Basophil

Expected dataset layout (ImageFolder format):
  data/wbc_classifier/
    train/
      Neutrophil/   *.jpg
      Lymphocyte/   *.jpg
      Monocyte/     *.jpg
      Eosinophil/   *.jpg
      Basophil/     *.jpg
    val/
      (same structure)

Usage:
    python training/train_resnet.py [--epochs 30] [--batch 32] [--lr 1e-4]
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, ensure_dir, project_path

log = get_logger("train_resnet50")

WBC_CLASSES = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]


def train(args):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import datasets, models, transforms
        from torch.utils.data import DataLoader
    except ImportError:
        raise ImportError("Install torch + torchvision: pip install torch torchvision")

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else args.device if args.device != "auto"
        else "cpu"
    )
    log.info("Training on device: %s", device)

    # ── Data transforms ───────────────────────────────────────────────────────
    IMG_SIZE = 224
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_root = project_path("data", "wbc_classifier")
    train_dir = data_root / "train"
    val_dir   = data_root / "val"

    if not train_dir.exists():
        log.error(
            "Training data not found at %s. "
            "Please download a WBC subtype dataset and arrange in ImageFolder format.", train_dir
        )
        sys.exit(1)

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(val_dir),   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    log.info("Train samples: %d  |  Val samples: %d", len(train_ds), len(val_ds))
    log.info("Classes: %s", train_ds.classes)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze backbone, only train classifier head first
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(WBC_CLASSES)),
    )
    model = model.to(device)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    out_dir      = ensure_dir(project_path("models", "resnet50_wbc"))

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after warmup
        if epoch == args.unfreeze_epoch:
            log.info("Epoch %d: Unfreezing backbone for full fine-tuning.", epoch)
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.unfreeze_epoch
            )

        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += imgs.size(0)

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total
        scheduler.step()

        log.info(
            "Epoch %02d/%02d  Train Loss: %.4f  Train Acc: %.2f%%  Val Acc: %.2f%%",
            epoch, args.epochs,
            train_loss / train_total, train_acc * 100, val_acc * 100,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = out_dir / "wbc_classifier_resnet.pth"
            torch.save(model.state_dict(), str(save_path))
            log.info("  ✓ New best model saved (val_acc=%.2f%%) → %s", val_acc * 100, save_path)

    log.info("Training complete. Best val accuracy: %.2f%%", best_val_acc * 100)
    log.info("Weights saved to: %s", out_dir / "wbc_classifier_resnet.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ResNet50 for WBC subtype classification")
    parser.add_argument("--epochs",         type=int,   default=30,    help="Total training epochs")
    parser.add_argument("--batch",          type=int,   default=32,    help="Batch size")
    parser.add_argument("--lr",             type=float, default=1e-4,  help="Initial learning rate")
    parser.add_argument("--unfreeze-epoch", type=int,   default=5,     help="Epoch to unfreeze backbone",
                        dest="unfreeze_epoch")
    parser.add_argument("--device",         type=str,   default="auto",help="'auto', 'cpu', or 'cuda'")
    args = parser.parse_args()
    train(args)
