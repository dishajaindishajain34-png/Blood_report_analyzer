"""Train a binary CNN for anemia diagnosis using the AneRBC dataset.

Dataset layout expected after running prepare_anemia_dataset.py:

    data/anemia_classifier/
      train/
        Healthy/
        Anemic/
      val/
        Healthy/
        Anemic/

This is a disease-classification model (Healthy vs Anemic), not a YOLO object detector.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_dir, get_logger, project_path

log = get_logger("train_anemia_classifier")


def train(args):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        from torchvision import datasets, models, transforms
    except ImportError as exc:
        raise ImportError("Install torch + torchvision: pip install torch torchvision") from exc

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
        if args.device != "auto" else "cpu"
    )
    log.info("Training on device: %s", device)

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset layout under {data_root}. Run training/prepare_anemia_dataset.py first."
        )

    img_size = 224
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    log.info("Train samples: %d  |  Val samples: %d", len(train_ds), len(val_ds))
    log.info("Classes: %s", train_ds.classes)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(train_ds.classes)),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    save_dir = ensure_dir(project_path("models", "anemia_classifier"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        train_acc = correct / total
        val_acc = val_correct / val_total
        scheduler.step()

        log.info(
            "Epoch %02d/%02d | Train Loss %.4f | Train Acc %.2f%% | Val Acc %.2f%%",
            epoch, args.epochs,
            train_loss / total,
            train_acc * 100,
            val_acc * 100,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(save_dir / "anemia_classifier_resnet.pth"))
            log.info("  ✓ Best model saved -> %s", save_dir / "anemia_classifier_resnet.pth")

    log.info("Training complete. Best validation accuracy: %.2f%%", best_val_acc * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a healthy-vs-anemic RBC classification model")
    parser.add_argument("--data-root", type=str, default=str(project_path("data", "anemia_classifier")), help="Path to the split dataset generated by prepare_anemia_dataset.py")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or 'cuda'")
    args = parser.parse_args()
    train(args)
