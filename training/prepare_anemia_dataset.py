"""Prepare the AneRBC dataset for healthy-vs-anemic classification.

This script scans the AneRBC archive and copies the original RBC images into an
ImageFolder-style dataset suitable for a binary classifier:

    data/anemia_classifier/
      train/
        Healthy/
        Anemic/
      val/
        Healthy/
        Anemic/

The dataset is intentionally organized as a disease classifier rather than a
YOLO detector because the AneRBC archive is labeled at whole-image level
(healthy vs. anemic), not with per-cell bounding boxes.

Usage:
    python training/prepare_anemia_dataset.py --dataset-root "C:\\path\\to\\AneRBC_dataset" \
        --output-root data/anemia_classifier --val-ratio 0.2
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_dir, get_logger, project_path

log = get_logger("prepare_anemia_dataset")


def iter_source_images(dataset_root: Path) -> dict[str, list[Path]]:
    """Find all original RBC images under AneRBC-I and AneRBC-II.

    Returns a mapping:
        {"Healthy": [paths...], "Anemic": [paths...]}
    """
    healthy: list[Path] = []
    anemic: list[Path] = []

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    for dataset_dir in sorted(dataset_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        for patient_group in ("Healthy_individuals", "Anemic_individuals"):
            original_dir = dataset_dir / patient_group / "Original_images"
            if not original_dir.exists():
                continue

            for img_path in sorted(original_dir.rglob("*")):
                if img_path.is_file() and img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                    if "Healthy" in patient_group or "Healthy" in str(dataset_dir):
                        healthy.append(img_path)
                    elif "Anemic" in patient_group or "Anemic" in str(dataset_dir):
                        anemic.append(img_path)

    # Handle the exact dataset layout exposed by the archive readme.
    for section in (dataset_root / "AneRBC-I", dataset_root / "AneRBC-II"):
        if not section.exists():
            continue
        for label, target_list in (("Healthy", healthy), ("Anemic", anemic)):
            original_dir = section / f"{label}_individuals" / "Original_images"
            if not original_dir.exists():
                continue
            for img_path in sorted(original_dir.rglob("*")):
                if img_path.is_file() and img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                    target_list.append(img_path)

    if not healthy or not anemic:
        raise ValueError(
            f"No Healthy/Anemic images found under {dataset_root}. "
            "Expected a structure like AneRBC-I/Healthy_individuals/Original_images and AneRBC-I/Anemic_individuals/Original_images."
        )

    return {"Healthy": healthy, "Anemic": anemic}


def copy_split(images: list[Path], destination_dir: Path, label: str) -> int:
    """Copy images into ImageFolder layout for a single label."""
    target_dir = destination_dir / label
    ensure_dir(target_dir)
    copied = 0
    for src in images:
        dest = target_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
        copied += 1
    return copied


def prepare_dataset(dataset_root: Path, output_root: Path, val_ratio: float = 0.2, seed: int = 42) -> dict[str, int]:
    """Create the train/val split in ImageFolder format."""
    random.seed(seed)

    source = iter_source_images(dataset_root)
    train_root = output_root / "train"
    val_root = output_root / "val"
    ensure_dir(train_root)
    ensure_dir(val_root)

    totals = {}
    for label, paths in source.items():
        random.shuffle(paths)
        split_index = max(1, int(len(paths) * (1.0 - val_ratio)))
        train_paths = paths[:split_index]
        val_paths = paths[split_index:]

        totals[label] = {
            "train": copy_split(train_paths, train_root, label),
            "val": copy_split(val_paths, val_root, label),
        }

    log.info("Dataset prepared at %s", output_root)
    log.info("Healthy train=%d val=%d | Anemic train=%d val=%d",
             totals["Healthy"]["train"], totals["Healthy"]["val"],
             totals["Anemic"]["train"], totals["Anemic"]["val"])
    return {"Healthy": totals["Healthy"], "Anemic": totals["Anemic"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare AneRBC healthy-vs-anemic dataset for binary classification.")
    parser.add_argument("--dataset-root", type=Path, default=Path(r"C:\Users\NI\Downloads\archive (1)\AneRBC dataset a benchmark dataset for computer-aided anemia diagnosis using RBC images. httpsdoi.org10.1093databasebaae120\AneRBC_dataset"), help="Path to the AneRBC_dataset folder")
    parser.add_argument("--output-root", type=Path, default=project_path("data", "anemia_classifier"), help="Destination folder for ImageFolder-style data")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction held out for validation (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stable splits")
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    prepare_dataset(args.dataset_root, args.output_root, val_ratio=args.val_ratio, seed=args.seed)
