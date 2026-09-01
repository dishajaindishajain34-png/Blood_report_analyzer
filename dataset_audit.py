"""
dataset_audit.py - Audit the BCCD dataset for class distribution
"""
import os
from pathlib import Path
from collections import Counter

DATA_ROOT = Path("C:/Users/HP/Desktop/Blood_report_analyzer-main/data/blood_cells")

CLASS_NAMES = {0: "Platelet", 1: "RBC", 2: "WBC"}

def audit_split(split_name: str):
    label_dir = DATA_ROOT / "labels" / split_name
    image_dir = DATA_ROOT / "images" / split_name

    if not label_dir.exists():
        print(f"  [MISSING] labels/{split_name} not found")
        return Counter(), 0, 0

    label_files = list(label_dir.glob("*.txt"))
    image_files = list(image_dir.glob("*")) if image_dir.exists() else []

    class_counts = Counter()
    empty_labels = 0
    corrupt = 0

    for lf in label_files:
        try:
            lines = lf.read_text().strip().splitlines()
            if not lines:
                empty_labels += 1
                continue
            for line in lines:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    class_counts[cls_id] += 1
        except Exception as e:
            corrupt += 1

    print(f"\n  [{split_name.upper()}]")
    print(f"    Images:      {len(image_files)}")
    print(f"    Label files: {len(label_files)}")
    print(f"    Empty labels:{empty_labels}")
    print(f"    Corrupt:     {corrupt}")
    print(f"    Annotations per class:")
    for cls_id, name in CLASS_NAMES.items():
        print(f"      Class {cls_id} ({name}): {class_counts.get(cls_id, 0)}")
    return class_counts, len(label_files), len(image_files)

def main():
    print("="*60)
    print("BCCD DATASET AUDIT")
    print("="*60)

    if not DATA_ROOT.exists():
        print(f"[ERROR] Dataset root not found: {DATA_ROOT}")
        # Try alternate locations
        for alt in [
            Path("C:/Users/HP/Desktop/Blood_report_analyzer-main/data"),
        ]:
            print(f"  Contents of {alt}:")
            if alt.exists():
                for item in alt.iterdir():
                    print(f"    {item.name} ({'DIR' if item.is_dir() else 'FILE'})")
        return

    total_counts = Counter()
    for split in ["train", "val", "test"]:
        counts, n_labels, n_images = audit_split(split)
        total_counts.update(counts)

    print("\n" + "="*60)
    print("TOTAL ACROSS ALL SPLITS:")
    for cls_id, name in CLASS_NAMES.items():
        print(f"  {name}: {total_counts.get(cls_id, 0)} annotations")
    print("="*60)

    if total_counts.get(0, 0) == 0:
        print("\n⚠️  WARNING: Platelet (class 0) has ZERO annotations in dataset!")
        print("   The trained model CANNOT reliably detect Platelets.")
    else:
        print(f"\n✅ Platelet annotations found: {total_counts[0]}")

if __name__ == "__main__":
    main()
