"""
setup_dataset.py — One-shot dataset preparation script.

What this does:
  1. Copies BCCD JPEGImages to data/bccd/images/
  2. Converts Pascal VOC XML annotations → YOLO .txt labels
  3. Splits into train/val/test (70/20/10)
  4. Copies a sample image to data/sample_images/sample.jpg
  5. Updates training/data.yaml with absolute paths

Run from project root:
    python setup_dataset.py
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATASET_SRC = Path(r"C:\Users\NI\Downloads\Complete-Blood-Cell-Count-Dataset-master")

DEST_ROOT   = ROOT / "data" / "bccd"
SAMPLE_DIR  = ROOT / "data" / "sample_images"

# BCCD class mapping (Pascal VOC class name → YOLO index)
CLASS_MAP = {
    "platelet": 0,
    "platelets": 0,
    "rbc":      1,
    "wbc":      2,
}


def parse_xml_annotation(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """
    Parse Pascal VOC XML → YOLO format lines.
    Each line: <class_id> <cx> <cy> <w> <h>  (all normalised 0–1)
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    lines = []

    for obj in root.findall("object"):
        name = (obj.find("name").text or "").strip()
        normalized = name.lower()
        if normalized not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[normalized]

        bbox  = obj.find("bndbox")
        xmin  = float(bbox.find("xmin").text)
        ymin  = float(bbox.find("ymin").text)
        xmax  = float(bbox.find("xmax").text)
        ymax  = float(bbox.find("ymax").text)

        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return lines


def setup():
    from PIL import Image

    print("=" * 60)
    print("  Complete-Blood-Cell-Count-Dataset Setup")
    print("=" * 60)

    folder_map = {
        "train": DATASET_SRC / "Training",
        "val":   DATASET_SRC / "Validation",
        "test":  DATASET_SRC / "Testing",
    }

    # ── Create destination directories ────────────────────────────────────────
    for split in folder_map:
        (DEST_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DEST_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Convert and copy ─────────────────────────────────────────────────────
    total_boxes = 0
    skipped     = 0

    first_sample = None

    for split, folder in folder_map.items():
        img_dir = folder / "Images"
        xml_dir = folder / "Annotations"

        imgs = sorted(img_dir.glob("*.jpg"))
        print(f"\nProcessing {split} split ({len(imgs)} images)...")

        for img_path in imgs:
            if first_sample is None:
                first_sample = img_path

            stem     = img_path.stem
            xml_path = xml_dir / f"{stem}.xml"

            if not xml_path.exists():
                skipped += 1
                continue

            # Get image dimensions
            with Image.open(str(img_path)) as im:
                img_w, img_h = im.size

            # Convert annotation
            yolo_lines = parse_xml_annotation(xml_path, img_w, img_h)
            if not yolo_lines:
                skipped += 1
                continue

            total_boxes += len(yolo_lines)

            # Copy image
            shutil.copy2(str(img_path), str(DEST_ROOT / "images" / split / img_path.name))

            # Write label
            label_path = DEST_ROOT / "labels" / split / f"{stem}.txt"
            label_path.write_text("\n".join(yolo_lines))

    print(f"\nConversion complete: {total_boxes} bounding boxes across all splits")
    if skipped:
        print(f"  (skipped {skipped} images with no/mismatched annotations)")

    # ── Sample image ──────────────────────────────────────────────────────────
    if first_sample and first_sample.exists():
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        sample_dest = SAMPLE_DIR / "sample.jpg"
        shutil.copy2(str(first_sample), str(sample_dest))
        print(f"\nSample image copied -> {sample_dest}")

    # ── Update data.yaml ──────────────────────────────────────────────────────
    data_yaml = ROOT / "training" / "data.yaml"
    yaml_content = f"""# Complete-Blood-Cell-Count-Dataset — YOLOv8 Configuration
path: {str(DEST_ROOT).replace(chr(92), '/')}
train: images/train
val:   images/val
test:  images/test

nc: 3

names:
  0: Platelet
  1: RBC
  2: WBC
"""
    data_yaml.write_text(yaml_content)
    print(f"data.yaml updated -> {data_yaml}")

    print("\n" + "=" * 60)
    print("  Setup Complete!")
    print("=" * 60)


if __name__ == "__main__":
    setup()
