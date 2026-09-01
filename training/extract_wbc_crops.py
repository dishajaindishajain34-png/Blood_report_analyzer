"""
extract_wbc_crops.py — Extract WBC bounding box crops from BCCD dataset.

Reads Pascal VOC XML annotations, crops each WBC region from the original
JPEG image, and saves them into data/wbc_crops/ for use in classifier training.

Since BCCD has no WBC subtype labels, this creates an UNLABELED crop pool.
We then use pseudo-labeling (cluster-based or model-based) to assign subtypes.

Usage:
    python training/extract_wbc_crops.py
"""

import sys
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from src.utils import get_logger, ensure_dir, project_path

log = get_logger("extract_wbc_crops")

IMAGES_DIR = project_path("data", "bccd_raw", "BCCD", "JPEGImages")
ANNOT_DIR  = project_path("data", "bccd_raw", "BCCD", "Annotations")
OUT_DIR    = project_path("data", "wbc_crops", "unlabeled")
PAD        = 12   # pixels of padding around each bounding box


def extract_crops() -> int:
    ensure_dir(OUT_DIR)
    total = 0

    for xml_path in sorted(ANNOT_DIR.glob("*.xml")):
        img_path = IMAGES_DIR / f"{xml_path.stem}.jpg"
        if not img_path.exists():
            continue

        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        img = Image.open(str(img_path)).convert("RGB")
        w, h = img.size

        for i, obj in enumerate(root.findall("object")):
            if obj.find("name").text.strip() != "WBC":
                continue

            bbox = obj.find("bndbox")
            x1 = max(0,   int(float(bbox.find("xmin").text)) - PAD)
            y1 = max(0,   int(float(bbox.find("ymin").text)) - PAD)
            x2 = min(w-1, int(float(bbox.find("xmax").text)) + PAD)
            y2 = min(h-1, int(float(bbox.find("ymax").text)) + PAD)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img.crop((x1, y1, x2, y2))
            crop_name = f"{xml_path.stem}_wbc_{i:02d}.jpg"
            crop.save(str(OUT_DIR / crop_name), quality=92)
            total += 1

    log.info("Extracted %d WBC crops -> %s", total, OUT_DIR)
    return total


if __name__ == "__main__":
    n = extract_crops()
    print(f"Done. {n} WBC crops saved to data/wbc_crops/unlabeled/")
