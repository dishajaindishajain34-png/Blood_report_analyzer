import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from PIL import Image

random.seed(42)  # Set seed for reproducibility

# Source paths
ROOT = Path("c:/Users/HP/Desktop/Blood_report_analyzer-main")
DATASET_SRC_IMAGES = ROOT / "data" / "bccd_raw" / "BCCD_Dataset-master" / "BCCD" / "JPEGImages"
DATASET_SRC_XML = ROOT / "data" / "bccd_raw" / "BCCD_Dataset-master" / "BCCD" / "Annotations"

# Destination paths
DEST_ROOT = ROOT / "data" / "blood_cells"
DEST_IMAGES = DEST_ROOT / "images"
DEST_LABELS = DEST_ROOT / "labels"

CLASS_MAP = {
    "platelet": 0,
    "platelets": 0,
    "rbc": 1,
    "wbc": 2
}

def parse_xml(xml_path, img_w, img_h):
    """Parses XML and returns list of (cls_id, cx, cy, w, h) tuples."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        name = (obj.find("name").text or "").strip().lower()
        if name not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[name]
        
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        
        # Clip coordinates to image boundary
        xmin = max(0.0, min(xmin, img_w))
        xmax = max(0.0, min(xmax, img_w))
        ymin = max(0.0, min(ymin, img_h))
        ymax = max(0.0, min(ymax, img_h))
        
        if xmax <= xmin or ymax <= ymin:
            continue
            
        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h
        
        boxes.append((cls_id, cx, cy, bw, bh))
    return boxes

def augment_horizontal_flip(img_path, boxes, dest_img_path, dest_lbl_path):
    # Flip image
    with Image.open(img_path) as img:
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_img.save(dest_img_path)
        
    # Flip boxes
    new_boxes = []
    for cls_id, cx, cy, w, h in boxes:
        new_boxes.append((cls_id, 1.0 - cx, cy, w, h))
        
    # Save label file
    with open(dest_lbl_path, "w") as f:
        for box in new_boxes:
            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

def augment_vertical_flip(img_path, boxes, dest_img_path, dest_lbl_path):
    # Flip image
    with Image.open(img_path) as img:
        flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_img.save(dest_img_path)
        
    # Flip boxes
    new_boxes = []
    for cls_id, cx, cy, w, h in boxes:
        new_boxes.append((cls_id, cx, 1.0 - cy, w, h))
        
    # Save label file
    with open(dest_lbl_path, "w") as f:
        for box in new_boxes:
            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

def augment_both_flip(img_path, boxes, dest_img_path, dest_lbl_path):
    # Flip both (rotate 180)
    with Image.open(img_path) as img:
        flipped_img = img.transpose(Image.ROTATE_180)
        flipped_img.save(dest_img_path)
        
    # Flip boxes
    new_boxes = []
    for cls_id, cx, cy, w, h in boxes:
        new_boxes.append((cls_id, 1.0 - cx, 1.0 - cy, w, h))
        
    # Save label file
    with open(dest_lbl_path, "w") as f:
        for box in new_boxes:
            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

def main():
    print("=" * 60)
    print("  Creating Clean Combined Blood Cells Dataset (70/15/15 Split)")
    print("=" * 60)
    
    # 1. Clean destination folder if it exists
    if DEST_ROOT.exists():
        print(f"Cleaning existing directory: {DEST_ROOT}")
        shutil.rmtree(DEST_ROOT)
        
    # Create directories
    for split in ["train", "val", "test"]:
        (DEST_IMAGES / split).mkdir(parents=True, exist_ok=True)
        (DEST_LABELS / split).mkdir(parents=True, exist_ok=True)
        
    # Get all XML annotations
    xml_files = sorted(list(DATASET_SRC_XML.glob("*.xml")))
    random.shuffle(xml_files)
    
    total_imgs = len(xml_files)
    n_train = int(total_imgs * 0.70)
    n_val = int(total_imgs * 0.15)
    n_test = total_imgs - n_train - n_val
    
    splits = {
        "train": xml_files[:n_train],
        "val": xml_files[n_train:n_train + n_val],
        "test": xml_files[n_train + n_val:]
    }
    
    print(f"Original Split counts: Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    # Audit counters for each split
    split_image_counts = Counter()
    split_annotation_counts = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter()
    }
    
    # Process each split
    for split, files in splits.items():
        print(f"\nProcessing {split} split...")
        for xml_file in files:
            stem = xml_file.stem
            img_file = DATASET_SRC_IMAGES / f"{stem}.jpg"
            if not img_file.exists():
                print(f"  [WARNING] Image not found for annotation: {xml_file.name}")
                continue
                
            # Get dimensions
            with Image.open(img_file) as im:
                img_w, img_h = im.size
                
            boxes = parse_xml(xml_file, img_w, img_h)
            if not boxes:
                continue
                
            # Basic validation checks
            valid_boxes = []
            for box in boxes:
                cls_id, cx, cy, w, h = box
                if cls_id not in [0, 1, 2]:
                    print(f"  [ERROR] Invalid class ID: {cls_id} in {xml_file.name}")
                    continue
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    print(f"  [ERROR] Out of bounds box: {box} in {xml_file.name}")
                    continue
                valid_boxes.append(box)
                
            if not valid_boxes:
                continue
                
            # Copy original image & write label file
            dest_img = DEST_IMAGES / split / f"{stem}.jpg"
            dest_lbl = DEST_LABELS / split / f"{stem}.txt"
            
            shutil.copy2(img_file, dest_img)
            with open(dest_lbl, "w") as f:
                for box in valid_boxes:
                    f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                    
            split_image_counts[split] += 1
            for box in valid_boxes:
                split_annotation_counts[split][box[0]] += 1
                
            # --- Perform Platelet Augmentation in Training Split Only ---
            # If training image contains at least one platelet (class 0), augment it to balance class distribution
            if split == "train" and any(box[0] == 0 for box in valid_boxes):
                # 1. Horizontal flip
                augment_horizontal_flip(
                    img_file, valid_boxes, 
                    DEST_IMAGES / "train" / f"{stem}_hflip.jpg",
                    DEST_LABELS / "train" / f"{stem}_hflip.txt"
                )
                split_image_counts["train"] += 1
                for box in valid_boxes:
                    split_annotation_counts["train"][box[0]] += 1
                    
                # 2. Vertical flip
                augment_vertical_flip(
                    img_file, valid_boxes, 
                    DEST_IMAGES / "train" / f"{stem}_vflip.jpg",
                    DEST_LABELS / "train" / f"{stem}_vflip.txt"
                )
                split_image_counts["train"] += 1
                for box in valid_boxes:
                    split_annotation_counts["train"][box[0]] += 1
                    
                # 3. Horizontal + Vertical flip
                augment_both_flip(
                    img_file, valid_boxes, 
                    DEST_IMAGES / "train" / f"{stem}_bflip.jpg",
                    DEST_LABELS / "train" / f"{stem}_bflip.txt"
                )
                split_image_counts["train"] += 1
                for box in valid_boxes:
                    split_annotation_counts["train"][box[0]] += 1
                    
    # Generate data.yaml
    data_yaml = DEST_ROOT / "data.yaml"
    yaml_content = f"""path: {str(DEST_ROOT).replace(chr(92), '/')}
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
    
    # ── Print Audit Summary ──
    print("\n" + "=" * 60)
    print("  DATASET SPLIT & ANNOTATION SUMMARY")
    print("=" * 60)
    for s in ["train", "val", "test"]:
        print(f"\nSplit: {s.upper()}")
        print(f"  Total Images: {split_image_counts[s]}")
        print(f"  Class breakdown:")
        print(f"    0 (Platelet): {split_annotation_counts[s][0]} instances")
        print(f"    1 (RBC):      {split_annotation_counts[s][1]} instances")
        print(f"    2 (WBC):      {split_annotation_counts[s][2]} instances")
        print(f"    Total Boxes:  {sum(split_annotation_counts[s].values())}")
    print("=" * 60)

if __name__ == "__main__":
    main()
