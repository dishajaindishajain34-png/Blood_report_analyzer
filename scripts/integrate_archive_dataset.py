"""
Script to integrate the archive dataset (JSON annotations) into the existing YOLO format project.
Converts JSON annotations to YOLO format and copies images to appropriate directories.
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

# Define paths
ARCHIVE_BASE = r"C:\Users\NI\Downloads\archive (1)"
PROJECT_BASE = r"C:\Users\NI\Downloads\Blood_report_analyzer"
DATA_BASE = os.path.join(PROJECT_BASE, "data", "bccd")

# Class mapping from meta.json classId to YOLO class index
# Platelets: 13304 -> 0
# RBC: 13302 -> 1
# WBC: 13303 -> 2
CLASS_MAP = {
    13304: 0,  # Platelets
    13302: 1,  # RBC
    13303: 2,  # WBC
}

# Reverse mapping for reference
CLASS_NAMES = {
    0: "Platelet",
    1: "RBC",
    2: "WBC"
}

def json_to_yolo(json_file, img_width, img_height):
    """
    Convert JSON annotation to YOLO format.
    Returns list of YOLO format lines: "class_id center_x center_y width height"
    """
    yolo_lines = []
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for obj in data.get('objects', []):
        class_id = CLASS_MAP.get(obj['classId'])
        if class_id is None:
            print(f"Warning: Unknown classId {obj['classId']} in {json_file}")
            continue
        
        # Extract bounding box coordinates
        points = obj['points']['exterior']
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        # Convert to YOLO format (normalized center coordinates and size)
        center_x = (x1 + x2) / 2.0 / img_width
        center_y = (y1 + y2) / 2.0 / img_height
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height
        
        # Clamp to [0, 1]
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)
    
    return yolo_lines

def process_split(split_name):
    """Process a single split (train/val/test)"""
    split_path = os.path.join(ARCHIVE_BASE, split_name)
    
    if not os.path.exists(split_path):
        print(f"Split {split_name} not found at {split_path}")
        return 0
    
    img_dir = os.path.join(split_path, "img")
    ann_dir = os.path.join(split_path, "ann")
    
    # Create output directories
    output_img_dir = os.path.join(DATA_BASE, "images", split_name)
    output_lbl_dir = os.path.join(DATA_BASE, "labels", split_name)
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)
    
    # Get list of annotation files
    ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.json')])
    
    count = 0
    stats = defaultdict(int)
    
    for ann_file in ann_files:
        ann_path = os.path.join(ann_dir, ann_file)
        img_name = ann_file.replace('.json', '')
        img_path = os.path.join(img_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found for {ann_file}")
            continue
        
        try:
            # Read JSON and extract image size
            with open(ann_path, 'r') as f:
                data = json.load(f)
            
            img_width = data['size']['width']
            img_height = data['size']['height']
            
            # Convert to YOLO format
            yolo_lines = json_to_yolo(ann_path, img_width, img_height)
            
            # Skip if no objects found
            if not yolo_lines:
                print(f"Warning: No objects found in {ann_file}")
                continue
            
            # Count objects by class
            for line in yolo_lines:
                class_id = int(line.split()[0])
                stats[CLASS_NAMES[class_id]] += 1
            
            # Copy image
            output_img_path = os.path.join(output_img_dir, img_name)
            if not os.path.exists(output_img_path):
                shutil.copy2(img_path, output_img_path)
            
            # Write label file
            label_name = img_name.replace('.jpeg', '.txt').replace('.jpg', '.txt')
            output_lbl_path = os.path.join(output_lbl_dir, label_name)
            with open(output_lbl_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            count += 1
            
        except Exception as e:
            print(f"Error processing {ann_file}: {e}")
            continue
    
    print(f"\n{split_name.upper()} Split Summary:")
    print(f"  Processed images: {count}")
    for class_name, obj_count in sorted(stats.items()):
        print(f"  {class_name} detections: {obj_count}")
    
    return count

def main():
    print("Starting archive dataset integration...")
    print(f"Archive location: {ARCHIVE_BASE}")
    print(f"Project location: {PROJECT_BASE}")
    print(f"Data output: {DATA_BASE}\n")
    
    # Read meta.json to verify class mapping
    meta_path = os.path.join(ARCHIVE_BASE, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print("Classes from meta.json:")
        for cls in meta['classes']:
            print(f"  {cls['title']}: id={cls['id']}")
        print()
    
    total_images = 0
    
    # Process each split
    for split in ['train', 'val', 'test']:
        count = process_split(split)
        total_images += count
    
    print(f"\n✓ Integration complete! Total images processed: {total_images}")
    print(f"Data is now available in {DATA_BASE}")

if __name__ == "__main__":
    main()
