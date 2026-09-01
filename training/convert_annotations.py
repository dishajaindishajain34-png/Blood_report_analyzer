"""
Convert Roboflow/JSON rectangle annotations to YOLOv8 label txt files.
Assumes extracted dataset at C:/Users/Admin/Downloads/archive_images with
subfolders train/img, val/img, test/img and corresponding ann/*.json files.

Creates labels in train/labels, val/labels, test/labels with filenames matching images.
"""
from pathlib import Path
import json

root = Path("C:/Users/Admin/Downloads/archive_images")
class_map = {"Platelet": 0, "RBC": 1, "WBC": 2}

for split in ["train", "val", "test"]:
    img_dir = root / split / "img"
    ann_dir = root / split / "ann"
    labels_dir = root / split / "labels"
    if not img_dir.exists():
        print(f"Skipping {split}, no {img_dir}")
        continue
    labels_dir.mkdir(parents=True, exist_ok=True)
    ann_files = list(ann_dir.glob('*.json')) if ann_dir.exists() else []
    print(f"Processing {split}: {len(ann_files)} annotation files")
    for ann in ann_files:
        try:
            data = json.loads(ann.read_text())
        except Exception as e:
            print(f"Failed to read {ann}: {e}")
            continue
        img_w = data.get('size', {}).get('width')
        img_h = data.get('size', {}).get('height')
        if not img_w or not img_h:
            print(f"No size for {ann}, skipping")
            continue
        lines = []
        for obj in data.get('objects', []):
            cls = obj.get('classTitle')
            if cls not in class_map:
                # skip unknown classes
                continue
            pts = obj.get('points', {}).get('exterior', [])
            if not pts or len(pts) < 2:
                continue
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            # sanitize
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)
            w = x_max - x_min
            h = y_max - y_min
            x_center = x_min + w / 2.0
            y_center = y_min + h / 2.0
            # normalize
            x_center /= img_w
            y_center /= img_h
            w /= img_w
            h /= img_h
            # clip to (0,1)
            def clip(v):
                return max(0.0, min(1.0, v))
            x_center, y_center, w, h = map(clip, [x_center, y_center, w, h])
            lines.append(f"{class_map[cls]} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        # write label file with same base name as image
        img_name = ann.name.replace('.json', '')
        label_path = labels_dir / (Path(img_name).stem + '.txt')
        if lines:
            label_path.write_text('\n'.join(lines))
        else:
            # write empty file to indicate no objects (optional)
            label_path.write_text('')
print('Done converting annotations')
