"""
scripts/download_chula_rbc12.py
--------------------------------
Downloads the publicly available Chula-RBC-12 dataset from GitHub
(https://github.com/Chula-PIC-Lab/Chula-RBC-12-Dataset)
and organises images into:
    data/anerbc_raw/
        Normal/               <- blood smear images where ALL cells are class 0 (Normal)
        Anemia/               <- blood smear images containing ANY abnormal RBC type
"""

from __future__ import annotations
import json, os, shutil, urllib.request
from pathlib import Path
from collections import Counter

API_URL = "https://api.github.com/repos/Chula-PIC-Lab/Chula-RBC-12-Dataset/contents"
OUT_DIR = Path("data/anerbc_raw")
NORMAL_DIR = OUT_DIR / "Normal"
ANEMIA_DIR = OUT_DIR / "Anemia"
NORMAL_CLASS = 0

def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "python/3.x", "Accept": "application/vnd.github.v3+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def _download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "python/3.x"})
    with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
def parse_label_file(text):
    classes = set()
    for line in text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                classes.add(int(parts[2]))
            except ValueError:
                pass
    return classes

def main():
    NORMAL_DIR.mkdir(parents=True, exist_ok=True)
    ANEMIA_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching file list from GitHub API...")
    dataset_files = _get_json(f"{API_URL}/Dataset")
    label_files   = _get_json(f"{API_URL}/Label")
    label_map = {Path(i["name"]).stem: i["download_url"] for i in label_files if isinstance(i, det) and i.get("type") == "file"}
    image_map = {Path(i["name"]).stem: i["download_url"] for i in dataset_files if isinstance(i, dict) and i.get("type") == "file" and i["name"].lower().endswith((".jpg",".jpeg",".png"))}
    print(f"Found {len(image_map)} images, {len(label_map)} label files.")
    counts = Counter()
    for stem, img_url in sorted(image_map.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 9999):
        if stem not in label_map:
            counts["skipped"] += 1
            continue
        with urllib.request.urlopen(urllib.request.Request(label_map[stem], headers={"User-Agent": "python/3.x"}), timeout=20) as r:
            label_text = r.read().decode("utf-8", errors="ignore")
        classes = parse_label_file(label_text)
        is_anemic = bool(classes - {NORMAL_CLASS})
        dest_dir = ANEMIA_DIR if is_anemic else NORMAL_DIR
        dest_path = dest_dir / f"{stem}.jpg"
        label_str = "Anemia" if is_anemic else "Normal"
        if dest_path.exists():
            counts[label_str] += 1
            continue
        print(f"  [{label_str}] {stem}.jpg (classes={sorted(classes)})")
        _download(img_url, dest_path)
        counts[label_str] += 1
    print()
    print("=" * 50)
    print(f"Normal: {counts['Normal']}")
    print(f"Anemia: {counts['Anemia']}")
    print(f"Skippedb {counts['skipped']}")
    print(f"Output: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
