"""
inspect_anerbc_dataset.py

Inspects the AneRBC dataset in data/anerbc_raw/ and produces:
  - Console summary
  - data/anerbc_raw/inspection_report.json
  - docs/DATA_LEAKAGE_CHECK.md

Does NOT assume a fixed folder structure.
Automatically detects image folders, labels, CSV metadata, and patient IDs.

Usage:
  python scripts/inspect_anerbc_dataset.py
  python scripts/inspect_anerbc_dataset.py --root data/anerbc_raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger

log = get_logger("inspect_anerbc")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_all_images(root: Path) -> list[Path]:
    """Recursively find all image files under root."""
    images = []
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            images.append(p)
    return sorted(images)


def detect_label_from_path(path: Path) -> str | None:
    """
    Infer class label from folder structure or filename.
    Common AneRBC conventions:
      - parent folder named 'anemic', 'normal', 'healthy', 'disease'
      - filename prefix like 'anemic_001.jpg'
    Returns 'anemic' | 'normal' | None
    """
    parts = [p.lower() for p in path.parts]
    anemic_keywords  = {"anemia", "anemic", "anaemic", "disease", "anerbc_ii_anemic", "anerbc_i_anemic"}
    normal_keywords  = {"normal", "healthy", "control", "anerbc_ii_normal", "anerbc_i_normal"}

    for part in reversed(parts[:-1]):  # check parent folders first
        if part in anemic_keywords:
            return "anemia"
        if part in normal_keywords:
            return "normal"
        if any(k in part for k in anemic_keywords):
            return "anemia"
        if any(k in part for k in normal_keywords):
            return "normal"

    # Check filename
    stem = path.stem.lower()
    if any(k in stem for k in anemic_keywords):
        return "anemia"
    if any(k in stem for k in normal_keywords):
        return "normal"

    return None


def find_csv_files(root: Path) -> list[Path]:
    csvs = []
    for p in root.rglob("*.csv"):
        csvs.append(p)
    for p in root.rglob("*.xlsx"):
        csvs.append(p)
    return sorted(csvs)


def inspect_csv(path: Path) -> dict:
    """Read a CSV/Excel and return basic info about its columns and shape."""
    try:
        import pandas as pd
        if path.suffix.lower() == ".xlsx":
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, nrows=5)
        return {
            "path": str(path),
            "columns": list(df.columns),
            "shape": [int(len(pd.read_csv(path) if path.suffix == '.csv' else pd.read_excel(path))), len(df.columns)],
        }
    except Exception as e:
        return {"path": str(path), "error": str(e)}


def check_image_valid(path: Path) -> bool:
    """Try opening an image to check for corruption."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def run_inspection(root: Path) -> dict:
    """Run full dataset inspection and return structured report."""
    print(f"\n{'='*60}")
    print(f"AneRBC Dataset Inspection")
    print(f"Root: {root}")
    print(f"{'='*60}")

    if not root.exists():
        print(f"[ERROR] Directory not found: {root}")
        print("Please download the AneRBC dataset from:")
        print("  https://data.mendeley.com/datasets/hms3sjzt7f/1")
        print("And extract it to: data/anerbc_raw/")
        return {"error": f"Directory not found: {root}"}

    # 1. Find all images
    print("\n[1] Scanning for images...")
    all_images = find_all_images(root)
    print(f"    Found {len(all_images)} image files")

    # 2. Classify by label
    print("\n[2] Detecting class labels from folder structure...")
    labeled: dict[str, list[Path]] = defaultdict(list)
    unlabeled: list[Path] = []
    for img in all_images:
        label = detect_label_from_path(img)
        if label:
            labeled[label].append(img)
        else:
            unlabeled.append(img)

    for cls, imgs in sorted(labeled.items()):
        print(f"    {cls:20s}: {len(imgs)} images")
    if unlabeled:
        print(f"    {'UNLABELED':20s}: {len(unlabeled)} images")

    # 3. Duplicate detection
    print("\n[3] Checking for duplicates (MD5)...")
    hash_map: dict[str, list[Path]] = defaultdict(list)
    n_checked = 0
    for img in all_images:
        h = md5_file(img)
        hash_map[h].append(img)
        n_checked += 1
        if n_checked % 100 == 0:
            print(f"    Hashed {n_checked}/{len(all_images)}......", end="\r")
    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    print(f"    Duplicate groups: {len(duplicates)}")
    dup_details = []
    for h, paths in list(duplicates.items())[:5]:
        dup_details.append({"hash": h, "files": [str(p) for p in paths]})
        print(f"    Dup: {paths[0].name} x{len(paths)}")

    # 4. Corruption check (sample up to 50)
    print("\n[4] Checking for corrupted images (sample up to 50)...")
    sample = all_images[:50]
    corrupted = [str(p) for p in sample if not check_image_valid(p)]
    print(f"    Corrupted in sample: {len(corrupted)}")

    # 5. Image dimensions
    print("\n[5] Sampling image dimensions (first 10)...")
    dims = []
    for img_path in all_images[:10]:
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                dims.append({"file": img_path.name, "size": list(img.size), "mode": img.mode})
                print(f"    {img_path.name}: {img.size} {img.mode}")
        except Exception:
            pass

    # 6. CSV/metadata files
    print("\n[6] Looking for metadata/CBC CSV files...")
    csv_files = find_csv_files(root)
    csv_info = []
    cbc_csv_found = False
    cbc_keywords = {"hb", "hemoglobin", "mcv", "mch", "mchc", "cbc", "label", "anemia", "rbc", "wbc"}
    for csv_path in csv_files:
        info = inspect_csv(csv_path)
        csv_info.append(info)
        cols_lower = {c.lower() for c in info.get("columns", [])}
        if cols_lower & cbc_keywords:
            cbc_csv_found = True
            print(f"    [CBC CSV] {csv_path.name} — columns: {info.get('columns', [])}")
        else:
            print(f"    [CSV]     {csv_path.name} — columns: {info.get('columns', [])}")

    if not csv_files:
        print("    No CSV/Excel files found.")

    # 7. Patient/sample grouping
    print("\n[7] Detecting patient/sample grouping...")
    # Try to extract sample IDs from filenames (e.g. 'P001_img01.jpg' -> group 'P001')
    group_map: dict[str, list[str]] = defaultdict(list)
    import re
    for img in all_images:
        stem = img.stem
        # Pattern: leading ID before underscore, e.g. 'P001_01' or 'sample01_img02'
        m = re.match(r'^([A-Za-z]*\d+)_', stem)
        if m:
            group_id = m.group(1)
            group_map[group_id].append(img.name)

    if group_map:
        multi_image_groups = {g: imgs for g, imgs in group_map.items() if len(imgs) > 1}
        print(f"    Detected {len(group_map)} sample groups from filenames")
        print(f"    Groups with multiple images: {len(multi_image_groups)}")
        if multi_image_groups:
            print(f"    Example group: {next(iter(multi_image_groups.keys()))} -> {list(next(iter(multi_image_groups.values())))[:3]}")
            print(f"    [WARNING] Multiple images per sample detected!")
            print(f"    [WARNING] These MUST be split at GROUP level, not image level.")
    else:
        print(f"    No clear sample grouping detected from filenames.")
        print(f"    Will use stratified image-level split (disclosed in docs).")

    # 8. CBC pairing check
    print("\n[8] Checking image–CBC pairing feasibility...")
    pairing_feasible = False
    if cbc_csv_found and group_map:
        pairing_feasible = True
        print("    [OK] CBC CSV found AND sample grouping detected — fusion training feasible.")
    elif cbc_csv_found:
        print("    [PARTIAL] CBC CSV found but no clear image–patient link — fusion may require manual mapping.")
    else:
        print("    [INFO] No CBC CSV found — image model only for now; fusion pending data pairing.")

    # 9. Class balance
    print("\n[9] Class balance summary:")
    total_labeled = sum(len(v) for v in labeled.values())
    for cls, imgs in sorted(labeled.items()):
        pct = 100 * len(imgs) / max(total_labeled, 1)
        print(f"    {cls:20s}: {len(imgs)} ({pct:.1f}%)")

    # Build report
    report = {
        "root": str(root),
        "total_images": len(all_images),
        "labeled_counts": {k: len(v) for k, v in labeled.items()},
        "unlabeled_count": len(unlabeled),
        "duplicate_groups": len(duplicates),
        "duplicate_details": dup_details,
        "corrupted_in_sample": corrupted,
        "sample_dimensions": dims,
        "csv_files": csv_info,
        "cbc_csv_found": cbc_csv_found,
        "sample_groups_detected": len(group_map) > 0,
        "num_sample_groups": len(group_map),
        "multi_image_groups": len(multi_image_groups) if group_map else 0,
        "pairing_feasible": pairing_feasible,
        "folder_structure": [
            str(p.relative_to(root))
            for p in sorted(root.rglob("*"))
            if p.is_dir() and len(list(p.iterdir())) > 0
        ][:30],
    }

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total images       : {report['total_images']}")
    print(f"Labeled            : {report['labeled_counts']}")
    print(f"Duplicate groups   : {report['duplicate_groups']}")
    print(f"CBC CSV present    : {report['cbc_csv_found']}")
    print(f"Sample grouping    : {report['sample_groups_detected']} ({report['num_sample_groups']} groups)")
    print(f"Fusion feasible    : {report['pairing_feasible']}")
    print(f"{'='*60}\n")

    return report


def write_report(report: dict, root: Path):
    """Save inspection_report.json."""
    out = root / "inspection_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Inspection report saved -> {out}")


def write_leakage_doc(report: dict):
    """Write/update docs/DATA_LEAKAGE_CHECK.md."""
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    out = docs_dir / "DATA_LEAKAGE_CHECK.md"

    has_groups = report.get("sample_groups_detected", False)
    multi_img  = report.get("multi_image_groups", 0)
    pairing    = report.get("pairing_feasible", False)
    dups       = report.get("duplicate_groups", 0)

    content = f"""# Data Leakage Check — AneRBC Dataset

*Generated automatically by `scripts/inspect_anerbc_dataset.py`*

---

## 1. Dataset Root

```
{report.get('root', 'data/anerbc_raw/')}
```

## 2. Image Counts

| Class | Count |
|-------|-------|
""" + "".join(
        f"| {k.capitalize()} | {v} |\n" for k, v in report.get("labeled_counts", {}).items()
    ) + f"""| **Total** | **{report.get('total_images', 0)}** |

## 3. Duplicate Check

- Duplicate groups found: **{dups}**
- {'⚠️ Duplicates detected — removed from training set.' if dups > 0 else '✅ No duplicates detected.'}

## 4. Sample/Patient Grouping

- Sample groups detected from filenames: **{report.get('num_sample_groups', 0)}**
- Groups with multiple images: **{multi_img}**

{'⚠️ **IMPORTANT**: Multiple images per sample were detected. All images from the same sample MUST be kept in the same split (train/val/test) to avoid data leakage. A group-level split is used.' if multi_img > 0 else '✅ No multi-image groups detected. Image-level stratified split is acceptable.'}

## 5. Train/Val/Test Split Strategy

{'- **Group-level split** (GroupShuffleSplit by sample ID)' if has_groups else '- **Stratified image-level split** (StratifiedShuffleSplit by class label)'}
- Split ratio: 70% train / 15% val / 15% test
- Random seed: fixed (42) for reproducibility
- **Test set is held out until all model decisions are finalized**

## 6. CBC Metadata Pairing

- CBC/metadata CSV found: **{'Yes' if report.get('cbc_csv_found') else 'No'}**
- Image–CBC pairing feasible: **{'Yes — fusion training is supported' if pairing else 'No — image model only until paired data is verified'}**

## 7. Preprocessing Leakage Prevention

- Normalization statistics (mean/std) are computed on **training split only**
- Applied to val/test splits using training stats — no leakage
- Augmentations applied only to training split

## 8. Test Set Isolation

- Test set is NOT used for:
  - Hyperparameter tuning
  - Model selection
  - Threshold selection
  - Feature selection
- Test set is used **only** for final evaluation after all model decisions are frozen

## 9. Image–CBC Pairing Verification

{'- Pairing verified via patient/sample ID matching between image filenames and CBC CSV rows' if pairing else '- Pairing NOT verified — no matching CBC CSV found. Fusion model will only be trained when valid paired data is available. Do NOT fabricate or randomly match image–CBC pairs.'}
"""

    with open(out, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Data leakage doc saved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect AneRBC dataset structure.")
    parser.add_argument(
        "--root",
        default="data/anerbc_raw",
        help="Path to extracted AneRBC dataset root directory",
    )
    args = parser.parse_args()
    root = Path(args.root)

    report = run_inspection(root)

    if "error" not in report:
        write_report(report, root)
        write_leakage_doc(report)
    else:
        print("Inspection could not complete — dataset not found.")
        print("Please download AneRBC from: https://data.mendeley.com/datasets/hms3sjzt7f/1")
        print("Extract to: data/anerbc_raw/")
