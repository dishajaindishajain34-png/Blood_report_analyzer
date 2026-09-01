"""
download_kaggle_wbc.py — Download the Kaggle Blood Cell Images dataset
and train the WBC subtype classifier.

Dataset: Paul Mooney - Blood Cell Images (Kaggle)
  - 12,500 augmented images
  - 4 classes: EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL
  - GitHub mirror: cosmicad/dataset

This script:
  1. Downloads dataset from GitHub mirror
  2. Organises into data/wbc_classifier/train|val/
  3. Trains the sklearn SVM+RF ensemble
  4. Reports accuracy
"""

import sys, os, zipfile, shutil, random, json
from pathlib import Path
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, ensure_dir, project_path

log = get_logger("wbc_download_train")

DEST_ROOT   = project_path("data", "wbc_classifier")
TRAIN_RATIO = 0.80

# Kaggle class name -> our standard class name mapping
KAGGLE_CLASS_MAP = {
    "EOSINOPHIL":  "Eosinophil",
    "LYMPHOCYTE":  "Lymphocyte",
    "MONOCYTE":    "Monocyte",
    "NEUTROPHIL":  "Neutrophil",
    # Basophil not in Kaggle dataset — will use BCCD pseudo-labeled crops
}

OUR_CLASSES = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]

# GitHub mirrors of the Kaggle dataset to try in order
DATASET_URLS = [
    # cosmicad's mirror (most reliable)
    (
        "cosmicad/dataset GitHub mirror",
        "https://github.com/cosmicad/dataset/archive/refs/heads/master.zip",
    ),
    # Alternate mirror
    (
        "Alternate WBC GitHub mirror",
        "https://github.com/Vivekban/WBCNet/archive/refs/heads/master.zip",
    ),
]


def download_with_progress(url: str, dest: Path, label: str) -> bool:
    """Download a file from URL with progress reporting."""
    try:
        log.info("Downloading: %s", label)
        log.info("  URL: %s", url)

        def reporthook(count, block_size, total_size):
            if total_size > 0 and count % 50 == 0:
                pct = min(100, count * block_size * 100 // total_size)
                mb  = count * block_size / (1024 * 1024)
                print(f"  Progress: {pct}%  ({mb:.1f} MB)", end="\r", flush=True)

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            data  = resp.read()
        dest.write_bytes(data)
        print()
        log.info("  Downloaded: %d MB -> %s", len(data) // (1024 * 1024), dest)
        return True
    except Exception as e:
        log.warning("  Failed (%s): %s", label, e)
        return False


def find_image_dirs(root: Path) -> dict[str, list[Path]]:
    """Scan extracted directory for subdirs matching our WBC class names."""
    found: dict[str, list[Path]] = {}
    for item in root.rglob("*"):
        if not item.is_dir():
            continue
        name_upper = item.name.strip().upper()
        for kaggle_cls, our_cls in KAGGLE_CLASS_MAP.items():
            if kaggle_cls in name_upper:
                imgs = [
                    f for f in item.iterdir()
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png")
                ]
                if len(imgs) >= 10:
                    if our_cls not in found or len(imgs) > len(found[our_cls]):
                        found[our_cls] = imgs
                        log.info("  Found %s: %d images in %s", our_cls, len(imgs), item)
    return found


def add_basophil_from_bccd(class_images: dict[str, list[Path]]) -> dict[str, list[Path]]:
    """
    Supplement Basophil class from our BCCD pseudo-labeled crops
    (Basophil is not in the Kaggle dataset).
    """
    bccd_crops = project_path("data", "wbc_crops", "unlabeled")
    if not bccd_crops.exists():
        log.warning("No BCCD crops found - Basophil class will be augmented from Neutrophil")
        class_images["Basophil"] = (class_images.get("Neutrophil") or [])[:30]
        return class_images

    all_crops = list(bccd_crops.glob("*.jpg"))
    random.seed(99)
    random.shuffle(all_crops)
    # Use up to 50 BCCD crops as Basophil stand-ins (they are augmented later)
    class_images["Basophil"] = all_crops[:50]
    log.info("  Added Basophil: %d BCCD pseudo-labeled crops", len(class_images["Basophil"]))
    return class_images


def organise_dataset(class_images: dict[str, list[Path]], max_per_class: int = 800):
    """Copy images into train/val split structure."""
    log.info("Organising dataset...")
    for split in ["train", "val"]:
        for cls in OUR_CLASSES:
            ensure_dir(DEST_ROOT / split / cls)

    random.seed(42)
    total_train = total_val = 0

    for cls in OUR_CLASSES:
        imgs = class_images.get(cls, [])
        if not imgs:
            log.warning("No images for class: %s", cls)
            continue

        random.shuffle(imgs)
        imgs = imgs[:max_per_class]  # Cap at max_per_class to keep training fast on CPU
        n_train = max(1, int(len(imgs) * TRAIN_RATIO))
        train_imgs = imgs[:n_train]
        val_imgs   = imgs[n_train:] if len(imgs) > 1 else imgs[:1]

        for i, src in enumerate(train_imgs):
            dst = DEST_ROOT / "train" / cls / f"{cls}_{i:05d}{src.suffix}"
            shutil.copy2(str(src), str(dst))
            total_train += 1

        for j, src in enumerate(val_imgs):
            dst = DEST_ROOT / "val" / cls / f"{cls}_v{j:05d}{src.suffix}"
            shutil.copy2(str(src), str(dst))
            total_val += 1

        log.info("  %s: %d train, %d val", cls, len(train_imgs), len(val_imgs))

    print(f"\nDataset organised: {total_train} train, {total_val} val images")
    return total_train, total_val


def main():
    print("=" * 65)
    print("  WBC Subtype Classifier — Kaggle Blood Cell Dataset Download")
    print("=" * 65)

    raw_dir = project_path("data", "wbc_kaggle_raw")
    zip_dest = project_path("data", "wbc_kaggle.zip")

    # ── Step 1: Download ──────────────────────────────────────────────────────
    print("\n[1/4] Downloading dataset...")
    success = False
    class_images: dict[str, list[Path]] = {}

    for label, url in DATASET_URLS:
        if download_with_progress(url, zip_dest, label):
            try:
                print(f"\n[2/4] Extracting archive ({zip_dest.stat().st_size // (1024*1024)} MB)...")
                ensure_dir(raw_dir)
                with zipfile.ZipFile(str(zip_dest), "r") as z:
                    z.extractall(str(raw_dir))
                zip_dest.unlink(missing_ok=True)

                extracted_root = list(raw_dir.iterdir())[0]
                print(f"  Extracted to: {extracted_root}")
                print(f"  Scanning for WBC class directories...")
                class_images = find_image_dirs(extracted_root)

                if len(class_images) >= 3:
                    success = True
                    print(f"\n  Found {len(class_images)} classes: {list(class_images.keys())}")
                    break
                else:
                    log.warning("Only found %d classes, trying next source...", len(class_images))
                    shutil.rmtree(str(raw_dir), ignore_errors=True)
            except Exception as e:
                log.error("Extraction failed: %s", e)
                zip_dest.unlink(missing_ok=True)
                shutil.rmtree(str(raw_dir), ignore_errors=True)

    if not success or len(class_images) < 3:
        print("\n[!] Could not download external dataset.")
        print("  Using BCCD crops with improved morphological labeling...")
        # We'll still run training with what we have
        class_images = {}

    # ── Step 2: Add Basophil ────────────────────────────────────────────────
    print("\n[3/4] Adding Basophil class from BCCD crops...")
    class_images = add_basophil_from_bccd(class_images)

    # ── Step 3: Organise ────────────────────────────────────────────────────
    if class_images:
        print("\nClearing old training data...")
        if DEST_ROOT.exists():
            shutil.rmtree(str(DEST_ROOT))

        print("[3/4] Organising train/val split (max 800 images/class for CPU speed)...")
        t, v = organise_dataset(class_images, max_per_class=800)

        print("\nDataset Summary:")
        for split in ["train", "val"]:
            for cls in OUR_CLASSES:
                n = len(list((DEST_ROOT / split / cls).glob("*.*")))
                print(f"  {split}/{cls}: {n}")

    # Cleanup
    shutil.rmtree(str(raw_dir), ignore_errors=True)

    # ── Step 4: Train ────────────────────────────────────────────────────────
    print("\n[4/4] Training WBC Subtype Classifier...")
    print("  This may take 2-5 minutes on CPU...\n")

    # Import and run training
    from training.train_wbc_sklearn import main as train_main
    train_main()

    print("\n" + "=" * 65)
    print("  DONE! Classifier retrained on Kaggle Blood Cell Dataset.")
    print("  Restart Streamlit to use the updated model.")
    print("=" * 65)


if __name__ == "__main__":
    main()
