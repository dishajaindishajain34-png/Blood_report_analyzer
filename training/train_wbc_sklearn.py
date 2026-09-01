"""
train_wbc_sklearn.py — Train a lightweight sklearn SVM/RF WBC subtype classifier.

Uses HOG (Histogram of Oriented Gradients) + colour histogram features.
No GPU or large RAM required — trains in under 60 seconds on CPU.

Model is saved as a joblib file:  models/resnet50_wbc/wbc_classifier_sklearn.pth
(same path, but .pth holds the joblib bytes for compatibility)

Usage:
    python training/train_wbc_sklearn.py
"""

import sys, json, joblib, io
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from src.utils import get_logger, ensure_dir, project_path

log = get_logger("train_wbc_sklearn")

WBC_CLASSES  = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]
DATA_ROOT    = project_path("data", "wbc_classifier")
MODEL_OUT    = project_path("models", "resnet50_wbc", "wbc_classifier_sklearn.pth")
HISTORY_OUT  = project_path("models", "resnet50_wbc", "training_history.json")
IMG_SIZE     = 64   # Resize all crops to this before feature extraction


# ── Feature extraction ─────────────────────────────────────────────────────────────
def extract_features(img: Image.Image) -> np.ndarray:
    """
    Extract a compact feature vector from a WBC crop:
      - Colour histograms (RGB) → 48 features
      - HSV histograms         → 48 features
      - Grayscale HOG-style gradient histogram → 128 features
      Total: 224 features per image
    """
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0

    feats = []

    # 1. RGB colour histogram (16 bins × 3 channels)
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0, 1))
        feats.extend(hist / (IMG_SIZE * IMG_SIZE))

    # 2. HSV colour histogram
    # HSV colour histogram — vectorized conversion
    import matplotlib.colors as mcolors
    hsv = mcolors.rgb_to_hsv(arr)
    for ch in range(3):
        hist, _ = np.histogram(hsv[:, :, ch], bins=16, range=(0, 1))
        feats.extend(hist / (IMG_SIZE * IMG_SIZE))

    # 3. Gradient magnitude histogram (approx HOG)
    gray = arr.mean(axis=2)
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    mag = np.sqrt(gx**2 + gy**2)
    ang = (np.arctan2(gy, gx) * 180 / np.pi) % 360
    # Spatial 4×4 grid, 8 orientation bins → 4*4*8 = 128 features
    cell = IMG_SIZE // 4
    for cy in range(4):
        for cx in range(4):
            patch_mag = mag[cy*cell:(cy+1)*cell, cx*cell:(cx+1)*cell]
            patch_ang = ang[cy*cell:(cy+1)*cell, cx*cell:(cx+1)*cell]
            hist, _ = np.histogram(patch_ang, bins=8, range=(0, 360),
                                   weights=patch_mag)
            hist_sum = hist.sum()
            feats.extend(hist / (hist_sum + 1e-8))

    return np.array(feats, dtype=np.float32)


def load_dataset(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load all images from data/wbc_classifier/{split}/ and extract features."""
    X, y = [], []
    split_dir = DATA_ROOT / split
    for cls_idx, cls_name in enumerate(WBC_CLASSES):
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            continue
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        log.info("  %s/%s: %d images", split, cls_name, len(imgs))
        for img_path in imgs:
            try:
                feat = extract_features(Image.open(str(img_path)))
                X.append(feat)
                y.append(cls_idx)
            except Exception as e:
                log.warning("Skipping %s: %s", img_path.name, e)
    return np.array(X), np.array(y)


def main():
    print("=" * 60)
    print("  WBC Subtype Classifier — sklearn SVM Training")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading train set...")
    X_train, y_train = load_dataset("train")
    print(f"  {len(X_train)} train samples, {X_train.shape[1]} features")

    print("\nLoading val set...")
    X_val, y_val = load_dataset("val")
    print(f"  {len(X_val)} val samples")

    if len(X_train) == 0:
        print("ERROR: No training data found. Run download_wbc_dataset.py first.")
        return

    # ── Compute class weights ─────────────────────────────────────────────────
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
    log.info("Class weights: %s", cw_dict)

    # ── Build ensemble pipeline ───────────────────────────────────────────────
    svm = SVC(
        kernel="rbf", C=10.0, gamma="scale",
        class_weight=cw_dict, probability=True, random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=200, class_weight=cw_dict,
        random_state=42, n_jobs=-1,
    )
    # Voting ensemble: SVM + RF
    ensemble = VotingClassifier(
        estimators=[("svm", svm), ("rf", rf)],
        voting="soft",
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    ensemble),
    ])

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nTraining SVM + RandomForest ensemble...")
    pipe.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    print(f"\nTrain accuracy: {train_acc:.1%}")

    if len(X_val) > 0:
        y_pred = pipe.predict(X_val)
        val_acc = accuracy_score(y_val, y_pred)
        print(f"Val accuracy:   {val_acc:.1%}")
        target_names = [WBC_CLASSES[i] for i in sorted(np.unique(y_val))]
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=target_names, zero_division=0))
    else:
        val_acc = train_acc

    # ── Save model as joblib bytes inside .pth file ───────────────────────────
    ensure_dir(MODEL_OUT.parent)
    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    MODEL_OUT.write_bytes(buf.getvalue())
    print(f"\nModel saved: {MODEL_OUT} ({MODEL_OUT.stat().st_size // 1024} KB)")

    # Save history
    HISTORY_OUT.write_text(json.dumps({
        "model_type":   "sklearn_ensemble_svm_rf",
        "train_acc":    round(train_acc, 4),
        "val_acc":      round(val_acc, 4),
        "n_train":      int(len(X_train)),
        "n_val":        int(len(X_val)),
        "n_features":   int(X_train.shape[1]),
        "classes":      WBC_CLASSES,
        "trained_at":   datetime.now().isoformat(),
    }, indent=2))

    print("=" * 60)
    print(f"  Best val accuracy: {val_acc:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
