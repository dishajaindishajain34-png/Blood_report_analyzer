"""
train_anemia_fusion.py

Trains a Logistic Regression fusion model combining image-based anemia prediction probability
and CBC-based anemia prediction probability.

Requires paired image + CBC metadata from AneRBC dataset.
If no paired CSV metadata exists under data/anerbc_raw/ containing both patient IDs,
image names, and CBC features (Gender, Hemoglobin, MCV, MCH, MCHC), the script exits cleanly
with a message.

Usage:
    python scripts/train_anemia_fusion.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, project_path

log = get_logger("train_anemia_fusion")

def find_metadata_csv(root: Path) -> Path | None:
    for p in root.rglob("*.csv"):
        # Look for a CSV containing cbc keywords or patient keys
        try:
            df = pd.read_csv(p, nrows=2)
            cols = {c.lower() for c in df.columns}
            if {"gender", "hemoglobin", "mcv", "mch", "mchc"}.issubset(cols) or "hb" in cols:
                return p
        except Exception:
            pass
    return None

def train_fusion():
    log.info("Starting fusion model training...")

    # Load image and cbc models
    from src.anemia_image_model import predict_image_anemia, _load_image_model
    from src.anemia_model import predict_anemia_ml, load_anemia_pipeline

    img_model, img_meta = _load_image_model()
    cbc_pipeline = load_anemia_pipeline()

    if img_model is None or cbc_pipeline is None:
        log.warning("Either image model or CBC model is missing. Train them first!")
        return False

    root = project_path("data", "anerbc_raw")
    csv_path = find_metadata_csv(root)

    if not csv_path or not csv_path.exists():
        log.warning("No paired image-CBC metadata CSV found in %s.", root)
        log.warning("Cannot train Fusion model. Fusion will fall back to average probability.")
        return False

    # Simulate paired predictions using metadata if available
    try:
        df_cbc = pd.read_csv(csv_path)
        # Normalize column names
        df_cbc.columns = [c.strip().lower() for c in df_cbc.columns]

        # Let's map target columns
        # Expected: filename, gender, hemoglobin/hb, mcv, mch, mchc, label/anemia
        # If the CSV does not contain filename/image links, we cannot cleanly match.
        img_col = None
        for col in df_cbc.columns:
            if "image" in col or "file" in col or "filename" in col:
                img_col = col
                break

        if not img_col:
            log.warning("CSV metadata does not contain image filename links. Cannot fuse accurately.")
            return False

        # Generate pairs
        paired_data = []
        for _, row in df_cbc.iterrows():
            filename = str(row[img_col])
            # Find the actual image file on disk
            img_paths = list(root.rglob(filename))
            if not img_paths:
                # Try finding matching name
                img_paths = [p for p in root.rglob("*") if p.name == filename]

            if not img_paths:
                continue

            # Run image model
            from PIL import Image
            img = Image.open(img_paths[0]).convert("RGB")
            img_res = predict_image_anemia(img)

            # Get CBC prob
            # Map gender
            gender_raw = row.get("gender", 0)
            if isinstance(gender_raw, str):
                g = 1 if "m" in gender_raw.lower() else 0
            else:
                g = int(gender_raw)

            hb = float(row.get("hemoglobin", row.get("hb", 12.0)))
            mcv = float(row.get("mcv", 85.0))
            mch = float(row.get("mch", 28.0))
            mchc = float(row.get("mchc", 33.0))

            cbc_res = predict_anemia_ml({
                "Gender": g,
                "Hemoglobin": hb,
                "MCV": mcv,
                "MCH": mch,
                "MCHC": mchc
            })

            label = int(row.get("label", row.get("anemia", 1 if hb < 12.0 else 0)))

            paired_data.append({
                "img_prob": img_res["probability"],
                "cbc_prob": cbc_res["probability"],
                "label": label
            })

        if len(paired_data) < 20:
            log.warning("Insufficient paired samples found (%d samples). Need at least 20.", len(paired_data))
            return False

        df_pair = pd.DataFrame(paired_data)
        X = df_pair[["img_prob", "cbc_prob"]].values
        y = df_pair["label"].values

        # Split
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        fusion_lr = LogisticRegression()
        fusion_lr.fit(X_train, y_train)

        preds = fusion_lr.predict(X_test)
        acc = accuracy_score(y_test, preds)

        log.info("Fusion model test accuracy: %.2f%%", acc * 100)

        # Save
        fusion_dir = project_path("models", "anemia_fusion")
        fusion_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(fusion_lr, fusion_dir / "fusion_model.joblib")

        with open(fusion_dir / "feature_columns.json", "w", encoding="utf-8") as f:
            json.dump(["image_probability", "cbc_probability"], f, indent=2)

        with open(fusion_dir / "model_version.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_name": "Logistic Regression Fusion",
                "model_version": "1.0.0",
                "coefficients": fusion_lr.coef_.tolist(),
                "intercept": fusion_lr.intercept_.tolist()
            }, f, indent=2)

        # Generate metrics
        from sklearn.metrics import precision_recall_fscore_support
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary')

        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "sample_count": len(y)
        }
        with open(fusion_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        log.info("Fusion model successfully saved.")
        return True

    except Exception as e:
        log.error("Failed to train fusion model: %s", e)
        return False

if __name__ == "__main__":
    train_fusion()
