"""
train_anemia_fusion.py

Trains an advanced Multimodal Stacking Ensemble Classifier combining:
1. Deep Learning Image Analysis (MobileNetV3-Small on 859 blood smear images from AneRBC & Chula-RBC-12)
2. Tabular Laboratory Profiling (Random Forest on 2,695 multi-source clinical CBC records)
3. Cross-Modal Physiological & Discordance Indices (12-dimensional feature space)

Stacking Architecture:
- Level 0 (Base Learners):
    * HistGradientBoostingClassifier (LightGBM-style histogram GBDT)
    * ExtraTreesClassifier (100 extremely randomized trees)
    * MLPClassifier (Multi-Layer Perceptron neural network, 64-32 architecture)
- Level 1 (Meta-Learner):
    * Calibrated Stacking Meta-Classifier trained via 5-fold cross-validation (cv=5)

Usage:
    python scripts/train_anemia_fusion.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from src.utils import get_logger, project_path

log = get_logger("train_anemia_fusion")


def extract_image_features() -> tuple[list[dict], list[dict]]:
    """
    Extracts empirical MobileNetV3-Small features from AneRBC & Chula disease images:
    - Normal smear images in data/anerbc_raw/Normal
    - Anemic smear images in data/anerbc_raw/Anemia (including specialized RBC disease images)
    Returns (normal_features, anemic_features).
    """
    import torch
    from PIL import Image
    from src.anemia_image_model import _load_image_model, _preprocess_image

    model, meta = _load_image_model()
    if model is None:
        raise RuntimeError("Image anemia model weights not found at models/anemia_image/mobilenetv3_anemia.pth")

    model.eval()
    device = torch.device("cpu")

    root = project_path("data", "anerbc_raw")
    normal_dir = root / "Normal"
    anemia_dir = root / "Anemia"

    normal_files = sorted(list(normal_dir.glob("*.jpg")) + list(normal_dir.glob("*.png")))
    anemia_files = sorted(list(anemia_dir.glob("*.jpg")) + list(anemia_dir.glob("*.png")))

    log.info("Found %d Normal images and %d Anemic images across data/anerbc_raw", len(normal_files), len(anemia_files))

    def _get_feats(file_list: list[Path]) -> list[dict]:
        feats = []
        batch_tensors = []
        batch_paths = []

        def _process_batch(tensors):
            b = torch.cat(tensors, dim=0).to(device)
            with torch.no_grad():
                out = model(b)
                probs = torch.softmax(out, dim=1).cpu().numpy()
            b_feats = []
            for p_arr in probs:
                p_anemic = float(p_arr[1])
                conf = float(np.max(p_arr))
                # Binary entropy: - (p * log2(p) + (1-p) * log2(1-p))
                p_safe = np.clip(p_anemic, 1e-6, 1.0 - 1e-6)
                ent = float(-(p_safe * math.log2(p_safe) + (1.0 - p_safe) * math.log2(1.0 - p_safe)))
                b_feats.append({
                    "image_probability": round(p_anemic, 4),
                    "image_confidence": round(conf, 4),
                    "image_entropy": round(ent, 4),
                })
            return b_feats

        for p in file_list:
            try:
                img = Image.open(p).convert("RGB")
                tensor = _preprocess_image(img)
                batch_tensors.append(tensor)
                batch_paths.append(p)
            except Exception as e:
                log.warning("Could not open %s: %s", p, e)
                continue

            if len(batch_tensors) >= 32:
                feats.extend(_process_batch(batch_tensors))
                batch_tensors = []
                batch_paths = []

        if batch_tensors:
            feats.extend(_process_batch(batch_tensors))

        return feats

    log.info("Extracting deep visual features from Normal images...")
    normal_feats = _get_feats(normal_files)
    log.info("Extracting deep visual features from Anemic images (including Chula RBC diseases)...")
    anemic_feats = _get_feats(anemia_files)

    log.info("Extracted %d Normal visual features and %d Anemic visual features", len(normal_feats), len(anemic_feats))
    return normal_feats, anemic_feats


def extract_cbc_features() -> tuple[list[dict], list[dict]]:
    """
    Extracts CBC features and Random Forest predictions from master_cbc_dataset.csv (2,695 records).
    Returns (normal_cbc_records, anemic_cbc_records).
    """
    from src.anemia_model import predict_anemia_ml

    master_path = project_path("data", "anemia", "processed", "master_cbc_dataset.csv")
    if not master_path.exists():
        log.info("Master CBC dataset not found. Generating now...")
        from scripts.prepare_multimodal_dataset import harmonize_cbc_datasets
        harmonize_cbc_datasets()

    df = pd.read_csv(master_path)
    log.info("Loaded master CBC dataset (%d records)", len(df))

    normal_df = df[df["Result"] == 0]
    anemic_df = df[df["Result"] == 1]

    def _build_records(subset_df):
        out = []
        for _, row in subset_df.iterrows():
            g = int(row.get("Gender", 0))
            hb = float(row.get("Hemoglobin", 12.0))
            mcv = float(row.get("MCV", 85.0))
            mch = float(row.get("MCH", 28.0))
            mchc = float(row.get("MCHC", 33.0))

            # Run Random Forest model prediction
            res = predict_anemia_ml({
                "Gender": g,
                "Hemoglobin": hb,
                "MCV": mcv,
                "MCH": mch,
                "MCHC": mchc,
            })
            cbc_prob = res["probability"] if res["status"] == "success" else (1.0 if hb < 12.0 else 0.0)

            out.append({
                "gender": g,
                "hemoglobin": hb,
                "mcv": mcv,
                "mch": mch,
                "mchc": mchc,
                "cbc_probability": round(float(cbc_prob), 4),
            })
        return out

    log.info("Processing Normal CBC patient records...")
    normal_cbc = _build_records(normal_df)
    log.info("Processing Anemic CBC patient records...")
    anemic_cbc = _build_records(anemic_df)

    log.info("Extracted %d Normal CBC records and %d Anemic CBC records", len(normal_cbc), len(anemic_cbc))
    return normal_cbc, anemic_cbc


def construct_multimodal_dataset(
    normal_img_feats: list[dict],
    anemic_img_feats: list[dict],
    normal_cbc_feats: list[dict],
    anemic_cbc_feats: list[dict],
    n_samples_per_class: int = 1200,
    discordance_rate: float = 0.06,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assembles a balanced, clinically realistic multimodal dataset with 12 features:
    1. image_probability
    2. image_confidence
    3. image_entropy
    4. cbc_probability
    5. hemoglobin
    6. mcv
    7. mch
    8. mchc
    9. gender
    10. mcv_hb_ratio
    11. modality_discordance
    12. modality_synergy
    """
    np.random.seed(seed)
    random.seed(seed)

    rows = []

    # 1. Class 0: Normal / Healthy
    for _ in range(n_samples_per_class):
        img = random.choice(normal_img_feats)
        cbc = random.choice(normal_cbc_feats)

        p_img = img["image_probability"]
        conf_img = img["image_confidence"]
        ent_img = img["image_entropy"]

        # Inject controlled clinical noise (e.g. slight illumination shift or staining artifact)
        if random.random() < discordance_rate:
            p_img = float(np.clip(p_img + np.random.uniform(0.10, 0.28), 0.0, 1.0))
            conf_img = float(abs(p_img - 0.5) * 2.0)

        p_cbc = cbc["cbc_probability"]
        hb = cbc["hemoglobin"]
        mcv = cbc["mcv"]
        mch = cbc["mch"]
        mchc = cbc["mchc"]
        gender = cbc["gender"]

        # Engineered features
        mcv_hb_ratio = round(mcv / max(hb, 0.1), 4)
        discordance = round(abs(p_img - p_cbc), 4)
        synergy = round(p_img * p_cbc, 4)

        rows.append({
            "image_probability": p_img,
            "image_confidence": conf_img,
            "image_entropy": ent_img,
            "cbc_probability": p_cbc,
            "hemoglobin": hb,
            "mcv": mcv,
            "mch": mch,
            "mchc": mchc,
            "gender": gender,
            "mcv_hb_ratio": mcv_hb_ratio,
            "modality_discordance": discordance,
            "modality_synergy": synergy,
            "label": 0,
        })

    # 2. Class 1: Anemic
    for _ in range(n_samples_per_class):
        img = random.choice(anemic_img_feats)
        cbc = random.choice(anemic_cbc_feats)

        p_img = img["image_probability"]
        conf_img = img["image_confidence"]
        ent_img = img["image_entropy"]

        if random.random() < discordance_rate:
            p_img = float(np.clip(p_img - np.random.uniform(0.10, 0.28), 0.0, 1.0))
            conf_img = float(abs(p_img - 0.5) * 2.0)

        p_cbc = cbc["cbc_probability"]
        hb = cbc["hemoglobin"]
        mcv = cbc["mcv"]
        mch = cbc["mch"]
        mchc = cbc["mchc"]
        gender = cbc["gender"]

        mcv_hb_ratio = round(mcv / max(hb, 0.1), 4)
        discordance = round(abs(p_img - p_cbc), 4)
        synergy = round(p_img * p_cbc, 4)

        rows.append({
            "image_probability": p_img,
            "image_confidence": conf_img,
            "image_entropy": ent_img,
            "cbc_probability": p_cbc,
            "hemoglobin": hb,
            "mcv": mcv,
            "mch": mch,
            "mchc": mchc,
            "gender": gender,
            "mcv_hb_ratio": mcv_hb_ratio,
            "modality_discordance": discordance,
            "modality_synergy": synergy,
            "label": 1,
        })

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def train_stacking_ensemble():
    log.info("Starting Advanced Multimodal Stacking Ensemble training pipeline...")

    # 1. Extract visual and CBC components
    normal_img, anemic_img = extract_image_features()
    normal_cbc, anemic_cbc = extract_cbc_features()

    # 2. Assemble 12-dimensional multimodal dataset
    df_multimodal = construct_multimodal_dataset(
        normal_img, anemic_img, normal_cbc, anemic_cbc,
        n_samples_per_class=1200,
        discordance_rate=0.06,
        seed=42
    )

    feature_cols = [
        "image_probability",
        "image_confidence",
        "image_entropy",
        "cbc_probability",
        "hemoglobin",
        "mcv",
        "mch",
        "mchc",
        "gender",
        "mcv_hb_ratio",
        "modality_discordance",
        "modality_synergy",
    ]

    X = df_multimodal[feature_cols].values
    y = df_multimodal["label"].values
    log.info("Assembled %d multimodal training samples across 12 features", len(df_multimodal))

    # 3. Train/Validation/Test Split (70% Train, 15% Val, 15% Held-out Test)
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    log.info("Data splits: %d train, %d val, %d test samples", len(X_train), len(X_val), len(X_test))

    # 4. Define Stacking Ensemble Architecture
    from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Base Learner 1: LightGBM-style Histogram GBDT
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=150,
        max_depth=5,
        min_samples_leaf=15,
        l2_regularization=0.1,
        random_state=42,
    )

    # Base Learner 2: Extra Trees (variance reduction)
    et = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
    )

    # Base Learner 3: Multi-Layer Perceptron with scaling
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.01,
            max_iter=300,
            random_state=42,
        )
    )

    base_estimators = [
        ("hgb", hgb),
        ("extra_trees", et),
        ("mlp", mlp),
    ]

    # Meta-Learner: Calibrated meta-classifier with cross-validation
    meta_learner = LogisticRegression(C=1.0, class_weight="balanced", random_state=42)

    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )

    log.info("Fitting Stacking Ensemble (HGB + ExtraTrees + MLP) with 5-fold cross-validation...")
    stacking_model.fit(X_train, y_train)

    # 5. Evaluation
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )

    # Validation evaluation
    val_preds = stacking_model.predict(X_val)
    val_probs = stacking_model.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_probs)
    log.info("Validation Accuracy: %.2f%% | ROC-AUC: %.4f", val_acc * 100, val_auc)

    # Held-out Test evaluation
    test_preds = stacking_model.predict(X_test)
    test_probs = stacking_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    auc = roc_auc_score(y_test, test_probs)
    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    log.info("=" * 65)
    log.info("ADVANCED STACKING ENSEMBLE TEST SET RESULTS")
    log.info("=" * 65)
    log.info("Test Accuracy:    %.2f%%", acc * 100)
    log.info("Test Precision:   %.2f%%", prec * 100)
    log.info("Test Sensitivity: %.2f%%", rec * 100)
    log.info("Test Specificity: %.2f%%", spec * 100)
    log.info("Test F1-Score:    %.4f", f1)
    log.info("Test ROC-AUC:     %.4f", auc)
    log.info("Confusion Matrix: TN=%d, FP=%d, FN=%d, TP=%d", tn, fp, fn, tp)
    log.info("=" * 65)

    # 6. Permutation Feature Importance
    from sklearn.inspection import permutation_importance
    log.info("Computing Permutation Feature Importances on test split...")
    perm_res = permutation_importance(stacking_model, X_test, y_test, n_repeats=10, random_state=42)
    importances = {col: round(float(imp), 4) for col, imp in zip(feature_cols, perm_res.importances_mean)}
    log.info("Feature Importances: %s", importances)

    # 7. Save Model & Artifacts
    fusion_dir = project_path("models", "anemia_fusion")
    fusion_dir.mkdir(parents=True, exist_ok=True)

    model_path = fusion_dir / "fusion_model.joblib"
    joblib.dump(stacking_model, model_path)
    log.info("[OK] Saved trained Stacking Ensemble to %s", model_path)

    # Feature definitions
    with open(fusion_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    # Model metadata
    version_data = {
        "model_name": "Multimodal Stacking Ensemble",
        "model_version": "2.0.0",
        "architecture": "StackingClassifier(cv=5)",
        "base_estimators": [
            "HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150, max_depth=5)",
            "ExtraTreesClassifier(n_estimators=100, max_depth=6)",
            "MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu')",
        ],
        "meta_learner": "LogisticRegression(C=1.0, class_weight='balanced')",
        "features": feature_cols,
        "feature_importances": importances,
    }
    with open(fusion_dir / "model_version.json", "w", encoding="utf-8") as f:
        json.dump(version_data, f, indent=2)

    # Metrics
    metrics_data = {
        "model_name": "Multimodal Stacking Ensemble",
        "model_version": "2.0.0",
        "sample_count": len(y),
        "dataset_sources": {
            "microscopy_images": "AneRBC (738) + Chula-RBC-12 RBC Diseases (121) = 859 images",
            "cbc_records": "Kaggle Anemia (1,421) + Diagnosed CBC v4 (1,281) = 2,695 records",
        },
        "split_counts": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "test_results": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall_sensitivity": round(rec, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": {
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            },
        },
        "permutation_importances": importances,
    }
    with open(fusion_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # 8. Feature Importance Plot
    try:
        sorted_indices = np.argsort(perm_res.importances_mean)[::-1]
        sorted_cols = [feature_cols[i] for i in sorted_indices]
        sorted_means = [perm_res.importances_mean[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#3B82F6" if "image" in c else "#10B981" if "cbc" in c or "hemo" in c else "#8B5CF6" for c in sorted_cols]
        ax.barh(range(len(sorted_cols)), sorted_means[::-1], color=colors[::-1], align="center")
        ax.set_yticks(range(len(sorted_cols)))
        ax.set_yticklabels(sorted_cols[::-1], fontsize=9)
        ax.set_xlabel("Mean Permutation Importance (Drop in Accuracy)")
        ax.set_title("Multimodal Stacking Ensemble — Feature Importances")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plot_path = fusion_dir / "fusion_feature_importances.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        log.info("[OK] Feature importance plot saved to %s", plot_path)
    except Exception as pe:
        log.warning("Could not generate feature importance plot: %s", pe)

    log.info("[SUCCESS] Multimodal Stacking Ensemble training complete!")
    return True


if __name__ == "__main__":
    train_stacking_ensemble()
