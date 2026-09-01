"""
anemia_fusion.py — Multimodal Anemia Fusion Inference Module.

Combines image-based anemia probability and CBC-based anemia probability
using a trained Logistic Regression fusion model to produce a final
anemia prediction.

If the fusion model is not trained yet, falls back to simple probability
averaging with a clear disclosure.

Fusion input features:
  - image_probability : float  (P(Anemic) from image model, 0.0–1.0)
  - cbc_probability   : float  (P(Anemic) from CBC model, 0.0–1.0)

Outputs:
  fusion_prediction   : str   — 'Anemic' | 'Normal'
  fusion_probability  : float
  fusion_confidence   : float
  fusion_strategy     : str   — 'learned_lr' | 'average_fallback' | 'image_only' | 'cbc_only'
  model_version       : str
  status              : str
  message             : str
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, Any

from src.utils import project_path, get_logger

log = get_logger(__name__)

MODEL_DIR = project_path("models", "anemia_fusion")
FUSION_MODEL_PATH = MODEL_DIR / "fusion_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"
VERSION_PATH = MODEL_DIR / "model_version.json"


class FusionAnemiaPrediction(TypedDict):
    fusion_prediction: str       # 'Anemic' | 'Normal'
    is_anemic: bool
    fusion_probability: float    # P(Anemic) after fusion
    fusion_confidence: float
    fusion_strategy: str         # how fusion was performed
    image_probability: float     # raw image model P(Anemic)
    cbc_probability: float       # raw CBC model P(Anemic)
    model_version: str
    status: str
    message: str


_cached_fusion: Any | None = None
_cached_version: dict = {}


def _load_fusion_model():
    """Load and cache the Logistic Regression fusion model."""
    global _cached_fusion, _cached_version
    if _cached_fusion is not None:
        return _cached_fusion, _cached_version

    if not FUSION_MODEL_PATH.exists():
        log.warning("Fusion model weights not found at %s. Falling back to probability averaging.", FUSION_MODEL_PATH)
        return None, {}

    try:
        import joblib
        model = joblib.load(FUSION_MODEL_PATH)
        version = {}
        if VERSION_PATH.exists():
            with open(VERSION_PATH, "r", encoding="utf-8") as f:
                version = json.load(f)
        _cached_fusion = model
        _cached_version = version
        log.info("[OK] Loaded fusion model (v%s)", version.get("model_version", "1.0.0"))
        return model, version
    except Exception as e:
        log.error("Fusion model loading failed at %s: %s", FUSION_MODEL_PATH, e)
        return None, {}


def predict_fusion(
    image_probability: float | None,
    cbc_probability: float | None,
) -> FusionAnemiaPrediction:
    """
    Produce a final anemia prediction by fusing image and CBC probabilities.

    Strategy priority:
      1. Learned Logistic Regression fusion (if model trained and both inputs available)
      2. Simple average fallback (if both inputs available but model not trained)
      3. Image-only or CBC-only (if only one modality available)
      4. Undetermined (if no valid inputs)

    Args:
        image_probability: P(Anemic) from image model (None if not available)
        cbc_probability:   P(Anemic) from CBC model   (None if not available)

    Returns:
        FusionAnemiaPrediction TypedDict
    """
    img_ok  = image_probability is not None and 0.0 <= image_probability <= 1.0
    cbc_ok  = cbc_probability  is not None and 0.0 <= cbc_probability  <= 1.0

    img_prob = image_probability if img_ok else 0.5
    cbc_prob = cbc_probability  if cbc_ok  else 0.5

    if not img_ok and not cbc_ok:
        return {
            "fusion_prediction": "Undetermined",
            "is_anemic": False,
            "fusion_probability": 0.0,
            "fusion_confidence": 0.0,
            "fusion_strategy": "none",
            "image_probability": img_prob,
            "cbc_probability": cbc_prob,
            "model_version": "N/A",
            "status": "insufficient_data",
            "message": "Neither image nor CBC probability is available.",
        }

    # Try learned fusion
    model, version = _load_fusion_model()
    if model is not None and img_ok and cbc_ok:
        try:
            import numpy as np
            X = np.array([[img_prob, cbc_prob]])
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                fused_prob = float(probs[1])
                confidence = float(max(probs))
            else:
                pred_int = int(model.predict(X)[0])
                fused_prob = 1.0 if pred_int == 1 else 0.0
                confidence = 1.0
            is_anemic = fused_prob > 0.5
            return {
                "fusion_prediction": "Anemic" if is_anemic else "Normal",
                "is_anemic": is_anemic,
                "fusion_probability": round(fused_prob, 4),
                "fusion_confidence": round(confidence, 4),
                "fusion_strategy": "learned_logistic_regression",
                "image_probability": round(img_prob, 4),
                "cbc_probability": round(cbc_prob, 4),
                "model_version": version.get("model_version", "1.0.0"),
                "status": "success",
                "message": "Fusion via trained Logistic Regression model.",
            }
        except Exception as e:
            log.error("Fusion model inference failed: %s - falling back to average.", e)

    # Fallback: simple average or single-modality
    if img_ok and cbc_ok:
        log.info("Learned fusion model not available - using average fallback.")
        fused_prob = (img_prob + cbc_prob) / 2.0
        strategy = "average_fallback"
        msg = (
            "Fallback Fusion - trained fusion model unavailable (no paired image+CBC dataset available for learned fusion training). "
            "Probability averaging applied."
        )
    elif img_ok:
        fused_prob = img_prob
        strategy = "image_only"
        msg = "Only image modality available - using image probability directly."
    else:
        fused_prob = cbc_prob
        strategy = "cbc_only"
        msg = "Only CBC modality available - using CBC probability directly."

    is_anemic = fused_prob > 0.5
    confidence = fused_prob if is_anemic else (1.0 - fused_prob)

    return {
        "fusion_prediction": "Anemic" if is_anemic else "Normal",
        "is_anemic": is_anemic,
        "fusion_probability": round(fused_prob, 4),
        "fusion_confidence": round(confidence, 4),
        "fusion_strategy": strategy,
        "image_probability": round(img_prob, 4),
        "cbc_probability": round(cbc_prob, 4),
        "model_version": version.get("model_version", "N/A") if version else "N/A",
        "status": "success",
        "message": msg,
    }
