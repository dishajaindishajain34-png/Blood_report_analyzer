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

GMU_EXPERIMENTAL_CKPT = MODEL_DIR / "gmu_fusion_experimental.pth"
GMU_SCALER_PATH = MODEL_DIR / "gmu_experimental_scaler.joblib"
GMU_METADATA_PATH = MODEL_DIR / "gmu_experimental_metadata.json"


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
    cbc_params: dict | None = None,
    image_details: dict | None = None,
) -> FusionAnemiaPrediction:
    """
    Produce a final anemia prediction by fusing image and CBC features
    using an advanced Multimodal Stacking Ensemble (HistGradientBoosting + ExtraTrees + MLP).

    Strategy priority:
      1. Learned Stacking Ensemble fusion (if model trained and both inputs available)
      2. Simple average fallback (if both inputs available but model not trained)
      3. Image-only or CBC-only (if only one modality available)
      4. Undetermined (if no valid inputs)

    Args:
        image_probability: P(Anemic) from image model (None if not available)
        cbc_probability:   P(Anemic) from CBC model   (None if not available)
        cbc_params:        Optional raw CBC dictionary (hemoglobin, mcv, mch, mchc, gender)
        image_details:     Optional dictionary containing image confidence and entropy

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
            import math
            import numpy as np

            n_features = getattr(model, "n_features_in_", 2)
            if n_features == 12:
                # 12-dimensional Multimodal Feature Vector
                # 1. Image stream
                img_conf = float(image_details.get("confidence", abs(img_prob - 0.5) * 2.0)) if image_details else float(abs(img_prob - 0.5) * 2.0)
                p_safe = float(np.clip(img_prob, 1e-6, 1.0 - 1e-6))
                img_ent = float(image_details.get("entropy", -(p_safe * math.log2(p_safe) + (1.0 - p_safe) * math.log2(1.0 - p_safe)))) if image_details else float(-(p_safe * math.log2(p_safe) + (1.0 - p_safe) * math.log2(1.0 - p_safe)))

                # 2. CBC stream
                if cbc_params:
                    g_val = cbc_params.get("Gender", cbc_params.get("gender", 0))
                    g = 1 if "m" in str(g_val).lower() or str(g_val) == "1" else 0
                    hb = float(cbc_params.get("Hemoglobin", cbc_params.get("hb", 12.0)) or 12.0)
                    mcv = float(cbc_params.get("MCV", cbc_params.get("mcv", 85.0)) or 85.0)
                    mch = float(cbc_params.get("MCH", cbc_params.get("mch", 28.0)) or 28.0)
                    mchc = float(cbc_params.get("MCHC", cbc_params.get("mchc", 33.0)) or 33.0)
                else:
                    # Robust clinical imputation consistent with cbc_prob
                    if cbc_prob > 0.5:
                        g, hb, mcv, mch, mchc = 0, 9.8, 79.0, 25.0, 31.0
                    else:
                        g, hb, mcv, mch, mchc = 0, 13.8, 89.0, 29.5, 33.5

                mcv_hb_ratio = float(mcv / max(hb, 0.1))
                discordance = float(abs(img_prob - cbc_prob))
                synergy = float(img_prob * cbc_prob)

                X = np.array([[
                    img_prob, img_conf, img_ent,
                    cbc_prob, hb, mcv, mch, mchc,
                    g, mcv_hb_ratio, discordance, synergy
                ]])
            else:
                # 2-feature fallback model
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
            model_name = version.get("model_name", "Multimodal Stacking Ensemble" if n_features == 12 else "Logistic Regression")
            if n_features == 12 and ("stacking" in model_name.lower() or "ensemble" in model_name.lower()):
                strategy_name = "learned_stacking_ensemble"
            else:
                strategy_name = "learned_logistic_regression"

            return {
                "fusion_prediction": "Anemic" if is_anemic else "Normal",
                "is_anemic": is_anemic,
                "fusion_probability": round(fused_prob, 4),
                "fusion_confidence": round(confidence, 4),
                "fusion_strategy": strategy_name,
                "image_probability": round(img_prob, 4),
                "cbc_probability": round(cbc_prob, 4),
                "model_version": version.get("model_version", "2.0.0" if n_features == 12 else "1.0.0"),
                "status": "success",
                "message": f"Fusion via trained {model_name} model.",
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


# ---------------------------------------------------------------------------
# Experimental GMU Inference Pipeline (Academic Prototype)
# ---------------------------------------------------------------------------

_cached_gmu_model: Any | None = None
_cached_gmu_scaler: Any | None = None
_cached_gmu_metadata: dict = {}


def _load_gmu_experimental():
    """Load and cache the trained experimental GMU model and CBC scaler."""
    global _cached_gmu_model, _cached_gmu_scaler, _cached_gmu_metadata
    if _cached_gmu_model is not None:
        return _cached_gmu_model, _cached_gmu_scaler, _cached_gmu_metadata

    if not GMU_EXPERIMENTAL_CKPT.exists():
        log.warning("Experimental GMU checkpoint not found at %s", GMU_EXPERIMENTAL_CKPT)
        return None, None, {}

    try:
        import torch
        import joblib
        from src.models.gmu_fusion import GMUAnemiaClassifier

        meta = {}
        if GMU_METADATA_PATH.exists():
            with open(GMU_METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)

        scaler = None
        if GMU_SCALER_PATH.exists():
            scaler = joblib.load(GMU_SCALER_PATH)

        model = GMUAnemiaClassifier(cbc_features=5, latent_dim=128)
        sd = torch.load(GMU_EXPERIMENTAL_CKPT, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        model.eval()

        _cached_gmu_model = model
        _cached_gmu_scaler = scaler
        _cached_gmu_metadata = meta
        log.info("[OK] Loaded experimental GMU model (v%s)", meta.get("model_version", "1.0.0-experimental-trained"))
        return model, scaler, meta
    except Exception as e:
        log.error("Failed to load experimental GMU model: %s", e)
        return None, None, {}


def predict_gmu_experimental(
    image: Any | None = None,
    cbc_params: dict | None = None,
) -> dict:
    """
    Run experimental inference using the trained PyTorch Gated Multimodal Unit (GMU).

    Academic prototype notice:
      Trained on synthetic combinations of image and CBC records.
      Not validated on genuine paired patient clinical data.

    Args:
        image: PIL.Image, torch.Tensor (224x224x3 or 576-dim), or None if missing.
        cbc_params: dict with Gender, Hemoglobin, MCV, MCH, MCHC, or None if missing.

    Returns:
        dict with prediction, probability, confidence, strategy, gate_routing, and status.
    """
    import torch
    import numpy as np

    has_img = image is not None
    has_cbc = cbc_params is not None and isinstance(cbc_params, dict) and any(v is not None for v in cbc_params.values())

    if not has_img and not has_cbc:
        return {
            "fusion_prediction": "Undetermined",
            "is_anemic": False,
            "fusion_probability": 0.0,
            "fusion_confidence": 0.0,
            "fusion_strategy": "none",
            "latent_gate_mean": 0.0,
            "model_version": "N/A",
            "status": "insufficient_data",
            "message": "Neither image nor CBC modality was provided for GMU inference.",
            "fusion_mode": "experimental_gmu",
            "is_experimental": True,
            "limitation_notice": (
                "Academic Prototype Notice: Trained on synthetic image-CBC pairings. "
                "Not validated on genuine clinical cohorts."
            ),
        }

    model, scaler, meta = _load_gmu_experimental()
    if model is None:
        return {
            "fusion_prediction": "Undetermined",
            "is_anemic": False,
            "fusion_probability": 0.0,
            "fusion_confidence": 0.0,
            "fusion_strategy": "experimental_gmu",
            "latent_gate_mean": 0.0,
            "model_version": "N/A",
            "status": "model_missing",
            "message": (
                "Experimental GMU checkpoint not found. "
                "Train it with: python scripts/train_gmu_fusion.py"
            ),
            "fusion_mode": "experimental_gmu",
            "is_experimental": True,
            "limitation_notice": (
                "Academic Prototype Notice: Trained on synthetic image-CBC pairings. "
                "Not validated on genuine clinical cohorts."
            ),
        }

    try:
        device = torch.device("cpu")

        # 1. Process Image Modality
        if has_img:
            if isinstance(image, torch.Tensor):
                img_tensor = image.to(device)
                if img_tensor.dim() == 1:
                    img_tensor = img_tensor.unsqueeze(0)
            else:
                from src.anemia_image_model import _preprocess_image
                img_tensor = _preprocess_image(image).to(device)
            mask_img = torch.ones((1, 1), device=device)
        else:
            img_tensor = torch.zeros((1, 576), device=device)
            mask_img = torch.zeros((1, 1), device=device)

        # 2. Process CBC Modality
        if has_cbc:
            g_raw = cbc_params.get("Gender", cbc_params.get("gender", 0))
            g = 1.0 if "m" in str(g_raw).lower() or str(g_raw) == "1" else 0.0
            hb = float(cbc_params.get("Hemoglobin", cbc_params.get("hb", 12.0)) or 12.0)
            mcv = float(cbc_params.get("MCV", cbc_params.get("mcv", 85.0)) or 85.0)
            mch = float(cbc_params.get("MCH", cbc_params.get("mch", 28.0)) or 28.0)
            mchc = float(cbc_params.get("MCHC", cbc_params.get("mchc", 33.0)) or 33.0)

            raw_cbc = np.array([[g, hb, mcv, mch, mchc]])
            scaled_cbc = scaler.transform(raw_cbc)[0] if scaler else raw_cbc[0]
            cbc_tensor = torch.tensor(scaled_cbc, dtype=torch.float32, device=device).unsqueeze(0)
            mask_cbc = torch.ones((1, 1), device=device)
        else:
            cbc_tensor = torch.zeros((1, 5), dtype=torch.float32, device=device)
            mask_cbc = torch.zeros((1, 1), device=device)

        # 3. GMU Forward Pass
        with torch.no_grad():
            out = model(img_tensor, cbc_tensor, mask_img=mask_img, mask_cbc=mask_cbc)
            prob = float(out["probability"].item())
            gate_mean = float(out["latent_gate_mean"].item())

        is_anemic = prob >= 0.5
        conf = prob if is_anemic else (1.0 - prob)

        if has_img and has_cbc:
            strat = "experimental_gmu_dual"
        elif has_img:
            strat = "experimental_gmu_image_only"
        else:
            strat = "experimental_gmu_cbc_only"

        return {
            "fusion_prediction": "Anemic" if is_anemic else "Normal",
            "is_anemic": is_anemic,
            "fusion_probability": round(prob, 4),
            "fusion_confidence": round(conf, 4),
            "fusion_strategy": strat,
            "latent_gate_mean": round(gate_mean, 4),
            "model_version": meta.get("model_version", "1.0.0-experimental-trained"),
            "status": "success",
            "message": "Experimental GMU inference executed (trained on synthetic combinations; research prototype).",
            "fusion_mode": "experimental_gmu",
            "is_experimental": True,
            "limitation_notice": (
                "Academic Prototype Notice: Trained on synthetic image-CBC pairings. "
                "Not validated on genuine clinical cohorts."
            ),
        }
    except Exception as e:
        log.error("Experimental GMU inference error: %s", e)
        return {
            "fusion_prediction": "Undetermined",
            "is_anemic": False,
            "fusion_probability": 0.0,
            "fusion_confidence": 0.0,
            "fusion_strategy": "experimental_gmu",
            "latent_gate_mean": 0.0,
            "model_version": meta.get("model_version", "1.0.0-experimental-trained"),
            "status": "error",
            "message": f"GMU inference failed: {e}",
            "fusion_mode": "experimental_gmu",
            "is_experimental": True,
            "limitation_notice": (
                "Academic Prototype Notice: Trained on synthetic image-CBC pairings. "
                "Not validated on genuine clinical cohorts."
            ),
        }

