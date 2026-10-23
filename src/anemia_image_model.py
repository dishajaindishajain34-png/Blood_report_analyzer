"""
anemia_image_model.py — Image-Based Anemia Inference Module.

Loads a trained MobileNetV3-Small model (fine-tuned on AneRBC dataset)
and runs binary anemia classification on a single blood-smear PIL Image.

Class mapping:
  0 = Normal / Healthy
  1 = Anemic

Outputs:
  prediction        : str  — 'Anemic' or 'Normal'
  probability       : float — probability of being Anemic (0.0–1.0)
  confidence        : float — max class probability
  threshold         : float — calibrated operating threshold
  model_version     : str
  status            : str  — 'success' | 'model_missing' | 'error'
  message           : str
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, Any

import numpy as np
from PIL import Image

from src.utils import project_path, get_logger

log = get_logger(__name__)

MODEL_DIR = project_path("models", "anemia_image")
WEIGHTS_PATH = MODEL_DIR / "mobilenetv3_anemia.pth"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
CLASS_MAP_PATH = MODEL_DIR / "class_mapping.json"
PREPROC_PATH = MODEL_DIR / "preprocessing_config.json"

# Default preprocessing (matches training config)
_DEFAULT_IMG_SIZE = 224
_DEFAULT_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
_DEFAULT_STD = [0.229, 0.224, 0.225]   # ImageNet std


class ImageAnemiaPrediction(TypedDict, total=False):
    prediction: str          # 'Anemic' | 'Normal'
    is_anemic: bool
    probability: float       # P(Anemic)
    confidence: float        # max class probability
    threshold: float         # operating decision threshold
    model_version: str
    model_name: str
    status: str              # 'success' | 'model_missing' | 'error'
    message: str


_cached_model: Any | None = None
_cached_metadata: dict = {}


def _load_image_model():
    """Load and cache the MobileNetV3-Small anemia model."""
    global _cached_model, _cached_metadata
    if _cached_model is not None:
        return _cached_model, _cached_metadata

    if not WEIGHTS_PATH.exists():
        log.warning("Image anemia model weights not found at %s", WEIGHTS_PATH)
        return None, {}

    try:
        import torch
        import torchvision.models as models

        # Load metadata
        metadata = {}
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        num_classes = metadata.get("num_classes", 2)

        # Build model architecture (must match training)
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        import torch.nn as nn
        model.classifier[-1] = nn.Linear(in_features, num_classes)

        device = torch.device("cpu")
        state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        _cached_model = model
        _cached_metadata = metadata
        log.info("[OK] Loaded image anemia model (v%s)", metadata.get("model_version", "1.0.0"))
        return model, metadata
    except Exception as e:
        log.error("Failed to load image anemia model: %s", e)
        return None, {}


def _preprocess_image(image: Image.Image) -> "torch.Tensor":
    """Preprocess a PIL image to a model-ready tensor."""
    import torch
    from torchvision import transforms

    # Load preprocessing config if available
    preproc = {}
    if PREPROC_PATH.exists():
        with open(PREPROC_PATH, "r", encoding="utf-8") as f:
            preproc = json.load(f)

    img_size = preproc.get("img_size", _DEFAULT_IMG_SIZE)
    mean = preproc.get("mean", _DEFAULT_MEAN)
    std = preproc.get("std", _DEFAULT_STD)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    img_rgb = image.convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0)  # Add batch dim
    return tensor


def predict_image_anemia(image: Image.Image) -> ImageAnemiaPrediction:
    """
    Predict anemia from a blood-smear image using the trained MobileNetV3-Small model.

    Args:
        image: PIL Image (blood smear microscope image)

    Returns:
        ImageAnemiaPrediction TypedDict with prediction, probability, confidence,
        threshold, model_version, status, and message.
    """
    model, metadata = _load_image_model()

    if model is None:
        return {
            "prediction": "Undetermined",
            "is_anemic": False,
            "probability": 0.0,
            "confidence": 0.0,
            "threshold": 0.5,
            "model_version": "N/A",
            "model_name": "MobileNetV3-Small (AneRBC)",
            "status": "model_missing",
            "message": (
                "Image anemia model not yet trained. "
                "Place the AneRBC dataset in data/anerbc_raw/ and run: "
                "python scripts/train_anemia_image.py"
            ),
        }

    try:
        import torch

        tensor = _preprocess_image(image)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()

        prob_anemic = float(probs[1])
        prob_normal = float(probs[0])
        threshold = float(metadata.get("optimal_threshold", 0.5))
        is_anemic = prob_anemic >= threshold
        prediction_str = "Anemic" if is_anemic else "Normal"
        confidence = float(np.max(probs))

        return {
            "prediction": prediction_str,
            "is_anemic": is_anemic,
            "probability": round(prob_anemic, 4),
            "confidence": round(confidence, 4),
            "threshold": round(threshold, 4),
            "model_version": metadata.get("model_version", "1.1.0"),
            "model_name": metadata.get("model_name", "MobileNetV3-Small (AneRBC)"),
            "status": "success",
            "message": "Image-based anemia prediction successful.",
        }
    except Exception as e:
        log.error("Image anemia prediction failed: %s", e)
        return {
            "prediction": "Undetermined",
            "is_anemic": False,
            "probability": 0.0,
            "confidence": 0.0,
            "threshold": 0.5,
            "model_version": metadata.get("model_version", "1.1.0"),
            "model_name": metadata.get("model_name", "MobileNetV3-Small (AneRBC)"),
            "status": "error",
            "message": f"Prediction error: {e}",
        }
