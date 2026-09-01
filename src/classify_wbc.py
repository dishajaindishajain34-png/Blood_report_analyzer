"""
classify_wbc.py — WBC Subtype Classifier (Phase 2).

Classifies each WBC crop into one of 5 subtypes:
  Neutrophil | Lymphocyte | Monocyte | Eosinophil | Basophil

Model priority:
  1. sklearn pipeline (.pth holds joblib bytes) — fast CPU inference, Phase 2
  2. Placeholder mode — distributes using healthy-blood ratios, clearly labelled "Estimated"

Phase 3: A GPU-trained deep model (ViT or EfficientNet) will replace sklearn.
"""

from __future__ import annotations

import io
from pathlib import Path
from PIL import Image
import numpy as np

from src.utils import get_logger, project_path

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────────
WBC_CLASSES: list[str] = [
    "Neutrophil",
    "Lymphocyte",
    "Monocyte",
    "Eosinophil",
    "Basophil",
]

# Healthy differential ratios (used in placeholder mode only)
_HEALTHY_RATIOS: dict[str, float] = {
    "Neutrophil": 0.60,
    "Lymphocyte": 0.30,
    "Monocyte":   0.06,
    "Eosinophil": 0.03,
    "Basophil":   0.01,
}

_WEIGHTS_PATH = project_path("models", "resnet50_wbc", "wbc_classifier_sklearn.pth")
_IMG_SIZE     = 64
_model_cache  = None   # sklearn Pipeline or None


# ── Feature extraction (must match training/train_wbc_sklearn.py) ─────────────────
def _extract_features(img: Image.Image) -> np.ndarray:
    """224-dim feature vector: RGB hist (48) + HSV hist (48) + gradient hist (128)."""
    import colorsys
    img = img.convert("RGB").resize((_IMG_SIZE, _IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    feats = []

    # RGB colour histogram
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0, 1))
        feats.extend(hist / (_IMG_SIZE * _IMG_SIZE))

    # HSV colour histogram — vectorized conversion
    import matplotlib.colors as mcolors
    hsv = mcolors.rgb_to_hsv(arr)
    for ch in range(3):
        hist, _ = np.histogram(hsv[:, :, ch], bins=16, range=(0, 1))
        feats.extend(hist / (_IMG_SIZE * _IMG_SIZE))

    # Gradient magnitude histogram (4×4 grid, 8 bins)
    gray = arr.mean(axis=2)
    gx   = np.gradient(gray, axis=1)
    gy   = np.gradient(gray, axis=0)
    mag  = np.sqrt(gx**2 + gy**2)
    ang  = (np.arctan2(gy, gx) * 180 / np.pi) % 360
    cell = _IMG_SIZE // 4
    for cy in range(4):
        for cx in range(4):
            pm = mag[cy*cell:(cy+1)*cell, cx*cell:(cx+1)*cell]
            pa = ang[cy*cell:(cy+1)*cell, cx*cell:(cx+1)*cell]
            hist, _ = np.histogram(pa, bins=8, range=(0, 360), weights=pm)
            s = hist.sum()
            feats.extend(hist / (s + 1e-8))

    return np.array(feats, dtype=np.float32)


# ── Model loader ──────────────────────────────────────────────────────────────────
def _load_model():
    """Load the WBC classifier (supports both sklearn Pipeline and PyTorch ResNet50)."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not _WEIGHTS_PATH.exists():
        log.warning("WBC classifier weights not found at %s -- fallback ratio estimation active.", _WEIGHTS_PATH)
        return None

    data = _WEIGHTS_PATH.read_bytes()
    
    # 1. Try loading as sklearn pipeline via joblib
    try:
        import joblib
        pipe = joblib.load(io.BytesIO(data))
        _model_cache = ("sklearn", pipe)
        log.info("Loaded sklearn WBC classifier from %s", _WEIGHTS_PATH)
        return _model_cache
    except Exception:
        pass

    # 2. Try loading as PyTorch ResNet50 model
    try:
        import torch
        from torchvision import models
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(WBC_CLASSES))
        state_dict = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            model.load_state_dict(state_dict["state_dict"])
        elif isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        else:
            model = state_dict
        model.eval()
        _model_cache = ("pytorch", model)
        log.info("Loaded PyTorch ResNet50 WBC classifier from %s", _WEIGHTS_PATH)
        return _model_cache
    except Exception as e:
        log.error("Failed to load WBC classifier model: %s -- using ratio fallback.", e)
        return None


# ── Public API ────────────────────────────────────────────────────────────────────
class WBCClassificationResult:
    """Result of WBC subtype classification across all detected WBC crops."""

    def __init__(
        self,
        per_cell:     list[tuple[str, float]],  # [(class_name, confidence), ...]
        total_wbc:    int,
        is_estimated: bool = False,
    ):
        self.per_cell     = per_cell
        self.total_wbc    = total_wbc
        self.is_estimated = is_estimated

    @property
    def differential(self) -> dict[str, float]:
        """Return WBC differential as percentages summing to exactly 100."""
        if hasattr(self, "_soft_diff") and self._soft_diff:
            return self._soft_diff

        if not self.per_cell or self.total_wbc == 0:
            return {cls: 0.0 for cls in WBC_CLASSES}

        counts: dict[str, int] = {cls: 0 for cls in WBC_CLASSES}
        for cls_name, _ in self.per_cell:
            if cls_name in counts:
                counts[cls_name] += 1

        total    = len(self.per_cell)
        raw_pcts = {cls: (cnt / total) * 100 for cls, cnt in counts.items()}

        # Largest-remainder rounding to guarantee sum == 100
        floored   = {cls: int(pct) for cls, pct in raw_pcts.items()}
        remainder = 100 - sum(floored.values())
        fractions = sorted(raw_pcts.keys(), key=lambda k: -(raw_pcts[k] % 1))
        for cls in fractions[:remainder]:
            floored[cls] += 1

        return {cls: float(floored[cls]) for cls in WBC_CLASSES}

    @property
    def avg_confidence(self) -> float:
        if not self.per_cell:
            return 0.0
        return round(sum(c for _, c in self.per_cell) / len(self.per_cell), 3)

    def __repr__(self) -> str:
        mode = "estimated" if self.is_estimated else "sklearn"
        return (
            f"WBCClassificationResult(total={self.total_wbc}, "
            f"mode={mode}, diff={self.differential})"
        )


def classify_wbc_crops(
    crops:     list[Image.Image],
    total_wbc: int,
) -> WBCClassificationResult:
    """
    Classify WBC crops into subtypes using loaded model (PyTorch or sklearn).
    Falls back to ratio estimation if no trained model or crops are provided.
    """
    model_info = _load_model()

    # ── Fallback mode ─────────────────────────────────────────────────────────
    if model_info is None or not crops:
        log.info(
            "WBC classifier: Fallback mode. Distributing %d WBCs by healthy ratios.", total_wbc
        )
        if total_wbc == 0:
            result = WBCClassificationResult(per_cell=[], total_wbc=0, is_estimated=True)
            result._soft_diff = {cls: 0.0 for cls in WBC_CLASSES}
            return result

        # Build per_cell proportionally WITHOUT forcing max(1, ...).
        # Using max(1,...) caused every subtype to be equal when very few WBCs are detected.
        per_cell: list[tuple[str, float]] = []
        for cls_name, ratio in _HEALTHY_RATIOS.items():
            count = round(total_wbc * ratio)
            per_cell.extend([(cls_name, 0.60)] * count)

        # Use canonical healthy ratios as the soft differential directly.
        # This gives correct proportions regardless of total_wbc count.
        result = WBCClassificationResult(per_cell=per_cell, total_wbc=total_wbc, is_estimated=True)
        raw_pcts  = {cls: ratio * 100.0 for cls, ratio in _HEALTHY_RATIOS.items()}
        floored   = {cls: int(pct) for cls, pct in raw_pcts.items()}
        remainder = 100 - sum(floored.values())
        fractions = sorted(raw_pcts.keys(), key=lambda k: -(raw_pcts[k] % 1))
        for cls in fractions[:remainder]:
            floored[cls] += 1
        result._soft_diff = {cls: float(floored[cls]) for cls in WBC_CLASSES}
        return result

    model_type, model = model_info
    log.info("WBC classifier: %s inference on %d crops.", model_type, len(crops))
    per_cell = []
    class_proba_sum = {cls: 0.0 for cls in WBC_CLASSES}

    if model_type == "pytorch":
        import torch
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        with torch.no_grad():
            for crop in crops:
                try:
                    t = tf(crop.convert("RGB")).unsqueeze(0)
                    out = model(t)
                    proba = torch.softmax(out, dim=1)[0].numpy()
                    idx = int(np.argmax(proba))
                    conf = float(proba[idx])
                    cls_name = WBC_CLASSES[idx]
                    per_cell.append((cls_name, round(conf, 4)))
                    for c_idx, c_name in enumerate(WBC_CLASSES):
                        class_proba_sum[c_name] += float(proba[c_idx])
                except Exception as e:
                    log.warning("PyTorch inference failed for crop: %s", e)
                    per_cell.append(("Neutrophil", 0.5))
                    class_proba_sum["Neutrophil"] += 1.0
    else:
        for crop in crops:
            try:
                feat   = _extract_features(crop).reshape(1, -1)
                proba  = model.predict_proba(feat)[0]
                idx    = int(np.argmax(proba))
                conf   = float(proba[idx])
                cls_name = WBC_CLASSES[idx]
                per_cell.append((cls_name, round(conf, 4)))
                for c_idx, c_name in enumerate(WBC_CLASSES):
                    class_proba_sum[c_name] += float(proba[c_idx])
            except Exception as e:
                log.warning("Feature extraction failed for crop: %s", e)
                per_cell.append(("Neutrophil", 0.5))
                class_proba_sum["Neutrophil"] += 1.0

    result = WBCClassificationResult(per_cell=per_cell, total_wbc=total_wbc, is_estimated=False)

    # Override differential using soft probability aggregation across crops for smooth physiological breakdown
    if crops and sum(class_proba_sum.values()) > 0:
        total_p = sum(class_proba_sum.values())
        raw_pcts = {cls: (val / total_p) * 100.0 for cls, val in class_proba_sum.items()}
        floored   = {cls: int(pct) for cls, pct in raw_pcts.items()}
        remainder = 100 - sum(floored.values())
        fractions = sorted(raw_pcts.keys(), key=lambda k: -(raw_pcts[k] % 1))
        for cls in fractions[:remainder]:
            floored[cls] += 1
        result._soft_diff = {cls: float(floored[cls]) for cls in WBC_CLASSES}

    return result


# ── Smoke test ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = classify_wbc_crops(crops=[], total_wbc=25)
    print(result)
    print("Differential:", result.differential)
    print("Sum:", sum(result.differential.values()))
    print("Mode: estimated" if result.is_estimated else "Mode: sklearn model")
