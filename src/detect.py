"""
detect.py — YOLOv8-based blood cell detection and counting.

Uses a pre-trained YOLOv8 model fine-tuned on the BCCD dataset.
Default model: keremberke/yolov8n-blood-cell-detection (via Ultralytics Hub)
Falls back to a locally saved best.pt if available.

Class mapping (BCCD):
  0 → Platelet
  1 → RBC
  2 → WBC
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

# Graceful import so the module is importable even without ultralytics installed
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

from src.utils import get_logger, project_path, pil_to_numpy, numpy_to_pil

log = get_logger(__name__)

# ── Model path config ─────────────────────────────────────────────────────────────
_LOCAL_WEIGHTS = project_path("models", "yolov8_blood", "best.pt")
_HF_WEIGHTS_URL = (
    "https://huggingface.co/keremberke/yolov8n-blood-cell-detection"
    "/resolve/main/best.pt"
)

# BCCD class names — must match the downloaded model's .names dict exactly
# Model output: {0: 'Platelets', 1: 'RBC', 2: 'WBC'}
CLASS_NAMES: dict[int, str] = {
    0: "Platelets",
    1: "RBC",
    2: "WBC",
}


# ── Detection result container ────────────────────────────────────────────────────
class DetectionResult:
    """Holds the output of a single detection pass."""

    def __init__(
        self,
        annotated_image: np.ndarray,         # RGB NumPy array with drawn boxes
        raw_counts:      dict[str, int],     # {"rbc": n, "wbc": n, "platelet": n}
        detections:      list[dict],         # Per-box details
        model_name:      str = "",
    ):
        self.annotated_image = annotated_image
        self.raw_counts      = raw_counts
        self.detections      = detections    # [{class, confidence, bbox}, ...]
        self.model_name      = model_name

    @property
    def annotated_pil(self) -> Image.Image:
        return numpy_to_pil(self.annotated_image)

    @property
    def total_cells(self) -> int:
        return sum(self.raw_counts.values())

    @property
    def avg_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return round(
            sum(d["confidence"] for d in self.detections) / len(self.detections), 3
        )

    def __repr__(self) -> str:
        return (
            f"DetectionResult(total={self.total_cells}, "
            f"counts={self.raw_counts}, avg_conf={self.avg_confidence:.2%})"
        )


# ── Model loader ──────────────────────────────────────────────────────────────────
_model_cache: dict[str, "YOLO"] = {}

def _load_model(model_path: str | None = None) -> "YOLO":
    """Load (and cache) the YOLOv8 model."""
    if not _YOLO_AVAILABLE:
        raise ImportError(
            "ultralytics is not installed. Run: pip install ultralytics"
        )

    key = model_path or "local"
    if key in _model_cache:
        log.info("Using cached YOLO model: %s", key)
        return _model_cache[key]

    # 1. Explicit override
    if model_path and Path(model_path).exists():
        log.info("Loading YOLO weights from override: %s", model_path)
        model = YOLO(model_path)

    # 2. Local best.pt (default — always exists after setup)
    elif _LOCAL_WEIGHTS.exists():
        log.info("Loading local YOLO weights: %s", _LOCAL_WEIGHTS)
        model = YOLO(str(_LOCAL_WEIGHTS))

    # 3. Auto-download from HuggingFace
    else:
        import urllib.request
        log.info("Local weights not found. Downloading from HuggingFace...")
        _LOCAL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_HF_WEIGHTS_URL, str(_LOCAL_WEIGHTS))
        log.info("Downloaded weights -> %s", _LOCAL_WEIGHTS)
        model = YOLO(str(_LOCAL_WEIGHTS))

    _model_cache[key] = model
    # Update CLASS_NAMES if the model exposes a .names mapping/list
    try:
        global CLASS_NAMES
        names = getattr(model, "names", None)
        if names is not None:
            # names may be list or dict; normalize to dict[int,str]
            if isinstance(names, dict):
                CLASS_NAMES = {int(k): str(v) for k, v in names.items()}
            else:
                CLASS_NAMES = {int(i): str(n) for i, n in enumerate(names)}
            log.info("Using model-provided class names: %s", CLASS_NAMES)
    except Exception:
        pass
    return model


def warmup_model(model_path: str | None = None) -> None:
    """Pre-warm the YOLOv8 model by running a dummy inference pass."""
    try:
        model = _load_model(model_path)
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(source=dummy_img, verbose=False)
        log.info("YOLO model pre-warmed successfully.")
    except Exception as e:
        log.warning("Failed to pre-warm YOLO model: %s", e)


# ── Core detection function ───────────────────────────────────────────────────────
def detect_cells(
    image: Image.Image | np.ndarray,
    conf_threshold: float = 0.15,
    iou_threshold:  float = 0.40,
    model_path:     str | None = None,
) -> DetectionResult:
    """
    Run YOLOv8 cell detection on a blood smear image.

    Args:
        image:          PIL Image or NumPy RGB array.
        conf_threshold: Minimum confidence to keep a detection.
        iou_threshold:  Non-max suppression IOU threshold.
        model_path:     Override path to custom .pt weights.

    Returns:
        DetectionResult with annotated image, counts, and per-box details.
    """
    model = _load_model(model_path)

    # ── CRITICAL: YOLO expects BGR when given NumPy arrays (like OpenCV).
    # PIL Images are RGB. Passing NumPy RGB to YOLO swaps red/blue channels,
    # causing RBCs (reddish) to be misclassified as WBCs (bluish) and vice versa,
    # producing wildly wrong counts (e.g. RBC=2, WBC=9 instead of RBC=15, WBC=2).
    # Fix: pass PIL Images directly to YOLO (it handles RGB PIL natively).
    # For NumPy arrays, convert RGB→BGR before prediction.
    if isinstance(image, Image.Image):
        yolo_source = image          # PIL RGB — YOLO handles natively ✓
        img_np = pil_to_numpy(image) # Keep RGB NumPy for tiled fallback
    else:
        import cv2 as _cv2
        img_np = image
        yolo_source = _cv2.cvtColor(image, _cv2.COLOR_RGB2BGR)  # RGB→BGR ✓

    log.info("Running YOLO inference (conf=%.2f, iou=%.2f) …", conf_threshold, iou_threshold)

    results = model.predict(
        source    = yolo_source,
        conf      = conf_threshold,
        iou       = iou_threshold,
        verbose   = False,
    )

    result = results[0]  # single image

    # ── Parse detections ──────────────────────────────────────────────────────
    raw_counts: dict[str, int] = {"rbc": 0, "wbc": 0, "platelet": 0}
    detections: list[dict]     = []

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id  = int(box.cls[0].item())
            conf    = float(box.conf[0].item())
            xyxy    = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}").lower()

            # Normalise to our internal keys (model uses 'platelets' plural)
            if cls_name in ("rbc",):
                raw_counts["rbc"] += 1
            elif cls_name in ("wbc",):
                raw_counts["wbc"] += 1
            elif cls_name in ("platelet", "platelets"):
                raw_counts["platelet"] += 1

            detections.append({
                "class":      cls_name,
                "class_id":   cls_id,
                "confidence": round(conf, 4),
                "bbox":       [round(v, 1) for v in xyxy],
            })

    log.info("Detection complete: %s (total=%d)", raw_counts, sum(raw_counts.values()))

    # ── Tiled detection fallback for dense/high-magnification smears ──────────
    # When 0 RBCs are found on the full image, the smear may be a dense high-mag
    # image where cells appear too small at full scale. Sliding-window tiling
    # runs the model at native cell resolution and aggregates with cross-tile NMS.
    if raw_counts["rbc"] == 0 and img_np.shape[0] > 400 and img_np.shape[1] > 400:
        log.info("0 RBCs found on full image — triggering tiled detection for dense smear...")
        tiled_raw, tiled_detections, tiled_annotated = _detect_cells_tiled(
            img_np, model, conf_threshold=0.25, iou_threshold=0.50
        )
        if tiled_raw["rbc"] > 0:
            log.info("Tiled detection found RBCs: %s", tiled_raw)
            return DetectionResult(
                annotated_image = tiled_annotated,
                raw_counts      = tiled_raw,
                detections      = tiled_detections,
                model_name      = str(_LOCAL_WEIGHTS),
            )

    # ── Annotated image ───────────────────────────────────────────────────────
    import cv2
    annotated_bgr = result.plot()          # YOLO returns OpenCV BGR
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return DetectionResult(
        annotated_image = annotated_rgb,
        raw_counts      = raw_counts,
        detections      = detections,
        model_name      = _HF_WEIGHTS_URL if not Path(_LOCAL_WEIGHTS).exists() else str(_LOCAL_WEIGHTS),
    )


# ── Tiled detection helper ────────────────────────────────────────────────────────
def _detect_cells_tiled(
    img_np: np.ndarray,
    model,
    tile_size: int = 320,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50,
) -> tuple[dict, list, np.ndarray]:
    """
    Sliding-window tiled detection for dense/high-magnification blood smears.
    Each tile is converted RGB→BGR before YOLO inference to match training format.

    Args:
        img_np: NumPy RGB array of the full image.
        model:  loaded YOLO model.

    Returns:
        (raw_counts, detections, annotated_rgb)
    """
    import cv2

    h, w = img_np.shape[:2]
    step = tile_size // 2  # 50% overlap

    all_raw_boxes: list = []  # [x1, y1, x2, y2, conf, cls_id]

    for y in range(0, max(1, h - tile_size + 1), step):
        for x in range(0, max(1, w - tile_size + 1), step):
            y2_crop = min(y + tile_size, h)
            x2_crop = min(x + tile_size, w)
            tile_rgb = img_np[y:y2_crop, x:x2_crop]
            if tile_rgb.shape[0] < 32 or tile_rgb.shape[1] < 32:
                continue
            # Convert tile RGB → BGR so YOLO sees correct channel order
            tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
            res = model.predict(source=tile_bgr, conf=conf_threshold, verbose=False)
            for box in res[0].boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                # Translate to original image coordinates
                all_raw_boxes.append([bx1 + x, by1 + y, bx2 + x, by2 + y, conf, cls_id])

    log.info("Tiled detection: %d raw boxes across tiles", len(all_raw_boxes))

    if not all_raw_boxes:
        return {"rbc": 0, "wbc": 0, "platelet": 0}, [], img_np.copy()

    # Per-class NMS deduplication
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        a_area = (a[2] - a[0]) * (a[3] - a[1])
        b_area = (b[2] - b[0]) * (b[3] - b[1])
        union = a_area + b_area - inter
        return inter / (union + 1e-6)

    raw_counts: dict[str, int] = {"rbc": 0, "wbc": 0, "platelet": 0}
    detections: list[dict] = []

    for cls_id in set(int(b[5]) for b in all_raw_boxes):
        cls_boxes = [b for b in all_raw_boxes if int(b[5]) == cls_id]
        cls_boxes.sort(key=lambda b: -b[4])  # sort by confidence descending
        kept = []
        for box in cls_boxes:
            if all(_iou(box, k) < iou_threshold for k in kept):
                kept.append(box)

        cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}").lower()
        if cls_name in ("rbc",):
            raw_counts["rbc"] += len(kept)
        elif cls_name in ("wbc",):
            raw_counts["wbc"] += len(kept)
        elif cls_name in ("platelet", "platelets"):
            raw_counts["platelet"] += len(kept)

        for b in kept:
            detections.append({
                "class":      cls_name,
                "class_id":   cls_id,
                "confidence": round(b[4], 4),
                "bbox":       [round(v, 1) for v in b[:4]],
            })

    log.info("After NMS dedup: %s", raw_counts)

    # Draw annotations on the original RGB image
    annotated = img_np.copy()
    # cv2 uses BGR, so swap color channels in the cv2 calls
    color_map_bgr = {
        "rbc":      (50,  50,  220),  # BGR: red in RGB display
        "wbc":      (50,  180, 50),   # BGR: green
        "platelet": (220, 80,  80),   # BGR: blue in RGB display
    }
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = color_map_bgr.get(det["class"], (200, 200, 200))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        label = f"{det['class'].upper()} {det['confidence']:.0%}"
        cv2.putText(annotated, label, (x1, max(y1 - 2, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    return raw_counts, detections, annotated


# ── WBC crop extractor ────────────────────────────────────────────────────────────
def extract_wbc_crops(
    image:      Image.Image | np.ndarray,
    detections: list[dict],
    pad:        int = 8,
) -> list[Image.Image]:
    """
    Crop each detected WBC from the original image for subtype classification.

    Args:
        image:      Original (unresized) image.
        detections: Detections list from DetectionResult.
        pad:        Pixel padding around each bounding box.

    Returns:
        List of PIL Image crops (one per detected WBC).
    """
    if isinstance(image, np.ndarray):
        img_pil = numpy_to_pil(image)
    else:
        img_pil = image

    w, h = img_pil.size
    crops: list[Image.Image] = []

    for det in detections:
        if det["class"] not in ("wbc",):
            continue
        x1, y1, x2, y2 = det["bbox"]
        x1 = max(0, int(x1) - pad)
        y1 = max(0, int(y1) - pad)
        x2 = min(w, int(x2) + pad)
        y2 = min(h, int(y2) + pad)
        crop = img_pil.crop((x1, y1, x2, y2))
        crops.append(crop)

    log.info("Extracted %d WBC crops for classification.", len(crops))
    return crops


# ── Smoke test ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    import sys

    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_images/sample.jpg"
    if not Path(img_path).exists():
        print(f"Image not found: {img_path}")
        sys.exit(1)

    pil_img = Image.open(img_path).convert("RGB")
    result = detect_cells(pil_img)
    print(result)

    out = "data/sample_images/detection_output.jpg"
    result.annotated_pil.save(out)
    print(f"Annotated image saved → {out}")
