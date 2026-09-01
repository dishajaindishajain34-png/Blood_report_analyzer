"""
preprocess.py — OpenCV image preprocessing pipeline for blood smear images.

Steps:
  1. Load image (from path or NumPy array)
  2. Convert to RGB
  3. Gaussian blur (noise removal)
  4. CLAHE on L-channel (contrast enhancement)
  5. Resize to target dimensions
  6. Return preprocessed image alongside a display-ready copy
"""

import cv2
import numpy as np
from PIL import Image
from src.utils import get_logger, pil_to_numpy, numpy_to_pil

log = get_logger(__name__)

# Default YOLO input size
TARGET_SIZE: tuple[int, int] = (640, 640)


def preprocess(
    image_input: Image.Image | np.ndarray,
    target_size: tuple[int, int] = TARGET_SIZE,
    blur_kernel: int = 3,
    clahe_clip: float = 2.0,
    clahe_grid: tuple[int, int] = (8, 8),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a blood smear image for YOLOv8 inference.

    Args:
        image_input: PIL Image or NumPy RGB array.
        target_size:  (width, height) to resize to.
        blur_kernel:  Gaussian kernel size (odd number).
        clahe_clip:   CLAHE clip limit.
        clahe_grid:   CLAHE tile grid size.

    Returns:
        (preprocessed_rgb, display_rgb):
            preprocessed_rgb — NumPy uint8 RGB array ready for YOLOv8.
            display_rgb      — Original image resized (no CLAHE) for UI display.
    """
    # ── 1. Normalize to NumPy RGB ─────────────────────────────────────────────
    if isinstance(image_input, Image.Image):
        img_rgb = pil_to_numpy(image_input)
    else:
        img_rgb = image_input.copy()
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        elif img_rgb.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)

    log.info("Input image shape: %s", img_rgb.shape)

    # ── 2. Resize (display copy before enhancement) ───────────────────────────
    w, h = target_size
    display_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    # ── 3. Gaussian blur — noise removal ─────────────────────────────────────
    blurred = cv2.GaussianBlur(img_rgb, (blur_kernel, blur_kernel), sigmaX=0)

    # ── 4. CLAHE on L channel (LAB colour space) ──────────────────────────────
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    l_eq = clahe.apply(l_ch)

    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    enhanced_rgb = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # ── 5. Resize to target size ───────────────────────────────────────────────
    preprocessed_rgb = cv2.resize(enhanced_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    log.info("Preprocessed image shape: %s", preprocessed_rgb.shape)
    return preprocessed_rgb, display_rgb


def preprocess_to_pil(
    image_input: Image.Image | np.ndarray,
    **kwargs,
) -> tuple[Image.Image, Image.Image]:
    """Convenience wrapper returning PIL Images instead of NumPy arrays."""
    pre, disp = preprocess(image_input, **kwargs)
    return numpy_to_pil(pre), numpy_to_pil(disp)


# ── Quick smoke test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path

    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_images/sample.jpg"
    if not Path(img_path).exists():
        print(f"Image not found: {img_path}")
        sys.exit(1)

    pil_img = Image.open(img_path).convert("RGB")
    pre_pil, disp_pil = preprocess_to_pil(pil_img)

    out_pre = "data/sample_images/preprocessed.png"
    out_disp = "data/sample_images/display.png"
    pre_pil.save(out_pre)
    disp_pil.save(out_disp)
    print(f"Saved preprocessed → {out_pre}")
    print(f"Saved display copy → {out_disp}")
