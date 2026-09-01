"""
utils.py — Shared utility helpers for the Blood Report Analyzer.
"""

import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image

# ── Logging setup ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ── Path helpers ─────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_path(*parts: str) -> Path:
    """Build an absolute path relative to the project root."""
    return ROOT_DIR.joinpath(*parts)


# ── ID / timestamp helpers ───────────────────────────────────────────────────────
def generate_analysis_id() -> str:
    """Generate a unique analysis/report ID."""
    return f"BRA-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"


def current_timestamp() -> str:
    """Human-readable timestamp for report headers."""
    return datetime.now().strftime("%d %B %Y, %I:%M %p")


# ── Image helpers ────────────────────────────────────────────────────────────────
def load_image_pil(path: str | Path) -> Image.Image:
    """Load an image file as a PIL Image (RGB)."""
    return Image.open(str(path)).convert("RGB")


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to NumPy array (H, W, C) uint8 RGB."""
    return np.array(img, dtype=np.uint8)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert NumPy array (H, W, C) uint8 RGB to PIL Image."""
    return Image.fromarray(arr.astype(np.uint8))


def bytes_to_pil(data: bytes) -> Image.Image:
    """Convert raw bytes (from upload) to PIL Image."""
    import io
    return Image.open(io.BytesIO(data)).convert("RGB")


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Serialize PIL Image to bytes."""
    import io
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ── Calibration ──────────────────────────────────────────────────────────────────
# Default calibration factor: cells detected in a 640×640 field of view
# to estimated cells per µL.  This is a demo-level approximation.
# Real calibration requires known slide magnification and field diameter.
DEFAULT_CALIBRATION = {
    # Typical counts per 640x640 frame at BCCD microscopy magnification (~100x oil).
    # At this magnification each field covers ~0.002 µL of blood.
    # Midpoints are WHO normal reference range centres.
    "rbc":      {"midpoint": 4_800_000, "typical_frame": 5},   # BCCD images show ~5 RBCs at 100x
    "wbc":      {"midpoint": 7_000,     "typical_frame": 3},   # ~3 WBCs visible at this magnification
    "platelet": {"midpoint": 250_000,   "typical_frame": 10},  # ~10 platelets per frame at 100x
}

def calibrate_counts(raw_counts: dict[str, int]) -> dict[str, int]:
    """
    Convert raw detected counts (cells in one microscopy frame) to estimated
    physiological values (cells per µL) using linear scaling against typical
    frame occupancy.

    If a cell type is not detected at all (count = 0), we return the normal
    midpoint rather than zero, because absence from one frame does not imply
    clinical absence — the field of view is simply too small to guarantee
    coverage of all cell types.

    In Phase 2 this will use real slide metadata (magnification, field diameter,
    haematocrit layer thickness) for calibrated estimates.
    """
    calibrated = {}
    logger = logging.getLogger(__name__)
    for cell_type, raw in raw_counts.items():
        ref = DEFAULT_CALIBRATION.get(cell_type)
        if ref is None:
            calibrated[cell_type] = raw
            continue

        midpoint     = ref["midpoint"]
        typical_frame = ref["typical_frame"]

        if raw == 0:
            # Cannot conclude deficiency from a single frame — mark as undetermined
            # so downstream analysis can request a re-scan or mark low confidence.
            logger.warning(
                "Zero detections for %s in sample frame — marking as undetermined",
                cell_type,
            )
            calibrated[cell_type] = None
        else:
            # Linear scale: detected / typical_frame * midpoint
            scale = raw / typical_frame
            calibrated[cell_type] = int(midpoint * scale)

    return calibrated

