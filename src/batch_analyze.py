"""
batch_analyze.py — Multi-image batch analysis pipeline.

Accepts a list of blood smear images (PIL Images or file paths),
runs the full pipeline on each, and returns an averaged result
that reflects more representative cell counts across multiple fields.

This mirrors real lab practice where multiple fields of view are
counted and averaged for a more accurate CBC estimate.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image

from src.preprocess     import preprocess_to_pil
from src.detect         import detect_cells, extract_wbc_crops, DetectionResult
from src.classify_wbc   import classify_wbc_crops, WBCClassificationResult
from src.normal_ranges  import run_full_analysis, AnalysisFlags
from src.report_generator import generate_report
from src.utils          import (
    get_logger, calibrate_counts, generate_analysis_id, current_timestamp,
    bytes_to_pil,
)

log = get_logger(__name__)

WBC_CLASSES = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]


@dataclass
class BatchResult:
    """Aggregated result across multiple images."""
    analysis_id:       str
    n_images:          int

    # Per-image raw counts (list of dicts)
    per_image_counts:  list[dict] = field(default_factory=list)

    # Averaged raw counts
    avg_raw:           dict       = field(default_factory=dict)
    avg_calibrated:    dict       = field(default_factory=dict)

    # Averaged WBC differential
    avg_differential:  dict       = field(default_factory=dict)
    is_estimated_diff: bool       = True

    # Average confidence
    avg_confidence:    float      = 0.0

    # Flags on averaged counts
    flags:             AnalysisFlags | None = None

    # Representative images (first image pair for report)
    original_pil:      Image.Image | None = None
    annotated_pil:     Image.Image | None = None

    # PDF path
    pdf_path:          Path | None = None


def run_batch(
    images:            list[Image.Image],
    conf_threshold:    float = 0.15,
    iou_threshold:     float = 0.40,
    progress_callback  = None,   # Optional callable(step: int, total: int, msg: str)
    anemia_ml_result:  dict | None = None,
    anemia_image_result: dict | None = None,
    anemia_fusion_result: dict | None = None,
) -> BatchResult:
    """
    Run the full pipeline on multiple images and return an averaged BatchResult.

    Args:
        images:           List of PIL Images (blood smear slides).
        conf_threshold:   YOLOv8 confidence threshold.
        iou_threshold:    NMS IOU threshold.
        progress_callback: Optional fn called with (current, total, message).

    Returns:
        BatchResult with averaged cell counts, differential, and flags.
    """
    n = len(images)
    log.info("Batch analysis: %d images", n)

    aid = generate_analysis_id()
    all_raw: list[dict]  = []
    all_diff: list[dict] = []
    all_conf: list[float] = []
    first_orig  = None
    first_ann   = None
    all_estimated = True

    def _progress(step: int, msg: str):
        if progress_callback:
            progress_callback(step, n, msg)

    for i, img in enumerate(images):
        _progress(i, f"Image {i+1}/{n}: Preprocessing...")
        pre, _ = preprocess_to_pil(img)

        _progress(i, f"Image {i+1}/{n}: Detecting cells (YOLOv8)...")
        det = detect_cells(pre, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

        _progress(i, f"Image {i+1}/{n}: Classifying WBCs...")
        crops  = extract_wbc_crops(pre, det.detections)
        wbc    = classify_wbc_crops(crops, det.raw_counts["wbc"])

        all_raw.append(det.raw_counts)
        all_diff.append(wbc.differential)
        all_conf.append(det.avg_confidence)
        if not wbc.is_estimated:
            all_estimated = False

        if i == 0:
            first_orig = img
            first_ann  = det.annotated_pil

        log.info("  Image %d: %s | conf=%.2f", i+1, det.raw_counts, det.avg_confidence)

    # ── Average counts ────────────────────────────────────────────────────────
    avg_raw = {
        "rbc":      int(sum(r["rbc"]      for r in all_raw) / n),
        "wbc":      int(sum(r["wbc"]      for r in all_raw) / n),
        "platelet": int(sum(r["platelet"] for r in all_raw) / n),
    }

    # ── Average differential (normalise to 100%) ──────────────────────────────
    avg_diff_raw = {
        cls: sum(d.get(cls, 0.0) for d in all_diff) / n
        for cls in WBC_CLASSES
    }
    total_pct = sum(avg_diff_raw.values())
    avg_diff  = {
        cls: round((v / total_pct) * 100, 1) if total_pct > 0 else 0.0
        for cls, v in avg_diff_raw.items()
    }

    avg_conf = round(sum(all_conf) / n, 4)

    # ── Calibrate and flag ────────────────────────────────────────────────────
    avg_cal   = calibrate_counts(avg_raw)
    flags     = run_full_analysis(avg_cal, avg_diff)

    log.info("Batch averaged: raw=%s | cal=%s | overall=%s", avg_raw, avg_cal, flags.overall)

    # ── Generate PDF ──────────────────────────────────────────────────────────
    pdf_path = None
    if first_orig and first_ann:
        pdf_path = generate_report(
            analysis_id        = aid,
            original_image     = first_orig,
            annotated_image    = first_ann,
            raw_counts         = avg_raw,
            calibrated_counts  = avg_cal,
            wbc_differential   = avg_diff,
            flags              = flags,
            avg_confidence     = avg_conf,
            is_estimated_diff  = all_estimated,
            anemia_ml_result   = anemia_ml_result,
            anemia_image_result  = anemia_image_result,
            anemia_fusion_result = anemia_fusion_result,
        )

    return BatchResult(
        analysis_id       = aid,
        n_images          = n,
        per_image_counts  = all_raw,
        avg_raw           = avg_raw,
        avg_calibrated    = avg_cal,
        avg_differential  = avg_diff,
        is_estimated_diff = all_estimated,
        avg_confidence    = avg_conf,
        flags             = flags,
        original_pil      = first_orig,
        annotated_pil     = first_ann,
        pdf_path          = pdf_path,
    )
