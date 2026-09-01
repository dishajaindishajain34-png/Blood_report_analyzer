import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from src.preprocess import preprocess_to_pil
from src.detect import detect_cells, extract_wbc_crops
from src.classify_wbc import classify_wbc_crops
from src.utils import generate_analysis_id, calibrate_counts, project_path, get_logger
from src.normal_ranges import run_full_analysis
from src.report_generator import generate_report

log = get_logger("smoke_test")

IMG = project_path("data", "sample_images", "sample.jpg")
if not IMG.exists():
    print("Sample image not found:", IMG)
    raise SystemExit(1)

try:
    original = Image.open(IMG).convert("RGB")
    pre, display = preprocess_to_pil(original)
    detection = detect_cells(pre)
    wbc_crops = extract_wbc_crops(pre, detection.detections)
    wbc_result = classify_wbc_crops(wbc_crops, total_wbc=detection.raw_counts.get("wbc", 0))
    calibrated = calibrate_counts(detection.raw_counts)
    flags = run_full_analysis(calibrated, wbc_result.differential)
    aid = generate_analysis_id()
    pdf_path = generate_report(
        analysis_id=aid,
        original_image=original,
        annotated_image=detection.annotated_pil,
        raw_counts=detection.raw_counts,
        calibrated_counts=calibrated,
        wbc_differential=wbc_result.differential,
        flags=flags,
        avg_confidence=detection.avg_confidence,
        is_estimated_diff=wbc_result.is_estimated,
    )
    print("Smoke test succeeded. PDF:", pdf_path)
except Exception as e:
    print("Smoke test failed:", e)
    raise
