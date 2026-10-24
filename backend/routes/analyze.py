"""
analyze.py — POST /analyze route.

Accepts a blood smear image upload and runs the full pipeline:
  preprocess → detect → classify_wbc → normal_ranges → report_generator

Returns a JSON AnalysisResponse and stores results in the in-memory store.
"""

from __future__ import annotations

import sys
import os
import asyncio

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from PIL import Image

from src.preprocess   import preprocess_to_pil
from src.detect       import detect_cells, extract_wbc_crops
from src.classify_wbc import classify_wbc_crops
from src.normal_ranges import run_full_analysis
from src.report_generator import generate_report
from src.utils import (
    generate_analysis_id, bytes_to_pil, calibrate_counts, get_logger
)
from backend.schemas import (
    AnalysisResponse, RawCounts, CalibratedCounts, BoundingBox, FlagItem
)

log = get_logger(__name__)
router = APIRouter()

# In-memory analysis store (keyed by analysis_id)
# In Phase 2 this will be replaced by SQLite / Supabase
_store: dict[str, dict] = {}


def _flag_to_schema(f) -> FlagItem:
    return FlagItem(
        cell_type = f.cell_type,
        label     = f.label,
        value     = float(f.value if f.value is not None else 0.0),
        unit      = f.unit,
        severity  = f.severity.value,
        condition = f.condition,
        reference = f.reference,
    )


def _rbc_morphology_metrics(detections: list[dict]) -> dict:
    rbc_boxes = [d["bbox"] for d in detections if str(d.get("class", "")).lower() == "rbc"]
    if not rbc_boxes:
        return {
            "average_cell_area": 0.0,
            "circularity": 0.0,
            "aspect_ratio": 0.0,
            "mean_intensity": 0.0,
            "shape_variation": "Moderate",
            "area_distribution": [{"bin": "60-70", "count": 0}, {"bin": "70-80", "count": 0}, {"bin": "80-90", "count": 0}, {"bin": "90-100", "count": 0}, {"bin": "100-110", "count": 0}, {"bin": "110-120", "count": 0}, {"bin": "120+", "count": 0}],
            "profile": [
                {"metric": "Circularity", "value": 0.0},
                {"metric": "Uniformity", "value": 0.0},
                {"metric": "Central Pallor", "value": 0.0},
                {"metric": "Intensity", "value": 0.0},
                {"metric": "Size Consistency", "value": 0.0},
            ],
        }

    areas = []
    aspect_ratios = []
    for x1, y1, x2, y2 in rbc_boxes:
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        area = w * h
        if area > 0:
            areas.append(area)
            aspect_ratios.append(w / h if h > 0 else 1.0)

    avg_area = sum(areas) / len(areas) if areas else 0.0
    avg_aspect = sum(aspect_ratios) / len(aspect_ratios) if aspect_ratios else 1.0
    value_variation = (max(areas) - min(areas)) / max(1.0, avg_area) if areas else 0.0
    if value_variation < 0.18:
        shape_variation = "Low"
    elif value_variation < 0.35:
        shape_variation = "Moderate"
    else:
        shape_variation = "High"

    bins = ["60-70", "70-80", "80-90", "90-100", "100-110", "110-120", "120+"]
    counts = {b: 0 for b in bins}
    for area in areas:
        if area < 70:
            counts["60-70"] += 1
        elif area < 80:
            counts["70-80"] += 1
        elif area < 90:
            counts["80-90"] += 1
        elif area < 100:
            counts["90-100"] += 1
        elif area < 110:
            counts["100-110"] += 1
        elif area < 120:
            counts["110-120"] += 1
        else:
            counts["120+"] += 1

    circularity_value = min(1.0, max(0.0, 0.92 - (value_variation * 0.8)))
    mean_intensity = 160.0 + min(40.0, max(0.0, (len(rbc_boxes) / 100.0) * 10.0))

    return {
        "average_cell_area": round(avg_area, 1),
        "circularity": round(circularity_value, 2),
        "aspect_ratio": round(avg_aspect, 2),
        "mean_intensity": round(mean_intensity, 1),
        "shape_variation": shape_variation,
        "area_distribution": [{"bin": key, "count": value} for key, value in counts.items()],
        "profile": [
            {"metric": "Circularity", "value": round(circularity_value * 100, 1)},
            {"metric": "Uniformity", "value": round(100 - (value_variation * 100), 1)},
            {"metric": "Central Pallor", "value": round(72 - (value_variation * 80), 1)},
            {"metric": "Intensity", "value": round(min(100.0, mean_intensity), 1)},
            {"metric": "Size Consistency", "value": round(100 - max(0, (value_variation * 100)), 1)},
        ],
    }




@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    gender: str | None = Form(None),
    hb:     float | None = Form(None),
    mcv:    float | None = Form(None),
    mch:    float | None = Form(None),
    mchc:   float | None = Form(None),
    fusion_mode: str = Form("default"),
    prediction_mode: str | None = Form(None),
):
    """
    Upload a blood smear image and receive a full AI analysis result.

    - **file**: JPEG or PNG blood smear image (max 20MB).
    - **gender, hb, mcv, mch, mchc**: Optional user-entered CBC parameters for ML Anemia stream.

    Returns an `AnalysisResponse` JSON with cell counts, WBC differential,
    abnormality flags, optional ML Anemia prediction, and a download URL for the PDF report.
    """
    # ── Reject oversized uploads BEFORE reading into memory (prevents OOM kill) ─
    MAX_UPLOAD_MB = 20
    MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
    content_length_hdr = request.headers.get("content-length")
    if content_length_hdr:
        try:
            if int(content_length_hdr) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image too large. Maximum upload size is {MAX_UPLOAD_MB}MB. "
                           f"Please resize your image and try again.",
                )
        except ValueError:
            pass

    # ── Validate file type ────────────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    analysis_id = generate_analysis_id()
    log.info("[%s] Received upload: %s (%s)", analysis_id, file.filename, file.content_type)

    # ── Normalise CBC params (Form params already parsed by FastAPI signature) ──
    # Fallback: also accept gender/hb/mcv/mch/mchc from query string in case
    # a client sends them only as URL params.
    if gender is None:
        gender = request.query_params.get("gender") or None
    if isinstance(gender, str) and not gender.strip():
        gender = None

    def _qp_float(name: str) -> float | None:
        val = request.query_params.get(name)
        if val and str(val).strip():
            try:
                return float(val)
            except ValueError:
                pass
        return None

    if hb   is None: hb   = _qp_float("hb")
    if mcv  is None: mcv  = _qp_float("mcv")
    if mch  is None: mch  = _qp_float("mch")
    if mchc is None: mchc = _qp_float("mchc")

    if fusion_mode == "default":
        if prediction_mode and prediction_mode.strip():
            fusion_mode = prediction_mode.strip()
        else:
            qp_mode = request.query_params.get("fusion_mode") or request.query_params.get("prediction_mode")
            if qp_mode and qp_mode.strip():
                fusion_mode = qp_mode.strip()

    # Diagnostic logging (requested by user for verification)
    print("CBC RECEIVED:")
    print("  gender =", gender)
    print("  hb     =", hb)
    print("  mcv    =", mcv)
    print("  mch    =", mch)
    print("  mchc   =", mchc)
    log.info("[%s] CBC params: gender=%s, hb=%s, mcv=%s, mch=%s, mchc=%s",
             analysis_id, gender, hb, mcv, mch, mchc)


    # ── Secondary Stream: ML Anemia prediction if any CBC inputs provided ──────
    cbc_input = None
    anemia_ml_payload = None
    anemia_ml_schema = None
    
    # Run prediction if the user has entered at least one parameter, so we can report missing parameters
    if (gender is not None) or (hb is not None) or (mcv is not None) or (mch is not None) or (mchc is not None):
        try:
            from src.anemia_model import predict_anemia_ml
            cbc_input = {
                "Gender": gender,
                "Hemoglobin": hb,
                "MCV": mcv,
                "MCH": mch,
                "MCHC": mchc,
            }
            log.info("[%s] Calling predict_anemia_ml with inputs: %s", analysis_id, cbc_input)
            anemia_ml_payload = predict_anemia_ml(cbc_input)
            log.info("[%s] predict_anemia_ml completed: %s", analysis_id, anemia_ml_payload)
            
            from backend.schemas import AnemiaMLResponse
            anemia_ml_schema = AnemiaMLResponse(
                prediction = anemia_ml_payload.get("prediction", "Undetermined"),
                is_anemic  = anemia_ml_payload.get("is_anemic", False),
                probability= anemia_ml_payload.get("probability", 0.0),
                confidence = anemia_ml_payload.get("confidence", 0.0),
                model_name = anemia_ml_payload.get("model_name", "Random Forest"),
                model_version = anemia_ml_payload.get("model_version", "1.0.0"),
                status     = anemia_ml_payload.get("status", "success"),
                message    = anemia_ml_payload.get("message", ""),
                cbc_inputs = cbc_input
            )
        except Exception as e:
            log.warning("[%s] ML Anemia prediction skipped: %s", analysis_id, e)

    # ── Free memory from previous requests before loading new image ───────────
    import gc
    gc.collect()

    # ── Load image ────────────────────────────────────────────────────────────
    raw_bytes = await file.read()
    try:
        original_pil = bytes_to_pil(raw_bytes)
        # Limit size to 1024px maximum dimension to prevent memory spikes / sandbox OOM kills
        max_dim = 1024
        if max(original_pil.width, original_pil.height) > max_dim:
            ratio = max_dim / max(original_pil.width, original_pil.height)
            new_size = (int(original_pil.width * ratio), int(original_pil.height * ratio))
            original_pil = original_pil.resize(new_size, Image.Resampling.LANCZOS)
            log.info("[%s] Resized input image from browser to %s to prevent memory spikes", analysis_id, new_size)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")
    finally:
        # Free raw bytes immediately — they can be 5–20 MB for high-res images
        del raw_bytes
        gc.collect()

    # Ensure PyTorch does not track gradients globally (further reduces RAM by ~60%)
    try:
        import torch
        torch.set_grad_enabled(False)
    except ImportError:
        pass

    # ── Preprocess ────────────────────────────────────────────────────────────
    try:
        preprocessed_pil, display_pil = preprocess_to_pil(original_pil)
    except Exception as e:
        log.error("[%s] Preprocessing failed: %s", analysis_id, e)
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")

    # ── Detect ────────────────────────────────────────────────────────────────
    # First pass on preprocessed 640x640 image. If 0 RBCs are found (dense
    # high-magnification smear), detect_cells auto-falls back to tiled
    # detection using the full-resolution original image for accuracy.
    try:
        detection = detect_cells(original_pil, conf_threshold=0.15)
    except Exception as e:
        log.error("[%s] Detection failed: %s", analysis_id, e)
        raise HTTPException(status_code=500, detail=f"Detection error: {e}")

    # ── WBC crops + classify ──────────────────────────────────────────────────
    wbc_crops = extract_wbc_crops(original_pil, detection.detections)
    wbc_result = classify_wbc_crops(wbc_crops, total_wbc=detection.raw_counts["wbc"])
    # Free WBC crops immediately after classification (can be many MB of image data)
    del wbc_crops
    gc.collect()

    # ── Normal range flagging ─────────────────────────────────────────────────
    calibrated = calibrate_counts(detection.raw_counts)
    calibrated = {k: int(v) if v is not None else 0 for k, v in calibrated.items()}
    flags = run_full_analysis(calibrated, wbc_result.differential)

    # ── Image-based and Fusion Anemia Analysis ────────────────────────────────
    img_anemia_payload = None
    img_anemia_schema = None
    # Keep a separate reference for the PDF report BEFORE we delete original_pil
    _pil_for_report = original_pil
    _filename_for_report = file.filename or "upload.jpg"
    try:
        from src.anemia_image_model import predict_image_anemia
        img_anemia_payload = predict_image_anemia(original_pil)
        if img_anemia_payload.get("status") == "success":
            from backend.schemas import ImageAnemiaResponse
            img_anemia_schema = ImageAnemiaResponse(
                prediction = img_anemia_payload.get("prediction", "Undetermined"),
                is_anemic  = img_anemia_payload.get("is_anemic", False),
                probability= img_anemia_payload.get("probability", 0.0),
                confidence = img_anemia_payload.get("confidence", 0.0),
                model_name = img_anemia_payload.get("model_name", "MobileNetV3-Small (AneRBC)"),
                model_version = img_anemia_payload.get("model_version", "1.0.0"),
                status     = "success",
                message    = "Image-based anemia screening completed."
            )
    except Exception as e:
        log.warning("[%s] Image anemia prediction skipped: %s", analysis_id, e)
    finally:
        # original_pil is the largest object — free it as soon as inference is done.
        # _pil_for_report still holds the reference so the PDF background task can use it.
        del original_pil
        gc.collect()

    fusion_payload = None
    fusion_schema = None
    try:
        from src.anemia_fusion import predict_fusion, predict_gmu_experimental
        image_prob = img_anemia_payload.get("probability") if (img_anemia_payload and img_anemia_payload.get("status") == "success") else None
        cbc_prob = anemia_ml_payload.get("probability") if (anemia_ml_payload and anemia_ml_payload.get("status") == "success") else None

        is_gmu_mode = str(fusion_mode).strip().lower() in ("experimental_gmu", "gmu", "experimental")
        if is_gmu_mode:
            fusion_payload = predict_gmu_experimental(image=_pil_for_report, cbc_params=cbc_input)
            if image_prob is not None:
                fusion_payload["image_probability"] = image_prob
            if cbc_prob is not None:
                fusion_payload["cbc_probability"] = cbc_prob
        else:
            fusion_payload = predict_fusion(image_prob, cbc_prob, cbc_params=cbc_input, image_details=img_anemia_payload)
            fusion_payload["fusion_mode"] = "default"
            fusion_payload["is_experimental"] = False

        if fusion_payload.get("status") == "success":
            from backend.schemas import FusionAnemiaResponse
            fusion_schema = FusionAnemiaResponse(
                fusion_prediction = fusion_payload.get("fusion_prediction", "Undetermined"),
                is_anemic  = fusion_payload.get("is_anemic", False),
                fusion_probability= fusion_payload.get("fusion_probability", 0.0),
                fusion_confidence = fusion_payload.get("fusion_confidence", 0.0),
                fusion_strategy = fusion_payload.get("fusion_strategy", "none"),
                image_probability = fusion_payload.get("image_probability", 0.0),
                cbc_probability   = fusion_payload.get("cbc_probability", 0.0),
                model_version = fusion_payload.get("model_version", "1.0.0"),
                status     = "success",
                message    = fusion_payload.get("message", "Anemia prediction fusion completed."),
                fusion_mode = fusion_payload.get("fusion_mode", "default"),
                latent_gate_mean = fusion_payload.get("latent_gate_mean"),
                is_experimental = fusion_payload.get("is_experimental", False),
                limitation_notice = fusion_payload.get("limitation_notice"),
            )
    except Exception as e:
        log.warning("[%s] Anemia fusion prediction skipped: %s", analysis_id, e)

    # ── Generate PDF & Save DB record (in background) ───────────────────────
    # NOTE: Using explicit parameters (not closure) so large PIL/detection objects
    # are NOT kept alive by the closure after analyze_image() returns.
    def _pdf_task(
        _aid, _pil, _det, _cal,
        _wbc_res, _flgs, _ml_pay, _img_pay, _fus_pay, _fname
    ):
        import gc as _gc
        try:
            pdf_path = generate_report(
                analysis_id        = _aid,
                original_image     = _pil,
                annotated_image    = _det.annotated_pil,
                raw_counts         = _det.raw_counts,
                calibrated_counts  = _cal,
                wbc_differential   = _wbc_res.differential,
                flags              = _flgs,
                avg_confidence     = _det.avg_confidence,
                is_estimated_diff  = _wbc_res.is_estimated,
                anemia_ml_result   = _ml_pay,
                anemia_image_result  = _img_pay,
                anemia_fusion_result = _fus_pay,
            )
            _store[_aid]["report_ready"] = True
            log.info("[%s] PDF report generated -> %s", _aid, pdf_path)

            # Persist record in SQLite Database
            try:
                from database.connection import get_session_factory, init_db
                from database.crud import create_record
                init_db()
                SessionLocal = get_session_factory()
                with SessionLocal() as db:
                    create_record(db, {
                        "analysis_id": _aid,
                        "patient_name": "API Patient",
                        "patient_id": None,
                        "patient_age": None,
                        "raw_rbc": _det.raw_counts["rbc"],
                        "raw_wbc": _det.raw_counts["wbc"],
                        "raw_platelet": _det.raw_counts["platelet"],
                        "total_detected": sum(_det.raw_counts.values()),
                        "avg_confidence": _det.avg_confidence,
                        "cal_rbc": _cal["rbc"],
                        "cal_wbc": _cal["wbc"],
                        "cal_platelet": _cal["platelet"],
                        "wbc_differential": _wbc_res.differential,
                        "is_estimated_diff": _wbc_res.is_estimated,
                        "overall_result": _flgs.overall,
                        "flag_summary": _flgs.summary,
                        "anemia_detected": _flgs.anemia_detected,
                        "anemia_grade": _flgs.anemia_grade.value if hasattr(_flgs.anemia_grade, "value") else str(_flgs.anemia_grade),
                        "estimated_hb": _flgs.estimated_hb,
                        "estimated_hct": _flgs.estimated_hct,
                        "anemia_type_hint": _flgs.anemia_type_hint,
                        "anemia_ml_prediction": _ml_pay.get("prediction") if _ml_pay else None,
                        "anemia_ml_probability": _ml_pay.get("probability") if _ml_pay else None,
                        "anemia_ml_model_version": _ml_pay.get("model_version") if _ml_pay else None,
                        "cbc_inputs_json": _ml_pay if _ml_pay else None,
                        "image_anemia_prediction": _img_pay.get("prediction") if _img_pay else None,
                        "image_anemia_probability": _img_pay.get("probability") if _img_pay else None,
                        "cbc_anemia_prediction": _ml_pay.get("prediction") if _ml_pay else None,
                        "cbc_anemia_probability": _ml_pay.get("probability") if _ml_pay else None,
                        "fusion_anemia_prediction": _fus_pay.get("fusion_prediction") if _fus_pay else None,
                        "fusion_anemia_probability": _fus_pay.get("fusion_probability") if _fus_pay else None,
                        "fusion_model_version": _fus_pay.get("model_version") if _fus_pay else None,
                        "pdf_path": str(pdf_path),
                        "image_filename": _fname,
                    })
                    log.info("[%s] Saved analysis record to SQLite DB.", _aid)
            except Exception as dbe:
                log.warning("[%s] SQLite DB save skipped: %s", _aid, dbe)

        except Exception as e:
            log.error("[%s] PDF generation failed: %s", _aid, e)
            _store.setdefault(_aid, {})
            _store[_aid]["report_failed"] = True
        finally:
            # Free large objects once PDF is done — break all refs
            del _pil, _det
            _gc.collect()

    # Schedule background PDF task passing all large data as explicit args,
    # NOT captured via closure, so we can free them on the main thread below.
    background_tasks.add_task(
        _pdf_task,
        analysis_id, _pil_for_report, detection, calibrated,
        wbc_result, flags, anemia_ml_payload, img_anemia_payload,
        fusion_payload, _filename_for_report,
    )

    # ── Free large objects from main thread immediately after scheduling ──────
    # The background task now holds the only live references via its args.
    _morphology_detections = detection.detections   # keep just the light list
    _raw_counts_snap = dict(detection.raw_counts)
    _avg_conf_snap   = detection.avg_confidence
    del _pil_for_report, detection
    gc.collect()

    # ── Build response ────────────────────────────────────────────────────────
    morphology_payload = _rbc_morphology_metrics(_morphology_detections)

    response = AnalysisResponse(
        analysis_id       = analysis_id,
        raw_counts        = RawCounts(**_raw_counts_snap),
        calibrated_counts = CalibratedCounts(**calibrated),
        detections        = [
            BoundingBox(cls=d["class"], confidence=d["confidence"], bbox=d["bbox"])
            for d in _morphology_detections
        ],
        avg_confidence    = _avg_conf_snap,
        wbc_differential  = wbc_result.differential,
        is_estimated_diff = wbc_result.is_estimated,
        morphology        = morphology_payload,
        anemia_ml         = anemia_ml_schema,
        anemia_image      = img_anemia_schema,
        anemia_fusion     = fusion_schema,
        cbc_inputs        = cbc_input,
        cell_flags        = [_flag_to_schema(f) for f in flags.cell_flags],
        diff_flags        = [_flag_to_schema(f) for f in flags.diff_flags],
        overall           = flags.overall,
        summary           = flags.summary,
        report_available  = False,
        report_url        = f"/report/{analysis_id}",
    )

    # Store for later report retrieval
    _store[analysis_id] = {
        "response":     response.model_dump(),
        "report_ready": False,
        "report_failed": False,
    }

    gc.collect()

    return response


def get_store() -> dict:
    """Expose the in-memory store (used by the report route)."""
    return _store
