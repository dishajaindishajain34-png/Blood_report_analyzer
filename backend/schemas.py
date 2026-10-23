"""
schemas.py — Pydantic request / response models for the FastAPI backend.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


# ── Request ───────────────────────────────────────────────────────────────────────
# File upload is handled via UploadFile (multipart/form-data) directly in the route.
# No extra schema needed for the request.


# ── Per-detection box ─────────────────────────────────────────────────────────────
class BoundingBox(BaseModel):
    cls:        str
    confidence: float
    bbox:       list[float]   # [x1, y1, x2, y2]


# ── Cell counts ───────────────────────────────────────────────────────────────────
class RawCounts(BaseModel):
    rbc:      int = 0
    wbc:      int = 0
    platelet: int = 0


class CalibratedCounts(BaseModel):
    rbc:      int | None = 0
    wbc:      int | None = 0
    platelet: int | None = 0


# ── Flag item ─────────────────────────────────────────────────────────────────────
class FlagItem(BaseModel):
    cell_type: str
    label:     str
    value:     float
    unit:      str
    severity:  str
    condition: str
    reference: str


class MorphologyMetrics(BaseModel):
    average_cell_area: float = 0.0
    circularity: float = 0.0
    aspect_ratio: float = 0.0
    mean_intensity: float = 0.0
    shape_variation: str = "Moderate"
    area_distribution: list[dict[str, float | str]] = []
    profile: list[dict[str, float | str]] = []


class RiskFactor(BaseModel):
    factor: str
    weight: float


class AnemiaRisk(BaseModel):
    score: float = 0.0
    level: str = "low"
    explanation: str = ""
    contributing_factors: list[RiskFactor] = []


class AnemiaMLResponse(BaseModel):
    prediction: str = "Undetermined"
    is_anemic: bool = False
    probability: float = 0.0
    confidence: float = 0.0
    model_name: str = "Random Forest"
    model_version: str = "1.0.0"
    status: str = "success"
    message: str = ""
    cbc_inputs: Optional[dict] = None


class ImageAnemiaResponse(BaseModel):
    prediction: str = "Undetermined"
    is_anemic: bool = False
    probability: float = 0.0
    confidence: float = 0.0
    model_name: str = "MobileNetV3-Small (AneRBC)"
    model_version: str = "1.0.0"
    status: str = "success"
    message: str = ""


class FusionAnemiaResponse(BaseModel):
    fusion_prediction: str = "Undetermined"
    is_anemic: bool = False
    fusion_probability: float = 0.0
    fusion_confidence: float = 0.0
    fusion_strategy: str = "none"
    image_probability: float = 0.0
    cbc_probability: float = 0.0
    model_version: str = "1.0.0"
    status: str = "success"
    message: str = ""
    fusion_mode: Optional[str] = "default"
    latent_gate_mean: Optional[float] = None
    is_experimental: bool = False
    limitation_notice: Optional[str] = None


# ── Main analysis response ────────────────────────────────────────────────────────
class AnalysisResponse(BaseModel):
    analysis_id:       str
    status:            str = "success"

    # Detection
    raw_counts:        RawCounts
    calibrated_counts: CalibratedCounts
    detections:        list[BoundingBox]
    avg_confidence:    float

    # WBC differential
    wbc_differential:  dict[str, float]
    is_estimated_diff: bool

    # Morphology + risk (anemia_risk removed from heuristic; kept optional for PDF compat)
    morphology:        MorphologyMetrics = MorphologyMetrics()
    anemia_risk:       Optional[AnemiaRisk] = None

    # ML Anemia Secondary Stream
    anemia_ml:         Optional[AnemiaMLResponse] = None

    # Image-based and Fusion Anemia results
    anemia_image:      Optional[ImageAnemiaResponse] = None
    anemia_fusion:     Optional[FusionAnemiaResponse] = None

    # Top-level CBC inputs
    cbc_inputs:        Optional[dict] = None

    # Flagging
    cell_flags:  list[FlagItem]
    diff_flags:  list[FlagItem]
    overall:     str
    summary:     list[str]

    # Report
    report_available: bool = False
    report_url:       Optional[str] = None


# ── Health check ──────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str = "1.0.0"
    message: str = "AI Blood Report Analyzer API is running."
