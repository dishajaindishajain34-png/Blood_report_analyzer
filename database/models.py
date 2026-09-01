"""
models.py — SQLAlchemy ORM models for patient record storage.
"""

from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, Text, Boolean
from database.connection import Base


class AnalysisRecord(Base):
    """Stores every completed blood analysis."""
    __tablename__ = "analysis_records"

    id              = Column(Integer, primary_key=True, index=True)
    analysis_id     = Column(String(40), unique=True, index=True, nullable=False)
    patient_name    = Column(String(120), nullable=True)
    patient_age     = Column(Integer,     nullable=True)
    patient_id      = Column(String(40),  nullable=True, index=True)
    created_at      = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Raw detection
    raw_rbc         = Column(Integer, default=0)
    raw_wbc         = Column(Integer, default=0)
    raw_platelet    = Column(Integer, default=0)
    total_detected  = Column(Integer, default=0)
    avg_confidence  = Column(Float,   default=0.0)

    # Calibrated estimates
    cal_rbc         = Column(Integer, default=0)
    cal_wbc         = Column(Integer, default=0)
    cal_platelet    = Column(Integer, default=0)

    # WBC Differential (stored as JSON)
    wbc_differential = Column(JSON, nullable=True)   # {"Neutrophil": 54.0, ...}
    is_estimated_diff = Column(Boolean, default=True)

    # Flagging
    overall_result  = Column(String(20), default="Normal")  # "Normal" | "Abnormal"
    flag_summary    = Column(JSON, nullable=True)    # list of flag strings
    # Anemia-specific fields
    anemia_detected = Column(Boolean, default=False)
    anemia_grade    = Column(String(60), nullable=True)
    estimated_hb    = Column(Float, nullable=True)
    estimated_hct   = Column(Float, nullable=True)
    anemia_type_hint= Column(String(200), nullable=True)

    # ML Anemia Secondary Stream Fields (Backward Compatible)
    anemia_ml_prediction    = Column(String(40), nullable=True)
    anemia_ml_probability   = Column(Float, nullable=True)
    anemia_ml_model_version = Column(String(40), nullable=True)
    cbc_inputs_json         = Column(JSON, nullable=True)

    # Image-based Anemia & Fusion Fields
    image_anemia_prediction  = Column(String(40), nullable=True)
    image_anemia_probability = Column(Float, nullable=True)
    cbc_anemia_prediction    = Column(String(40), nullable=True)
    cbc_anemia_probability   = Column(Float, nullable=True)
    fusion_anemia_prediction  = Column(String(40), nullable=True)
    fusion_anemia_probability = Column(Float, nullable=True)
    fusion_model_version      = Column(String(40), nullable=True)

    # File paths
    pdf_path        = Column(String(260), nullable=True)
    image_filename  = Column(String(260), nullable=True)

    def to_dict(self) -> dict:
        return {
            "id":                     self.id,
            "analysis_id":            self.analysis_id,
            "patient_name":           self.patient_name,
            "patient_age":            self.patient_age,
            "patient_id":             self.patient_id,
            "created_at":             self.created_at.isoformat() if self.created_at else None,
            "raw_rbc":                self.raw_rbc,
            "raw_wbc":                self.raw_wbc,
            "raw_platelet":           self.raw_platelet,
            "total_detected":         self.total_detected,
            "avg_confidence":         self.avg_confidence,
            "cal_rbc":                self.cal_rbc,
            "cal_wbc":                self.cal_wbc,
            "cal_platelet":           self.cal_platelet,
            "wbc_differential":       self.wbc_differential,
            "is_estimated_diff":      self.is_estimated_diff,
            "overall_result":         self.overall_result,
            "flag_summary":           self.flag_summary,
            "anemia_detected":        self.anemia_detected,
            "anemia_grade":           self.anemia_grade,
            "estimated_hb":           self.estimated_hb,
            "estimated_hct":          self.estimated_hct,
            "anemia_type_hint":       self.anemia_type_hint,
            "anemia_ml_prediction":   self.anemia_ml_prediction,
            "anemia_ml_probability":  self.anemia_ml_probability,
            "anemia_ml_model_version":self.anemia_ml_model_version,
            "cbc_inputs_json":        self.cbc_inputs_json,
            "image_anemia_prediction":  self.image_anemia_prediction,
            "image_anemia_probability": self.image_anemia_probability,
            "cbc_anemia_prediction":    self.cbc_anemia_prediction,
            "cbc_anemia_probability":   self.cbc_anemia_probability,
            "fusion_anemia_prediction":  self.fusion_anemia_prediction,
            "fusion_anemia_probability": self.fusion_anemia_probability,
            "fusion_model_version":      self.fusion_model_version,
            "pdf_path":               self.pdf_path,
            "image_filename":         self.image_filename,
        }
