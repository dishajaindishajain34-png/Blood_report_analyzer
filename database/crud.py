"""
crud.py — CRUD operations for patient analysis records.
"""

from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session
from database.models import AnalysisRecord
from src.utils import get_logger

log = get_logger(__name__)


def create_record(db: Session, data: dict) -> AnalysisRecord:
    """Insert a new analysis record into the database."""
    record = AnalysisRecord(
        analysis_id      = data["analysis_id"],
        patient_name     = data.get("patient_name"),
        patient_age      = data.get("patient_age"),
        patient_id       = data.get("patient_id"),
        raw_rbc          = data.get("raw_rbc", 0),
        raw_wbc          = data.get("raw_wbc", 0),
        raw_platelet     = data.get("raw_platelet", 0),
        total_detected   = data.get("total_detected", 0),
        avg_confidence   = data.get("avg_confidence", 0.0),
        cal_rbc          = data.get("cal_rbc"),
        cal_wbc          = data.get("cal_wbc"),
        cal_platelet     = data.get("cal_platelet"),
        wbc_differential = data.get("wbc_differential"),
        is_estimated_diff= data.get("is_estimated_diff", True),
        overall_result   = data.get("overall_result", "Normal"),
        flag_summary     = data.get("flag_summary", []),
        anemia_detected         = data.get("anemia_detected", False),
        anemia_grade            = data.get("anemia_grade"),
        estimated_hb            = data.get("estimated_hb"),
        estimated_hct           = data.get("estimated_hct"),
        anemia_type_hint        = data.get("anemia_type_hint"),
        anemia_ml_prediction    = data.get("anemia_ml_prediction"),
        anemia_ml_probability   = data.get("anemia_ml_probability"),
        anemia_ml_model_version = data.get("anemia_ml_model_version"),
        cbc_inputs_json         = data.get("cbc_inputs_json"),
        image_anemia_prediction  = data.get("image_anemia_prediction"),
        image_anemia_probability = data.get("image_anemia_probability"),
        cbc_anemia_prediction    = data.get("cbc_anemia_prediction"),
        cbc_anemia_probability   = data.get("cbc_anemia_probability"),
        fusion_anemia_prediction  = data.get("fusion_anemia_prediction"),
        fusion_anemia_probability = data.get("fusion_anemia_probability"),
        fusion_model_version      = data.get("fusion_model_version"),
        pdf_path                = data.get("pdf_path"),
        image_filename          = data.get("image_filename"),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    log.info("Record saved: %s (overall=%s)", record.analysis_id, record.overall_result)
    return record


def get_record_by_analysis_id(db: Session, analysis_id: str) -> Optional[AnalysisRecord]:
    return db.query(AnalysisRecord).filter(
        AnalysisRecord.analysis_id == analysis_id
    ).first()


def get_records_by_patient(db: Session, patient_id: str) -> list[AnalysisRecord]:
    return (
        db.query(AnalysisRecord)
        .filter(AnalysisRecord.patient_id == patient_id)
        .order_by(AnalysisRecord.created_at.desc())
        .all()
    )


def get_all_records(
    db: Session, skip: int = 0, limit: int = 100
) -> list[AnalysisRecord]:
    return (
        db.query(AnalysisRecord)
        .order_by(AnalysisRecord.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_stats(db: Session) -> dict:
    """Return aggregate statistics across all records."""
    total  = db.query(AnalysisRecord).count()
    normal = db.query(AnalysisRecord).filter(
        AnalysisRecord.overall_result == "Normal"
    ).count()
    abnormal = total - normal

    return {
        "total_analyses": total,
        "normal":         normal,
        "abnormal":       abnormal,
        "abnormal_pct":   round((abnormal / total * 100) if total else 0, 1),
    }


def delete_record(db: Session, analysis_id: str) -> bool:
    record = get_record_by_analysis_id(db, analysis_id)
    if record:
        db.delete(record)
        db.commit()
        return True
    return False
