"""
report.py — GET /report/{analysis_id} route.

Serves the generated PDF report for a given analysis ID.
"""

import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.utils import get_logger, project_path
from backend.routes.analyze import get_store

log = get_logger(__name__)
router = APIRouter()

REPORTS_DIR = project_path("reports")


@router.get("/report/{analysis_id}")
async def download_report(analysis_id: str):
    """
    Download the generated PDF report for a completed analysis.

    - **analysis_id**: The ID returned by `/analyze`.

    Returns the PDF file as a downloadable attachment.
    """
    store = get_store()

    if analysis_id not in store:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis '{analysis_id}' not found. Please run /analyze first.",
        )

    pdf_path = REPORTS_DIR / f"{analysis_id}.pdf"

    if not pdf_path.exists():
        if store[analysis_id].get("report_failed", False):
            raise HTTPException(
                status_code=500,
                detail="Report generation failed on the server. See logs.",
            )
        if not store[analysis_id].get("report_ready", False):
            raise HTTPException(
                status_code=202,
                detail="Report is still being generated. Please retry in a moment.",
            )
        raise HTTPException(
            status_code=500,
            detail="Report file not found on server. PDF generation may have failed.",
        )

    log.info("Serving PDF report: %s", pdf_path)
    return FileResponse(
        path        = str(pdf_path),
        media_type  = "application/pdf",
        filename    = f"BloodReport_{analysis_id}.pdf",
        headers     = {"Content-Disposition": f'attachment; filename="BloodReport_{analysis_id}.pdf"'},
    )
