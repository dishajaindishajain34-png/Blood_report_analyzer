"""
main.py — FastAPI application entry point for the Blood Report Analyzer.

Run with:
    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""

import sys
import os

# Ensure project root is on path when running from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import HealthResponse
from backend.routes.analyze import router as analyze_router
from backend.routes.report  import router as report_router
from src.utils import get_logger, ensure_dir, project_path

log = get_logger(__name__)

# ── Ensure runtime directories exist ─────────────────────────────────────────────
ensure_dir(project_path("reports"))
ensure_dir(project_path("models", "yolov8_blood"))
ensure_dir(project_path("models", "resnet50_wbc"))
ensure_dir(project_path("data", "sample_images"))

# ── App factory ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "AI Blood Report Analyzer",
    description = (
        "Automated blood smear analysis using YOLOv8 cell detection, "
        "ResNet50 WBC classification, and WHO reference range flagging. "
        "Generates downloadable PDF reports."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

@app.on_event("startup")
async def preload_models():
    """Synchronously load all heavy ML models into memory during startup.

    By blocking here we guarantee every model is warm before the first
    HTTP request arrives, eliminating per-request cold-start delays.
    """
    import time
    t0 = time.time()

    try:
        import torch
        torch.set_num_threads(1)
        log.info("Startup: Set PyTorch num_threads to 1 for thread safety.")
    except ImportError:
        pass

    try:
        log.info("Startup: preloading YOLOv8 detection model...")
        from src.detect import _load_model
        _load_model()
        log.info("Startup: YOLOv8 loaded (%.1fs)", time.time() - t0)
    except Exception as e:
        log.error("Startup: YOLOv8 preload failed — %s", e)

    try:
        log.info("Startup: preloading WBC classifier...")
        from src.classify_wbc import _load_model as load_wbc
        load_wbc()
        log.info("Startup: WBC classifier loaded (%.1fs)", time.time() - t0)
    except Exception as e:
        log.warning("Startup: WBC classifier preload failed (non-fatal) — %s", e)

    try:
        log.info("Startup: preloading ML Anemia pipeline...")
        from src.anemia_model import load_anemia_pipeline
        load_anemia_pipeline()
        log.info("Startup: Anemia pipeline loaded (%.1fs)", time.time() - t0)
    except Exception as e:
        log.warning("Startup: Anemia pipeline preload failed (non-fatal) — %s", e)

    try:
        log.info("Startup: preloading Image Anemia model (MobileNetV3-Small)...")
        from src.anemia_image_model import _load_image_model
        _load_image_model()
        log.info("Startup: Image Anemia model loaded (%.1fs)", time.time() - t0)
    except Exception as e:
        log.warning("Startup: Image Anemia model preload failed (non-fatal) — %s", e)

    try:
        log.info("Startup: preloading Fusion model...")
        from src.anemia_fusion import _load_fusion_model
        _load_fusion_model()
        log.info("Startup: Fusion model loaded (%.1fs)", time.time() - t0)
    except Exception as e:
        log.warning("Startup: Fusion model preload failed (non-fatal) — %s", e)

    log.info("Startup: all models ready in %.1fs — accepting requests.", time.time() - t0)


# ── CORS (allow local frontends that talk to the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = False,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────────
app.include_router(analyze_router, tags=["Analysis"])
app.include_router(report_router,  tags=["Reports"])


# ── Health check ──────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Quick health check endpoint."""
    return HealthResponse()


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "name":    "AI Blood Report Analyzer API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    })


# ── Dev runner ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host    = os.getenv("API_HOST", "127.0.0.1"),
        port    = int(os.getenv("API_PORT", 8000)),
        reload  = True,
        workers = 1,
    )
