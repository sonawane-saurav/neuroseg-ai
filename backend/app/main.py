"""
FastAPI Brain Tumor Segmentation Application Entry Point.
Modular architecture with model checkpoint caching, database persistence, and CORS support.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import sys

# Ensure repository root is on Python sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app.core.config import settings
from backend.app.database.db import init_db
from backend.app.schemas.schemas import HealthResponse, ModelInfoResponse
from backend.app.api.endpoints import segment, uncertainty, robustness, metrics, history
from cv.inference.predictor import BrainTumorPredictor

# Global singleton predictor instance
predictor: BrainTumorPredictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("[Server Startup] Initializing SQLite database...")
    init_db()
    print("[Server Startup] Loading Deep Learning model checkpoints...")
    predictor = BrainTumorPredictor(artifacts_dir=settings.ARTIFACTS_DIR)
    yield
    print("[Server Shutdown] Cleaning up resources...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Research-Grade Brain Tumor Segmentation and Uncertainty Analysis API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS setup for React / Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("FRONTEND_ORIGIN", "*").split(",")
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(segment.router, prefix=settings.API_V1_PREFIX, tags=["Segmentation"])
app.include_router(uncertainty.router, prefix=settings.API_V1_PREFIX, tags=["Uncertainty"])
app.include_router(robustness.router, prefix=settings.API_V1_PREFIX, tags=["Robustness"])
app.include_router(metrics.router, prefix=settings.API_V1_PREFIX, tags=["Metrics & Experiments"])
app.include_router(history.router, prefix=settings.API_V1_PREFIX, tags=["History"])

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    return {
        "status": "healthy",
        "project_name": settings.PROJECT_NAME,
        "device": device_name,
        "torch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "unet_available": predictor.is_unet_ready if predictor else False,
        "multitask_available": predictor.is_multitask_ready if predictor else False
    }

@app.get("/api/model/info", response_model=ModelInfoResponse, tags=["Model Info"])
async def get_model_info():
    if predictor and predictor.is_unet_ready:
        param_counts = predictor.unet_model.get_parameter_count()
        ckpt_path = os.path.join(settings.ARTIFACTS_DIR, "checkpoints", "best_unet.pth")
        return {
            "model_name": "U-Net (Baseline Encoder-Decoder)",
            "input_resolution": [256, 256],
            "input_channels": 1,
            "output_classes": 1,
            "total_parameters": param_counts["total_parameters"],
            "trainable_parameters": param_counts["trainable_parameters"],
            "checkpoint_available": True,
            "checkpoint_path": ckpt_path if os.path.exists(ckpt_path) else "final_unet.pth",
            "features": predictor.unet_model.features,
            "architecture": "Standard U-Net with skip connections and Monte Carlo Dropout"
        }
    return {
        "model_name": "U-Net (Awaiting Training)",
        "input_resolution": [256, 256],
        "input_channels": 1,
        "output_classes": 1,
        "total_parameters": 0,
        "trainable_parameters": 0,
        "checkpoint_available": False,
        "checkpoint_path": None,
        "features": 32,
        "architecture": "U-Net (Not yet loaded)"
    }

from fastapi.staticfiles import StaticFiles

# Mount static frontend build if present for unified full-stack serving
frontend_dist = os.path.join(BASE_DIR, "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend-static")
else:
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "message": "Brain Tumor Segmentation API is running.",
            "docs_url": "/docs"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)
