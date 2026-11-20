"""
Robustness Evaluation Endpoint supporting JSON Base64 image payload.
"""

from fastapi import APIRouter, HTTPException, Depends
from backend.app.schemas.schemas import RobustnessResponse, RobustnessRequest
from backend.app.database.db import save_analysis

router = APIRouter()

def get_predictor():
    from backend.app.main import predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Prediction service is initializing.")
    return predictor

@router.post("/robustness/evaluate", response_model=RobustnessResponse)
async def evaluate_robustness(
    req: RobustnessRequest,
    predictor = Depends(get_predictor)
):
    if not predictor.is_unet_ready:
        raise HTTPException(status_code=503, detail="Trained model checkpoint not available.")

    try:
        result = predictor.evaluate_robustness(
            image_input=req.image_base64,
            degradation_type=req.degradation_type,
            severity=req.severity
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Robustness evaluation error: {str(e)}")

    analysis_id = save_analysis(
        filename=req.filename or "mri_scan.png",
        model_used="Baseline-UNet-RobustnessLab",
        tumor_detected=(result["degraded_tumor_pixels"] > 0),
        tumor_pixel_count=result["degraded_tumor_pixels"],
        tumor_area_percentage=round((result["degraded_tumor_pixels"] / (256 * 256)) * 100.0, 3),
        degradation_type=req.degradation_type,
        severity=req.severity,
        details={
            "clean_tumor_pixels": result["clean_tumor_pixels"],
            "degraded_tumor_pixels": result["degraded_tumor_pixels"],
            "pixel_difference": result["pixel_difference"]
        }
    )
    result["analysis_id"] = analysis_id
    return result
