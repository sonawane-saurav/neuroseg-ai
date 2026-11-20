"""
Uncertainty Evaluation Endpoint supporting JSON Base64 image payload.
"""

from fastapi import APIRouter, HTTPException, Depends
from backend.app.schemas.schemas import UncertaintyResponse, UncertaintyRequest
from backend.app.database.db import save_analysis

router = APIRouter()

def get_predictor():
    from backend.app.main import predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Prediction service is initializing.")
    return predictor

@router.post("/uncertainty/evaluate", response_model=UncertaintyResponse)
async def evaluate_uncertainty(
    req: UncertaintyRequest,
    predictor = Depends(get_predictor)
):
    if not predictor.is_unet_ready:
        raise HTTPException(status_code=503, detail="Trained U-Net checkpoint not available.")

    try:
        result = predictor.evaluate_uncertainty(
            image_input=req.image_base64,
            num_passes=req.num_passes,
            threshold=req.threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Uncertainty evaluation error: {str(e)}")

    analysis_id = save_analysis(
        filename=req.filename or "mri_scan.png",
        model_used="Baseline-UNet-MCDropout",
        tumor_detected=(result["tumor_pixels"] > 0),
        tumor_pixel_count=result["tumor_pixels"],
        tumor_area_percentage=round((result["tumor_pixels"] / (256 * 256)) * 100.0, 3),
        has_uncertainty=True,
        mean_uncertainty=round(result["mean_uncertainty"], 5),
        details={
            "num_passes": result["num_passes"],
            "max_uncertainty": result["max_uncertainty"],
            "mean_tumor_uncertainty": result["mean_tumor_uncertainty"],
            "mean_boundary_uncertainty": result["mean_boundary_uncertainty"],
            "high_uncertainty_pixels": result["high_uncertainty_pixels"]
        }
    )
    result["analysis_id"] = analysis_id
    return result
