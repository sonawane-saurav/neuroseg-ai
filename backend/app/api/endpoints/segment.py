"""
Segmentation Endpoint supporting JSON Base64 image payload.
"""

from fastapi import APIRouter, HTTPException, Depends
from backend.app.schemas.schemas import SegmentationResponse, SegmentRequest
from backend.app.database.db import save_analysis

router = APIRouter()

def get_predictor():
    from backend.app.main import predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Prediction service is initializing.")
    return predictor

@router.post("/segment", response_model=SegmentationResponse)
async def segment_mri(
    req: SegmentRequest,
    predictor = Depends(get_predictor)
):
    if not predictor.is_unet_ready:
        raise HTTPException(
            status_code=503,
            detail="Trained U-Net model checkpoint is not available. Please ensure model training has completed."
        )

    try:
        result = predictor.segment_image(
            image_input=req.image_base64,
            threshold=req.threshold,
            use_multitask=req.use_multitask
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    model_name = "MultiTask-UNet" if (req.use_multitask and predictor.is_multitask_ready) else "Baseline-UNet"
    cls_label = result["classification"]["predicted_class"] if result.get("classification") else None
    conf = result["classification"]["confidence"] if result.get("classification") else None

    analysis_id = save_analysis(
        filename=req.filename or "mri_scan.png",
        model_used=model_name,
        tumor_detected=result["tumor_detected"],
        tumor_pixel_count=result["tumor_pixel_count"],
        tumor_area_percentage=result["tumor_area_percentage"],
        classification_label=cls_label,
        confidence=conf,
        details={
            "bounding_box": result["bounding_box"],
            "centroid": result["centroid"],
            "resolution": result["resolution"]
        }
    )
    result["analysis_id"] = analysis_id
    return result
