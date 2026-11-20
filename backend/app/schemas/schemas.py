"""
Pydantic Request and Response Schemas.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    project_name: str
    device: str
    torch_version: str
    cuda_available: bool
    unet_available: bool
    multitask_available: bool

class ModelInfoResponse(BaseModel):
    model_name: str
    input_resolution: List[int]
    input_channels: int
    output_classes: int
    total_parameters: int
    trainable_parameters: int
    checkpoint_available: bool
    checkpoint_path: Optional[str] = None
    features: int
    architecture: str

class BoundingBox(BaseModel):
    ymin: int
    ymax: int
    xmin: int
    xmax: int
    width: int
    height: int

class Centroid(BaseModel):
    x: float
    y: float

class SegmentRequest(BaseModel):
    image_base64: str
    threshold: float = 0.5
    use_multitask: bool = False
    filename: Optional[str] = "mri_scan.png"

class UncertaintyRequest(BaseModel):
    image_base64: str
    num_passes: int = 15
    threshold: float = 0.5
    filename: Optional[str] = "mri_scan.png"

class RobustnessRequest(BaseModel):
    image_base64: str
    degradation_type: str = "gaussian_noise"
    severity: int = 3
    filename: Optional[str] = "mri_scan.png"

class ClassificationResult(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]

class SegmentationResponse(BaseModel):
    tumor_detected: bool
    tumor_pixel_count: int
    tumor_area_percentage: float
    bounding_box: Optional[BoundingBox] = None
    centroid: Optional[Centroid] = None
    classification: Optional[ClassificationResult] = None
    mask_base64: str
    overlay_base64: str
    original_base64: str
    resolution: List[int]
    analysis_id: Optional[int] = None

class UncertaintyResponse(BaseModel):
    num_passes: int
    mean_uncertainty: float
    max_uncertainty: float
    mean_tumor_uncertainty: float
    mean_boundary_uncertainty: float
    high_uncertainty_pixels: int
    tumor_pixels: int
    mask_base64: str
    heatmap_base64: str
    uncertainty_overlay_base64: str
    original_base64: str
    analysis_id: Optional[int] = None

class RobustnessMetricsDiff(BaseModel):
    clean_dice: float
    degraded_dice: float
    absolute_dice_drop: float
    relative_dice_drop_pct: float
    clean_iou: float
    degraded_iou: float

class RobustnessResponse(BaseModel):
    degradation_type: str
    severity: int
    original_base64: str
    degraded_image_base64: str
    clean_mask_base64: str
    degraded_mask_base64: str
    clean_overlay_base64: str
    degraded_overlay_base64: str
    clean_tumor_pixels: int
    degraded_tumor_pixels: int
    pixel_difference: int
    metrics_comparison: Optional[RobustnessMetricsDiff] = None
    analysis_id: Optional[int] = None

class AnalysisHistoryItem(BaseModel):
    id: int
    filename: str
    timestamp: str
    model_used: str
    tumor_detected: bool
    tumor_pixel_count: int
    tumor_area_percentage: float
    classification_label: Optional[str] = None
    confidence: Optional[float] = None
    has_uncertainty: bool
    mean_uncertainty: float
    degradation_type: str
    severity: int
    details: Optional[Dict[str, Any]] = None
