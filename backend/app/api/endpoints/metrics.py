"""
Metrics, Experiments, and Sample Data Endpoints.
"""

from fastapi import APIRouter, HTTPException
import os
import io
import json
import base64
from PIL import Image

from backend.app.core.config import settings

router = APIRouter()

def _load_json_file(relative_path: str):
    full_path = os.path.join(settings.ARTIFACTS_DIR, relative_path)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            return json.load(f)
    return None

@router.get("/metrics")
async def get_metrics():
    """Returns training curves, test set evaluation metrics, and dataset inspection summary."""
    inspection = _load_json_file("metrics/dataset_inspection_report.json")
    unet_history = _load_json_file("metrics/unet_training_history.json")
    unet_summary = _load_json_file("metrics/unet_summary.json")
    multitask_history = _load_json_file("metrics/multitask_training_history.json")
    multitask_summary = _load_json_file("metrics/multitask_summary.json")
    comparison = _load_json_file("metrics/model_comparison.json")
    robustness = _load_json_file("metrics/robustness_benchmark.json")

    return {
        "dataset_inspection": inspection,
        "unet_training": {
            "summary": unet_summary,
            "history": unet_history
        },
        "multitask_training": {
            "summary": multitask_summary,
            "history": multitask_history
        },
        "model_comparison": comparison,
        "robustness_benchmark": robustness
    }

@router.get("/experiments")
async def get_experiments():
    """Returns comparative experimental findings between Baseline U-Net and Multi-Task U-Net."""
    comparison = _load_json_file("metrics/model_comparison.json")
    if comparison is None:
        return {
            "status": "pending",
            "message": "Model evaluation has not yet been executed."
        }
    return comparison

@router.get("/samples")
async def get_test_samples():
    """
    Returns preloaded real BRISC 2025 test samples with base64 images and ground truth masks
    for immediate interactive exploration in the UI without needing manual uploads.
    """
    preds_dir = os.path.join(settings.ARTIFACTS_DIR, "predictions")
    samples = []
    
    # Check if prepared samples exist in predictions/
    for idx in range(6):
        img_p = os.path.join(preds_dir, f"sample_{idx}_image.png")
        gt_p = os.path.join(preds_dir, f"sample_{idx}_gt.png")
        if os.path.exists(img_p) and os.path.exists(gt_p):
            with open(img_p, "rb") as f:
                img_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
            with open(gt_p, "rb") as f:
                gt_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
            
            samples.append({
                "id": idx + 1,
                "name": f"BRISC Test Sample {idx + 1}",
                "image_base64": img_b64,
                "gt_mask_base64": gt_b64
            })

    # If predictions/ has no pre-exported samples yet, load from cv/data/brisc2025 test partition
    if not samples:
        seg_test_dir = os.path.join(settings.DATA_DIR, "segmentation_task", "test")
        img_dir = os.path.join(seg_test_dir, "images")
        mask_dir = os.path.join(seg_test_dir, "masks")
        if os.path.exists(img_dir):
            files = sorted(os.listdir(img_dir))[:6]
            for idx, fn in enumerate(files):
                img_p = os.path.join(img_dir, fn)
                stem = os.path.splitext(fn)[0]
                mask_p = os.path.join(mask_dir, f"{stem}.png")
                if os.path.exists(mask_p):
                    # Resize to 256x256 for clean transmission
                    with Image.open(img_p) as im:
                        im = im.resize((256, 256), Image.BILINEAR)
                        buf = io.BytesIO()
                        im.save(buf, format="PNG")
                        img_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
                    with Image.open(mask_p) as m:
                        m = m.resize((256, 256), Image.NEAREST)
                        buf = io.BytesIO()
                        m.save(buf, format="PNG")
                        gt_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
                    
                    samples.append({
                        "id": idx + 1,
                        "name": f"Test Sample: {stem[:24]}",
                        "image_base64": img_b64,
                        "gt_mask_base64": gt_b64
                    })

    return {"samples": samples}
