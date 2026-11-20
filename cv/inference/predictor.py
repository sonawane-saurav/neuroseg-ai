"""
Unified Model Inference and Experimentation Engine.
Serves as the bridge between trained PyTorch checkpoints and the FastAPI backend.
Supports single-pass segmentation, Monte Carlo Dropout uncertainty estimation,
synthetic degradation testing, and multi-task pathology classification.
"""

from typing import Dict, Any, Optional, Tuple, Union
import os
import io
import base64
import numpy as np
import torch
from PIL import Image
import cv2

from cv.models.unet import UNet
from cv.models.multitask_unet import MultiTaskUNet
from cv.preprocessing.transform import MRIPreprocessor
from cv.uncertainty.mc_dropout import MonteCarloDropoutEstimator
from cv.robustness.degradations import MRIDegradationPipeline
from cv.training.metrics import calculate_segmentation_metrics

class BrainTumorPredictor:
    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        device: Optional[str] = None
    ):
        self.artifacts_dir = os.path.abspath(artifacts_dir)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.preprocessor = MRIPreprocessor(target_size=(256, 256), in_channels=1)

        self.unet_model: Optional[UNet] = None
        self.multitask_model: Optional[MultiTaskUNet] = None
        self.mc_estimator: Optional[MonteCarloDropoutEstimator] = None

        self._load_checkpoints()

    def _load_checkpoints(self):
        checkpoints_dir = os.path.join(self.artifacts_dir, 'checkpoints')
        
        # 1. Load Baseline U-Net
        unet_path = os.path.join(checkpoints_dir, 'best_unet.pth')
        if not os.path.exists(unet_path):
            unet_path = os.path.join(checkpoints_dir, 'final_unet.pth')
            
        if os.path.exists(unet_path):
            try:
                state = torch.load(unet_path, map_location=self.device)
                features = state.get('features', 32)
                self.unet_model = UNet(in_channels=1, out_classes=1, features=features, dropout_p=0.2).to(self.device)
                self.unet_model.load_state_dict(state['model_state_dict'])
                self.unet_model.eval()
                self.mc_estimator = MonteCarloDropoutEstimator(self.unet_model, device=self.device)
                print(f"[Predictor] Baseline U-Net checkpoint loaded successfully from: {unet_path}")
            except Exception as e:
                print(f"[Predictor] Error loading U-Net checkpoint: {e}")
        else:
            print(f"[Predictor] Note: No U-Net checkpoint found at {checkpoints_dir}.")

        # 2. Load Multi-Task Model
        mt_path = os.path.join(checkpoints_dir, 'best_multitask.pth')
        if not os.path.exists(mt_path):
            mt_path = os.path.join(checkpoints_dir, 'final_multitask.pth')

        if os.path.exists(mt_path):
            try:
                state = torch.load(mt_path, map_location=self.device)
                features = state.get('features', 32)
                self.multitask_model = MultiTaskUNet(in_channels=1, out_classes=1, num_classes=4, features=features, dropout_p=0.2).to(self.device)
                self.multitask_model.load_state_dict(state['model_state_dict'])
                self.multitask_model.eval()
                print(f"[Predictor] Multi-Task model checkpoint loaded successfully from: {mt_path}")
            except Exception as e:
                print(f"[Predictor] Error loading Multi-Task checkpoint: {e}")

    @property
    def is_unet_ready(self) -> bool:
        return self.unet_model is not None

    @property
    def is_multitask_ready(self) -> bool:
        return self.multitask_model is not None

    def segment_image(
        self,
        image_input: Union[bytes, np.ndarray, Image.Image],
        threshold: float = 0.5,
        use_multitask: bool = False
    ) -> Dict[str, Any]:
        """
        Executes single-pass tumor segmentation.
        Returns:
            - binary mask
            - probability heatmap
            - base64 encoded mask & overlay
            - tumor morphometric statistics (area, bounding box, centroid)
            - classification predictions (if multi-task model requested)
        """
        model = self.multitask_model if (use_multitask and self.is_multitask_ready) else self.unet_model
        if model is None:
            raise RuntimeError("Model checkpoint is not loaded. Train or deploy a checkpoint first.")

        # Load into PIL/numpy
        raw_pil = self._to_pil_image(image_input)
        raw_np = np.array(raw_pil.convert('L'))

        # Preprocess
        tensor = self.preprocessor.preprocess_image(raw_np).unsqueeze(0).to(self.device)

        model.eval()
        with torch.no_grad():
            outputs = model(tensor)
            cls_info = None
            if isinstance(outputs, tuple):
                seg_logits, cls_logits = outputs
                probs = torch.sigmoid(seg_logits).cpu().numpy()[0, 0]
                cls_probs = torch.softmax(cls_logits, dim=-1).cpu().numpy()[0]
                classes = ['Glioma', 'Meningioma', 'Pituitary Tumor', 'No Tumor']
                pred_c_idx = int(np.argmax(cls_probs))
                cls_info = {
                    "predicted_class": classes[pred_c_idx],
                    "confidence": float(cls_probs[pred_c_idx]),
                    "all_probabilities": {c: float(p) for c, p in zip(classes, cls_probs)}
                }
            else:
                probs = torch.sigmoid(outputs).cpu().numpy()[0, 0]

        # Binary Mask
        bin_mask = (probs > threshold).astype(np.uint8)

        # Morphometrics
        morphometrics = self._compute_morphometrics(bin_mask)

        # Visualizations
        denorm_img = self.preprocessor.denormalize_image(tensor[0])
        overlay_rgb = self._create_overlay(denorm_img, bin_mask)

        return {
            "tumor_detected": morphometrics["tumor_detected"],
            "tumor_pixel_count": morphometrics["tumor_pixel_count"],
            "tumor_area_percentage": morphometrics["tumor_area_percentage"],
            "bounding_box": morphometrics["bounding_box"],
            "centroid": morphometrics["centroid"],
            "classification": cls_info,
            "mask_base64": self._numpy_to_base64(bin_mask * 255),
            "overlay_base64": self._numpy_to_base64(overlay_rgb),
            "original_base64": self._numpy_to_base64(denorm_img),
            "resolution": list(bin_mask.shape)
        }

    def evaluate_uncertainty(
        self,
        image_input: Union[bytes, np.ndarray, Image.Image],
        num_passes: int = 15,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Runs Monte Carlo Dropout epistemic uncertainty quantification.
        """
        if not self.is_unet_ready or self.mc_estimator is None:
            raise RuntimeError("U-Net model checkpoint is not loaded for uncertainty evaluation.")

        raw_pil = self._to_pil_image(image_input)
        raw_np = np.array(raw_pil.convert('L'))
        tensor = self.preprocessor.preprocess_image(raw_np).unsqueeze(0)

        uncert_result = self.mc_estimator.estimate_uncertainty(
            tensor, num_passes=num_passes, threshold=threshold
        )

        denorm_img = self.preprocessor.denormalize_image(tensor[0])
        bin_mask = uncert_result["binary_mask"]
        overlay_rgb = self._create_overlay(denorm_img, bin_mask)

        # Variance heatmap color rendering (using Plasma/Viridis)
        var_norm = np.clip(uncert_result["variance_map"] / 0.15, 0.0, 1.0)
        heatmap_colored = cv2.applyColorMap((var_norm * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Combined uncertainty overlay
        combined_overlay = denorm_img.copy()
        if combined_overlay.ndim == 2:
            combined_overlay = np.stack([combined_overlay]*3, axis=-1)
        # Blend heatmap onto MRI: 60% MRI + 40% heatmap
        blended = cv2.addWeighted(combined_overlay, 0.6, heatmap_rgb, 0.4, 0)

        return {
            "num_passes": uncert_result["num_passes"],
            "mean_uncertainty": uncert_result["mean_uncertainty"],
            "max_uncertainty": uncert_result["max_uncertainty"],
            "mean_tumor_uncertainty": uncert_result["mean_tumor_uncertainty"],
            "mean_boundary_uncertainty": uncert_result["mean_boundary_uncertainty"],
            "high_uncertainty_pixels": uncert_result["high_uncertainty_pixels"],
            "tumor_pixels": uncert_result["tumor_pixels"],
            "mask_base64": self._numpy_to_base64(bin_mask * 255),
            "heatmap_base64": self._numpy_to_base64(heatmap_rgb),
            "uncertainty_overlay_base64": self._numpy_to_base64(blended),
            "original_base64": self._numpy_to_base64(denorm_img)
        }

    def evaluate_robustness(
        self,
        image_input: Union[bytes, np.ndarray, Image.Image],
        degradation_type: str,
        severity: int = 3,
        ground_truth_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Applies synthetic degradation and benchmarks segmentation reliability.
        """
        if not self.is_unet_ready:
            raise RuntimeError("U-Net checkpoint is not loaded for robustness testing.")

        raw_pil = self._to_pil_image(image_input)
        raw_np = np.array(raw_pil.convert('L'))

        # 1. Clean inference
        clean_seg = self.segment_image(raw_np)

        # 2. Degrade image
        degraded_np = MRIDegradationPipeline.apply_degradation(raw_np, degradation_type, severity)

        # 3. Degraded inference
        deg_seg = self.segment_image(degraded_np)

        # 4. Compare with ground truth if provided, or against clean prediction as pseudo-reference
        ref_mask = ground_truth_mask if ground_truth_mask is not None else (clean_seg["tumor_pixel_count"] > 0)
        metrics_diff = None
        if ground_truth_mask is not None:
            # Resize GT mask to match target
            gt_resized = cv2.resize((ground_truth_mask > 0).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
            clean_m = calculate_segmentation_metrics(clean_seg["tumor_pixel_count"] > 0, gt_resized)
            deg_m = calculate_segmentation_metrics(deg_seg["tumor_pixel_count"] > 0, gt_resized)
            metrics_diff = {
                "clean_dice": clean_m["dice"],
                "degraded_dice": deg_m["dice"],
                "absolute_dice_drop": clean_m["dice"] - deg_m["dice"],
                "relative_dice_drop_pct": ((clean_m["dice"] - deg_m["dice"]) / (clean_m["dice"] + 1e-7)) * 100.0,
                "clean_iou": clean_m["iou"],
                "degraded_iou": deg_m["iou"]
            }

        return {
            "degradation_type": degradation_type,
            "severity": severity,
            "original_base64": clean_seg["original_base64"],
            "degraded_image_base64": deg_seg["original_base64"],
            "clean_mask_base64": clean_seg["mask_base64"],
            "degraded_mask_base64": deg_seg["mask_base64"],
            "clean_overlay_base64": clean_seg["overlay_base64"],
            "degraded_overlay_base64": deg_seg["overlay_base64"],
            "clean_tumor_pixels": clean_seg["tumor_pixel_count"],
            "degraded_tumor_pixels": deg_seg["tumor_pixel_count"],
            "pixel_difference": deg_seg["tumor_pixel_count"] - clean_seg["tumor_pixel_count"],
            "metrics_comparison": metrics_diff
        }

    def _to_pil_image(self, img_input: Union[bytes, str, np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(img_input, str):
            if ';base64,' in img_input:
                img_input = img_input.split(';base64,')[1]
            raw_bytes = base64.b64decode(img_input)
            return Image.open(io.BytesIO(raw_bytes))
        elif isinstance(img_input, bytes):
            return Image.open(io.BytesIO(img_input))
        elif isinstance(img_input, np.ndarray):
            return Image.fromarray(img_input)
        elif isinstance(img_input, Image.Image):
            return img_input
        raise ValueError("Unsupported image input format.")

    def _compute_morphometrics(self, bin_mask: np.ndarray) -> Dict[str, Any]:
        pixels = int(np.sum(bin_mask))
        total = int(bin_mask.size)
        pct = float((pixels / total) * 100.0)

        if pixels == 0:
            return {
                "tumor_detected": False,
                "tumor_pixel_count": 0,
                "tumor_area_percentage": 0.0,
                "bounding_box": None,
                "centroid": None
            }

        # Connected components and bounding box
        y_indices, x_indices = np.where(bin_mask == 1)
        ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
        xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))
        cy, cx = float(np.mean(y_indices)), float(np.mean(x_indices))

        return {
            "tumor_detected": True,
            "tumor_pixel_count": pixels,
            "tumor_area_percentage": round(pct, 3),
            "bounding_box": {"ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax, "width": xmax - xmin + 1, "height": ymax - ymin + 1},
            "centroid": {"x": round(cx, 1), "y": round(cy, 1)}
        }

    def _create_overlay(self, image_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if image_np.ndim == 2:
            rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        else:
            rgb = image_np.copy()
        # Add high-visibility medical emerald/red mask tint
        rgb[mask == 1, 0] = np.clip(rgb[mask == 1, 0].astype(int) + 140, 0, 255)
        rgb[mask == 1, 1] = rgb[mask == 1, 1] // 2
        rgb[mask == 1, 2] = rgb[mask == 1, 2] // 2
        return rgb

    def _numpy_to_base64(self, arr: np.ndarray) -> str:
        pil = Image.fromarray(arr.astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
