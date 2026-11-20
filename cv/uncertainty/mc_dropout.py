"""
Feature One: Monte Carlo Dropout for Epistemic Uncertainty Estimation in Tumor Segmentation.
Performs stochastic forward passes with dropout enabled at test-time to quantify
pixel-level predictive uncertainty, identify borderline boundaries, and evaluate model confidence.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

class MonteCarloDropoutEstimator:
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Args:
            model: Segmentation model with dropout layers.
            device: torch.device (CPU or CUDA).
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def estimate_uncertainty(
        self,
        image_tensor: torch.Tensor,
        num_passes: int = 15,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Runs T stochastic forward passes with dropout enabled.
        Args:
            image_tensor: (1, C, H, W) or (C, H, W) preprocessed input tensor.
            num_passes: Number of Monte Carlo iterations (T >= 5).
            threshold: Probability threshold for final binary mask.
        Returns:
            Dictionary containing predictive mean, variance, entropy, binary mask,
            and uncertainty summary statistics.
        """
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        image_tensor = image_tensor.to(self.device)

        # 1. Ensure model is in eval mode, but activate dropout layers
        self.model.eval()
        if hasattr(self.model, 'enable_mc_dropout'):
            self.model.enable_mc_dropout()
        else:
            for m in self.model.modules():
                if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                    m.train()

        # 2. Collect Monte Carlo predictions
        predictions = []
        with torch.no_grad():
            for _ in range(num_passes):
                outputs = self.model(image_tensor)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take segmentation logits if multi-task
                probs = torch.sigmoid(outputs)
                predictions.append(probs.cpu().numpy())

        # Shape: (T, 1, 1, H, W) -> squeeze to (T, H, W)
        preds_stack = np.squeeze(np.array(predictions)) # (T, H, W)

        # 3. Calculate Predictive Statistics
        # Predictive Mean: mu(x) = (1/T) sum p_t(x)
        mean_pred = np.mean(preds_stack, axis=0)
        
        # Predictive Variance (Epistemic Uncertainty): sigma^2(x) = (1/T) sum (p_t(x) - mu(x))^2
        variance_map = np.var(preds_stack, axis=0)

        # Predictive Shannon Entropy: H(x) = - mu log2(mu) - (1-mu) log2(1-mu)
        eps = 1e-7
        entropy_map = -(mean_pred * np.log2(mean_pred + eps) + (1.0 - mean_pred) * np.log2(1.0 - mean_pred + eps))

        # Binary Segmentation Mask
        binary_mask = (mean_pred > threshold).astype(np.uint8)

        # High-uncertainty region mask (regions where variance exceeds 90th percentile of tumor area or > 0.05)
        high_uncert_mask = (variance_map > 0.05).astype(np.uint8)
        
        # Tumor region stats
        tumor_pixels = int(np.sum(binary_mask))
        high_uncert_pixels = int(np.sum(high_uncert_mask))
        mean_tumor_uncertainty = float(np.mean(variance_map[binary_mask == 1])) if tumor_pixels > 0 else 0.0
        boundary_pixels = self._get_boundary_mask(binary_mask)
        mean_boundary_uncertainty = float(np.mean(variance_map[boundary_pixels == 1])) if np.sum(boundary_pixels) > 0 else 0.0

        return {
            "mean_prediction": mean_pred,
            "variance_map": variance_map,
            "entropy_map": entropy_map,
            "binary_mask": binary_mask,
            "num_passes": num_passes,
            "mean_uncertainty": float(np.mean(variance_map)),
            "max_uncertainty": float(np.max(variance_map)),
            "mean_tumor_uncertainty": mean_tumor_uncertainty,
            "mean_boundary_uncertainty": mean_boundary_uncertainty,
            "high_uncertainty_pixels": high_uncert_pixels,
            "tumor_pixels": tumor_pixels,
            "total_pixels": int(mean_pred.size)
        }

    def _get_boundary_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """Extracts 1-pixel boundary of predicted tumor using morphological gradient."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary_mask, kernel, iterations=1)
        eroded = cv2.erode(binary_mask, kernel, iterations=1)
        return dilated - eroded

    def generate_visualization(
        self,
        image_np: np.ndarray,
        result: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Creates an academic 4-panel visual summary:
        [1] Original MRI, [2] Predicted Mask & Overlay, [3] Variance Heatmap, [4] High Uncertainty Highlight.
        """
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        plt.style.use('dark_background')

        # 1. Original MRI
        axes[0].imshow(image_np, cmap='gray')
        axes[0].set_title("Original T1 MRI", fontsize=11, color='#38bdf8')
        axes[0].axis('off')

        # 2. Predicted Mask Overlay
        overlay = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB) if image_np.ndim == 2 else image_np.copy()
        mask = result["binary_mask"]
        overlay[mask == 1, 0] = np.clip(overlay[mask == 1, 0].astype(int) + 120, 0, 255)
        axes[1].imshow(overlay)
        axes[1].set_title(f"Prediction (Tumor Pixels: {result['tumor_pixels']})", fontsize=11, color='#10b981')
        axes[1].axis('off')

        # 3. Variance Uncertainty Heatmap
        var_im = axes[2].imshow(result["variance_map"], cmap='plasma', vmin=0, vmax=max(0.1, result["max_uncertainty"]))
        axes[2].set_title(f"Epistemic Variance (T={result['num_passes']})", fontsize=11, color='#f59e0b')
        axes[2].axis('off')
        plt.colorbar(var_im, ax=axes[2], fraction=0.046, pad=0.04)

        # 4. Boundary Uncertainty Overlay
        uncert_overlay = overlay.copy()
        high_u = result["variance_map"] > 0.05
        uncert_overlay[high_u, 0] = 255
        uncert_overlay[high_u, 1] = 200
        uncert_overlay[high_u, 2] = 0
        axes[3].imshow(uncert_overlay)
        axes[3].set_title(f"Borderline Uncertainty Regions", fontsize=11, color='#f43f5e')
        axes[3].axis('off')

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=180, facecolor='#090d16')
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        plt.close()
        return rgba
