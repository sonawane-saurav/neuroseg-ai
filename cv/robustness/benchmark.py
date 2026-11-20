"""
Systematic Robustness Evaluation & Stress Testing Module.
Quantifies performance degradation of the segmentation model across 4 clinical
image quality hazards across 5 severity levels against held-out ground truth.
"""

from typing import Dict, Any, List, Optional
import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cv.robustness.degradations import MRIDegradationPipeline
from cv.preprocessing.transform import MRIPreprocessor
from cv.training.metrics import calculate_segmentation_metrics

class RobustnessBenchmark:
    def __init__(
        self,
        model: nn.Module,
        preprocessor: Optional[MRIPreprocessor] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.preprocessor = preprocessor or MRIPreprocessor()

    def evaluate_single_sample(
        self,
        image_np: np.ndarray,
        ground_truth_mask: np.ndarray,
        degradation_type: str,
        severity: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluates one MRI slice under a specific degradation and compares against clean prediction.
        Note: Ground truth mask is NEVER modified because anatomy is constant.
        """
        # 1. Clean inference
        clean_tensor = self.preprocessor.preprocess_image(image_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            clean_out = self.model(clean_tensor)
            if isinstance(clean_out, tuple): clean_out = clean_out[0]
            clean_probs = torch.sigmoid(clean_out).cpu().numpy()[0, 0]
        clean_mask = (clean_probs > 0.5).astype(np.uint8)
        clean_metrics = calculate_segmentation_metrics(clean_mask, ground_truth_mask)

        # 2. Apply degradation
        degraded_img = MRIDegradationPipeline.apply_degradation(image_np, degradation_type, severity)
        deg_tensor = self.preprocessor.preprocess_image(degraded_img).unsqueeze(0).to(self.device)

        # 3. Degraded inference
        with torch.no_grad():
            deg_out = self.model(deg_tensor)
            if isinstance(deg_out, tuple): deg_out = deg_out[0]
            deg_probs = torch.sigmoid(deg_out).cpu().numpy()[0, 0]
        deg_mask = (deg_probs > 0.5).astype(np.uint8)
        deg_metrics = calculate_segmentation_metrics(deg_mask, ground_truth_mask)

        # 4. Performance drops
        dice_drop = clean_metrics["dice"] - deg_metrics["dice"]
        iou_drop = clean_metrics["iou"] - deg_metrics["iou"]
        relative_dice_drop = (dice_drop / (clean_metrics["dice"] + 1e-7)) * 100.0

        return {
            "degradation_type": degradation_type,
            "severity": severity,
            "clean_metrics": clean_metrics,
            "degraded_metrics": deg_metrics,
            "absolute_dice_drop": float(dice_drop),
            "relative_dice_drop_pct": float(relative_dice_drop),
            "absolute_iou_drop": float(iou_drop),
            "clean_prediction": clean_mask,
            "degraded_prediction": deg_mask,
            "degraded_image": degraded_img
        }

    def run_benchmark_suite(
        self,
        test_samples: List[Dict[str, Any]],
        output_dir: str,
        degradations: Optional[List[str]] = None,
        severities: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Runs comprehensive multi-sample stress testing across all degradations and severities.
        """
        os.makedirs(output_dir, exist_ok=True)
        degradations = degradations or ["gaussian_noise", "low_contrast", "motion_blur", "resolution_degradation"]
        severities = severities or [1, 2, 3, 4, 5]

        print(f"Running Robustness Benchmark on {len(test_samples)} test samples...")
        results_by_condition = {}

        # First evaluate baseline clean performance on the test sample cohort
        clean_dices, clean_ious, clean_precs, clean_recs = [], [], [], []
        for s in test_samples:
            img = np.array(Image.open(s["image_path"]))
            raw_gt = (np.array(Image.open(s["mask_path"])) > 127).astype(np.uint8)
            gt = cv2.resize(raw_gt, (self.preprocessor.target_size[1], self.preprocessor.target_size[0]), interpolation=cv2.INTER_NEAREST)
            t = self.preprocessor.preprocess_image(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(t)
                if isinstance(out, tuple): out = out[0]
                prob = torch.sigmoid(out).cpu().numpy()[0, 0]
            m = calculate_segmentation_metrics(prob > 0.5, gt)
            clean_dices.append(m["dice"])
            clean_ious.append(m["iou"])
            clean_precs.append(m["precision"])
            clean_recs.append(m["recall"])

        baseline_summary = {
            "mean_dice": float(np.mean(clean_dices)),
            "mean_iou": float(np.mean(clean_ious)),
            "mean_precision": float(np.mean(clean_precs)),
            "mean_recall": float(np.mean(clean_recs)),
        }
        print(f"Baseline Clean Test Metrics: Dice={baseline_summary['mean_dice']:.4f}, IoU={baseline_summary['mean_iou']:.4f}")

        # Benchmark every degradation and severity
        for deg in degradations:
            results_by_condition[deg] = {}
            for sev in severities:
                dices, ious, precs, recs = [], [], [], []
                for s in test_samples:
                    img = np.array(Image.open(s["image_path"]))
                    raw_gt = (np.array(Image.open(s["mask_path"])) > 127).astype(np.uint8)
                    gt = cv2.resize(raw_gt, (self.preprocessor.target_size[1], self.preprocessor.target_size[0]), interpolation=cv2.INTER_NEAREST)
                    
                    deg_img = MRIDegradationPipeline.apply_degradation(img, deg, sev)
                    t = self.preprocessor.preprocess_image(deg_img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        out = self.model(t)
                        if isinstance(out, tuple): out = out[0]
                        prob = torch.sigmoid(out).cpu().numpy()[0, 0]
                    m = calculate_segmentation_metrics(prob > 0.5, gt)
                    dices.append(m["dice"])
                    ious.append(m["iou"])
                    precs.append(m["precision"])
                    recs.append(m["recall"])

                mean_d = float(np.mean(dices))
                mean_i = float(np.mean(ious))
                abs_drop = float(baseline_summary["mean_dice"] - mean_d)
                rel_drop_pct = float((abs_drop / (baseline_summary["mean_dice"] + 1e-7)) * 100.0)

                results_by_condition[deg][str(sev)] = {
                    "mean_dice": mean_d,
                    "mean_iou": mean_i,
                    "mean_precision": float(np.mean(precs)),
                    "mean_recall": float(np.mean(recs)),
                    "absolute_dice_drop": abs_drop,
                    "relative_dice_drop_pct": rel_drop_pct
                }
                print(f"  [{deg}] Sev {sev}: Dice={mean_d:.4f} (drop: -{rel_drop_pct:.1f}%)")

        benchmark_report = {
            "num_test_samples": len(test_samples),
            "baseline_clean": baseline_summary,
            "conditions": results_by_condition
        }

        # Save JSON
        json_path = os.path.join(output_dir, "robustness_benchmark.json")
        with open(json_path, 'w') as f:
            json.dump(benchmark_report, f, indent=2)

        # Generate plots
        self._plot_robustness_curves(benchmark_report, output_dir)
        print(f"Robustness Benchmark complete! Saved to: {json_path}")
        return benchmark_report

    def _plot_robustness_curves(self, report: Dict[str, Any], output_dir: str):
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        baseline_dice = report["baseline_clean"]["mean_dice"]
        colors = {
            "gaussian_noise": "#f43f5e",
            "low_contrast": "#38bdf8",
            "motion_blur": "#eab308",
            "resolution_degradation": "#10b981"
        }
        labels = {
            "gaussian_noise": "Gaussian Noise",
            "low_contrast": "Low Contrast",
            "motion_blur": "Motion Blur",
            "resolution_degradation": "Resolution Drop"
        }

        # Plot 1: Dice score vs Severity
        ax1 = axes[0]
        ax1.axhline(baseline_dice, color='#94a3b8', linestyle='--', label=f'Clean Baseline ({baseline_dice:.3f})')
        for deg, sevs in report["conditions"].items():
            x = [int(k) for k in sevs.keys()]
            y = [sevs[str(k)]["mean_dice"] for k in x]
            ax1.plot(x, y, marker='o', linewidth=2, color=colors.get(deg, '#ffffff'), label=labels.get(deg, deg))
        ax1.set_title("Dice Score vs. Degradation Severity", color='#f8fafc', fontsize=12)
        ax1.set_xlabel("Severity Level (1 = Mild, 5 = Extreme)", color='#cbd5e1')
        ax1.set_ylabel("Dice Coefficient", color='#cbd5e1')
        ax1.set_xticks([1, 2, 3, 4, 5])
        ax1.grid(True, alpha=0.2)
        ax1.legend(loc='lower left', fontsize=9)

        # Plot 2: Relative Performance Drop % vs Severity
        ax2 = axes[1]
        for deg, sevs in report["conditions"].items():
            x = [int(k) for k in sevs.keys()]
            y = [sevs[str(k)]["relative_dice_drop_pct"] for k in x]
            ax2.plot(x, y, marker='s', linewidth=2, color=colors.get(deg, '#ffffff'), label=labels.get(deg, deg))
        ax2.set_title("Relative Performance Degradation (%)", color='#f8fafc', fontsize=12)
        ax2.set_xlabel("Severity Level", color='#cbd5e1')
        ax2.set_ylabel("Dice Drop (%)", color='#cbd5e1')
        ax2.set_xticks([1, 2, 3, 4, 5])
        ax2.grid(True, alpha=0.2)
        ax2.legend(loc='upper left', fontsize=9)

        plt.tight_layout()
        chart_path = os.path.join(output_dir, "robustness_curves.png")
        plt.savefig(chart_path, dpi=180, facecolor='#090d16')
        plt.close()
