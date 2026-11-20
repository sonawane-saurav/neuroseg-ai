"""
Unit Tests for Uncertainty Quantification and Robustness Degradation Modules.
"""

import unittest
import torch
import numpy as np

from cv.models.unet import UNet
from cv.uncertainty.mc_dropout import MonteCarloDropoutEstimator
from cv.robustness.degradations import MRIDegradationPipeline

class TestResearchFeatures(unittest.TestCase):
    def test_mc_dropout_uncertainty(self):
        model = UNet(in_channels=1, out_classes=1, features=16, dropout_p=0.3)
        estimator = MonteCarloDropoutEstimator(model, device=torch.device("cpu"))
        x = torch.rand(1, 1, 64, 64)
        res = estimator.estimate_uncertainty(x, num_passes=5)

        self.assertIn("mean_prediction", res)
        self.assertIn("variance_map", res)
        self.assertIn("entropy_map", res)
        self.assertEqual(res["variance_map"].shape, (64, 64))
        self.assertTrue(np.all(res["variance_map"] >= 0.0))

    def test_degradations(self):
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        
        # 1. Noise
        noisy = MRIDegradationPipeline.apply_gaussian_noise(img, severity=3)
        self.assertEqual(noisy.shape, img.shape)
        
        # 2. Low Contrast
        low_c = MRIDegradationPipeline.apply_low_contrast(img, severity=3)
        self.assertEqual(low_c.shape, img.shape)
        
        # 3. Motion Blur
        blurred = MRIDegradationPipeline.apply_motion_blur(img, severity=3)
        self.assertEqual(blurred.shape, img.shape)
        
        # 4. Resolution Degradation
        low_res = MRIDegradationPipeline.apply_resolution_degradation(img, severity=3)
        self.assertEqual(low_res.shape, img.shape)

if __name__ == '__main__':
    unittest.main()
