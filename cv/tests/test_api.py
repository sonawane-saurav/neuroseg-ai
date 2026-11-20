"""
Integration and Unit Tests for FastAPI Endpoints.
"""

import os
import unittest
import base64
import numpy as np
import cv2
from starlette.testclient import TestClient

from backend.app.main import app

class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a synthetic 256x256 test MRI image base64
        synthetic_img = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(synthetic_img, (128, 128), 40, 200, -1)
        _, buffer = cv2.imencode('.png', synthetic_img)
        cls.test_b64 = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def test_root_serves_frontend(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("<html", resp.text.lower())

    def test_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("unet_available", data)
        self.assertTrue(data.get("unet_available"))

    def test_model_info_endpoint(self):
        resp = self.client.get("/api/model/info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("model_name", data)
        self.assertIn("total_parameters", data)
        self.assertTrue(data.get("checkpoint_available"))

    def test_metrics_endpoint(self):
        resp = self.client.get("/api/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("dataset_inspection", data)
        self.assertIn("model_comparison", data)

    def test_samples_endpoint(self):
        resp = self.client.get("/api/samples")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("samples", data)
        self.assertIsInstance(data["samples"], list)
        if len(data["samples"]) > 0:
            sample = data["samples"][0]
            self.assertIn("id", sample)
            self.assertIn("image_base64", sample)

    def test_segment_endpoint(self):
        payload = {
            "image_base64": self.test_b64,
            "threshold": 0.5,
            "use_multitask": False
        }
        resp = self.client.post("/api/segment", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("mask_base64", data)
        self.assertIn("overlay_base64", data)
        self.assertIn("tumor_detected", data)
        self.assertIn("tumor_pixel_count", data)

    def test_uncertainty_endpoint(self):
        payload = {
            "image_base64": self.test_b64,
            "num_passes": 3
        }
        resp = self.client.post("/api/uncertainty/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("heatmap_base64", data)
        self.assertIn("mean_uncertainty", data)
        self.assertIn("uncertainty_overlay_base64", data)

    def test_robustness_endpoint(self):
        payload = {
            "image_base64": self.test_b64,
            "degradation_type": "gaussian_noise",
            "severity": 2
        }
        resp = self.client.post("/api/robustness/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("degraded_image_base64", data)
        self.assertIn("clean_mask_base64", data)
        self.assertIn("degraded_mask_base64", data)

    def test_history_endpoint(self):
        resp = self.client.get("/api/history")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)

if __name__ == '__main__':
    unittest.main()
