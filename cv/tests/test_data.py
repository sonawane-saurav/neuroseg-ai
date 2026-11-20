"""
Unit Tests for Data Pipeline and Dataset Loading.
"""

import os
import unittest
import torch
import numpy as np

from cv.data.dataset import BRISCDataset
from cv.preprocessing.transform import MRIPreprocessor
from cv.preprocessing.augment import SynchronizedAugmentor

class TestDataPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.abspath("cv/data/brisc2025")

    def test_preprocessor_image_and_mask(self):
        pre = MRIPreprocessor(target_size=(256, 256), in_channels=1)
        dummy_img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        dummy_mask = np.zeros((512, 512), dtype=np.uint8)
        dummy_mask[100:200, 100:200] = 255

        t_img = pre.preprocess_image(dummy_img)
        t_mask = pre.preprocess_mask(dummy_mask)

        self.assertEqual(t_img.shape, (1, 256, 256))
        self.assertEqual(t_mask.shape, (1, 256, 256))
        self.assertTrue(t_img.min() >= 0.0 and t_img.max() <= 1.0)
        self.assertTrue(torch.all((t_mask == 0.0) | (t_mask == 1.0)))

    def test_synchronized_augmentor(self):
        aug = SynchronizedAugmentor(is_training=True)
        img = torch.rand(1, 128, 128)
        mask = torch.zeros(1, 128, 128)
        mask[0, 30:70, 30:70] = 1.0

        aug_img, aug_mask = aug(img, mask)
        self.assertEqual(aug_img.shape, img.shape)
        self.assertEqual(aug_mask.shape, mask.shape)
        # Verify mask remains strictly binary float {0.0, 1.0}
        unique_vals = torch.unique(aug_mask)
        for v in unique_vals:
            self.assertIn(v.item(), [0.0, 1.0])

    def test_dataset_loading(self):
        if not os.path.exists(self.data_dir):
            self.skipTest("BRISC dataset directory not found.")
        ds = BRISCDataset(data_dir=self.data_dir, split="train", max_samples=10)
        self.assertGreater(len(ds), 0)
        sample = ds[0]
        self.assertIn("image", sample)
        self.assertIn("mask", sample)
        self.assertEqual(sample["image"].shape, (1, 256, 256))
        self.assertEqual(sample["mask"].shape, (1, 256, 256))

if __name__ == '__main__':
    unittest.main()
