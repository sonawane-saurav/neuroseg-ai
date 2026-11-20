"""
Unit Tests for Losses and Evaluation Metrics.
"""

import unittest
import torch
import numpy as np

from cv.training.losses import DiceLoss, BCEDiceLoss, MultiTaskLoss
from cv.training.metrics import calculate_segmentation_metrics, calculate_classification_metrics

class TestLossesAndMetrics(unittest.TestCase):
    def test_bce_dice_loss(self):
        loss_fn = BCEDiceLoss()
        logits = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(logits, targets)
        self.assertGreater(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss))

    def test_multitask_loss(self):
        loss_fn = MultiTaskLoss(lambda_cls=0.5)
        seg_logits = torch.randn(2, 1, 64, 64)
        seg_targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        cls_logits = torch.randn(2, 4)
        cls_targets = torch.tensor([1, 2], dtype=torch.long)

        tot, s_l, c_l = loss_fn(seg_logits, seg_targets, cls_logits, cls_targets)
        self.assertGreater(tot.item(), 0.0)
        self.assertAlmostEqual(tot.item(), s_l.item() + 0.5 * c_l.item(), places=4)

    def test_segmentation_metrics_perfect(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 20:60] = 1
        m = calculate_segmentation_metrics(mask, mask)
        self.assertAlmostEqual(m["dice"], 1.0, places=4)
        self.assertAlmostEqual(m["iou"], 1.0, places=4)
        self.assertAlmostEqual(m["precision"], 1.0, places=4)
        self.assertAlmostEqual(m["recall"], 1.0, places=4)

    def test_classification_metrics(self):
        preds = np.array([0, 1, 2, 3])
        targets = np.array([0, 1, 2, 3])
        m = calculate_classification_metrics(preds, targets, num_classes=4)
        self.assertAlmostEqual(m["accuracy"], 1.0, places=4)

if __name__ == '__main__':
    unittest.main()
