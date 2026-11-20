"""
Unit Tests for Model Architectures: U-Net, MultiTaskUNet, AttentionUNet.
"""

import unittest
import torch

from cv.models.unet import UNet
from cv.models.multitask_unet import MultiTaskUNet
from cv.models.architectures.attention_unet import AttentionUNet

class TestModelArchitectures(unittest.TestCase):
    def test_baseline_unet_forward(self):
        model = UNet(in_channels=1, out_classes=1, features=16, dropout_p=0.2)
        x = torch.randn(2, 1, 128, 128)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, 128, 128))

    def test_unet_mc_dropout_enable(self):
        model = UNet(in_channels=1, out_classes=1, features=16, dropout_p=0.2)
        model.eval()
        model.enable_mc_dropout()
        # Verify Dropout layers are in training mode while BatchNorm is eval
        has_dropout = False
        for m in model.modules():
            if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
                self.assertTrue(m.training)
                has_dropout = True
            elif isinstance(m, torch.nn.BatchNorm2d):
                self.assertFalse(m.training)
        self.assertTrue(has_dropout)

    def test_multitask_unet_forward(self):
        model = MultiTaskUNet(in_channels=1, out_classes=1, num_classes=4, features=16)
        x = torch.randn(2, 1, 128, 128)
        seg_logits, cls_logits = model(x)
        self.assertEqual(seg_logits.shape, (2, 1, 128, 128))
        self.assertEqual(cls_logits.shape, (2, 4))

    def test_attention_unet_forward(self):
        model = AttentionUNet(in_channels=1, out_classes=1, features=16)
        x = torch.randn(2, 1, 128, 128)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, 128, 128))

if __name__ == '__main__':
    unittest.main()
