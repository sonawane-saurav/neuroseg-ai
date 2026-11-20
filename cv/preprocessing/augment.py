"""
Synchronized Data Augmentation Pipeline for MRI Images and Segmentation Masks.
Ensures identical geometric transformations for images and masks,
applies nearest-neighbor interpolation on masks, and restricts photometric/intensity
augmentations strictly to the MRI image.
"""

from typing import Tuple
import random
import numpy as np
import torch
import cv2

class SynchronizedAugmentor:
    def __init__(
        self,
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        p_rotate: float = 0.5,
        max_angle: float = 15.0,
        p_scale: float = 0.3,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        p_noise: float = 0.3,
        noise_std: float = 0.03,
        p_contrast: float = 0.3,
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        is_training: bool = True
    ):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rotate = p_rotate
        self.max_angle = max_angle
        self.p_scale = p_scale
        self.scale_range = scale_range
        self.p_noise = p_noise
        self.noise_std = noise_std
        self.p_contrast = p_contrast
        self.contrast_range = contrast_range
        self.is_training = is_training

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: FloatTensor of shape (C, H, W) normalized to [0, 1].
            mask: FloatTensor of shape (1, H, W) with binary values {0.0, 1.0}.
        Returns:
            Augmented (image, mask) tensors with identical geometric alignment.
        """
        if not self.is_training:
            return image, mask

        img_np = image.numpy()  # (C, H, W)
        mask_np = mask.numpy()  # (1, H, W)
        _, h, w = img_np.shape

        # 1. Synchronized Horizontal Flip
        if random.random() < self.p_hflip:
            img_np = np.flip(img_np, axis=-1).copy()
            mask_np = np.flip(mask_np, axis=-1).copy()

        # 2. Synchronized Vertical Flip
        if random.random() < self.p_vflip:
            img_np = np.flip(img_np, axis=-2).copy()
            mask_np = np.flip(mask_np, axis=-2).copy()

        # 3. Synchronized Rotation and Scaling (Affine)
        do_rotate = random.random() < self.p_rotate
        do_scale = random.random() < self.p_scale
        if do_rotate or do_scale:
            angle = random.uniform(-self.max_angle, self.max_angle) if do_rotate else 0.0
            scale = random.uniform(self.scale_range[0], self.scale_range[1]) if do_scale else 1.0
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, scale)

            # Apply to image channels with bilinear
            img_transformed = []
            for c in range(img_np.shape[0]):
                ch = cv2.warpAffine(img_np[c], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                img_transformed.append(ch)
            img_np = np.stack(img_transformed, axis=0)

            # Apply to mask with STRICT NEAREST NEIGHBOR
            mask_ch = cv2.warpAffine(mask_np[0], M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask_np = np.expand_dims(mask_ch, axis=0)

        # 4. Photometric Augmentations (IMAGE ONLY - NEVER MASK)
        # Additive Gaussian Noise
        if random.random() < self.p_noise:
            noise = np.random.normal(0, self.noise_std, img_np.shape).astype(np.float32)
            img_np = np.clip(img_np + noise, 0.0, 1.0)

        # Contrast & Brightness Adjustment
        if random.random() < self.p_contrast:
            factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
            img_np = np.clip(img_np * factor, 0.0, 1.0)

        # Ensure mask remains strictly binary float {0.0, 1.0}
        mask_np = (mask_np > 0.5).astype(np.float32)

        return torch.from_numpy(img_np).float(), torch.from_numpy(mask_np).float()
