"""
Feature Two: Synthetic MRI Image Degradations for Robustness Testing.
Simulates clinical image quality degradation:
1. Gaussian Noise (thermal/scanner electronic noise)
2. Low Contrast (sub-optimal contrast agent uptake / sequence settings)
3. Motion Blur (patient movement artifacts)
4. Resolution Degradation (low-field / low-resolution scanner downsampling)
"""

from typing import Union
import numpy as np
import cv2

class MRIDegradationPipeline:
    SEVERITY_LEVELS = [1, 2, 3, 4, 5]

    @staticmethod
    def apply_gaussian_noise(image: np.ndarray, severity: int = 3) -> np.ndarray:
        """
        Adds zero-mean Gaussian noise to simulate thermal and sensor noise.
        Severity 1-5 maps to stddev in [0.03, 0.06, 0.10, 0.15, 0.22] for [0, 1] images
        or [8, 15, 25, 38, 55] for uint8 [0, 255] images.
        """
        is_float = np.issubdtype(image.dtype, np.floating)
        img = image.copy()
        
        if is_float:
            stds = {1: 0.03, 2: 0.06, 3: 0.10, 4: 0.15, 5: 0.22}
            std = stds.get(severity, 0.10)
            noise = np.random.normal(0, std, img.shape)
            degraded = np.clip(img + noise, 0.0, 1.0).astype(image.dtype)
        else:
            stds = {1: 8.0, 2: 15.0, 3: 25.0, 4: 38.0, 5: 55.0}
            std = stds.get(severity, 25.0)
            noise = np.random.normal(0, std, img.shape)
            degraded = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return degraded

    @staticmethod
    def apply_low_contrast(image: np.ndarray, severity: int = 3) -> np.ndarray:
        """
        Compresses dynamic range to simulate poor soft-tissue contrast.
        Formula: I' = alpha * (I - mid) + mid
        """
        is_float = np.issubdtype(image.dtype, np.floating)
        img = image.copy()
        mid = 0.5 if is_float else 128.0

        factors = {1: 0.85, 2: 0.70, 3: 0.50, 4: 0.35, 5: 0.20}
        alpha = factors.get(severity, 0.50)

        degraded = alpha * (img.astype(np.float32) - mid) + mid
        if is_float:
            return np.clip(degraded, 0.0, 1.0).astype(image.dtype)
        else:
            return np.clip(degraded, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_motion_blur(image: np.ndarray, severity: int = 3, angle: float = 45.0) -> np.ndarray:
        """
        Applies directional motion blur using a linear convolution kernel.
        Severity 1-5 maps to kernel sizes [3, 7, 11, 15, 21].
        """
        kernel_sizes = {1: 3, 2: 7, 3: 11, 4: 15, 5: 21}
        ksize = kernel_sizes.get(severity, 11)

        # Create directional motion blur kernel
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        mid = ksize // 2
        kernel[mid, :] = 1.0

        # Rotate kernel to desired angle
        M = cv2.getRotationMatrix2D((mid, mid), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel = kernel / kernel_sum

        if image.ndim == 2:
            return cv2.filter2D(image, -1, kernel)
        else:
            channels = [cv2.filter2D(image[:, :, c], -1, kernel) for c in range(image.shape[-1])]
            return np.stack(channels, axis=-1)

    @staticmethod
    def apply_resolution_degradation(image: np.ndarray, severity: int = 3) -> np.ndarray:
        """
        Downsamples the MRI slice to simulate low spatial resolution,
        then resamples back using bicubic interpolation.
        Severity 1-5 downsamples by factors [1.5x, 2x, 3x, 4x, 6x].
        """
        scale_factors = {1: 1.5, 2: 2.0, 3: 3.0, 4: 4.0, 5: 6.0}
        factor = scale_factors.get(severity, 3.0)

        h, w = image.shape[:2]
        down_w = max(16, int(w / factor))
        down_h = max(16, int(h / factor))

        # Downsample
        downsampled = cv2.resize(image, (down_w, down_h), interpolation=cv2.INTER_AREA)
        # Resample back to original resolution
        restored = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_CUBIC)
        
        if np.issubdtype(image.dtype, np.floating):
            return np.clip(restored, 0.0, 1.0).astype(image.dtype)
        else:
            return np.clip(restored, 0, 255).astype(np.uint8)

    @classmethod
    def apply_degradation(
        cls,
        image: np.ndarray,
        degradation_type: str,
        severity: int = 3
    ) -> np.ndarray:
        """Dispatcher for degradation types."""
        deg = degradation_type.lower()
        if deg in ("noise", "gaussian_noise"):
            return cls.apply_gaussian_noise(image, severity)
        elif deg in ("contrast", "low_contrast"):
            return cls.apply_low_contrast(image, severity)
        elif deg in ("motion_blur", "blur"):
            return cls.apply_motion_blur(image, severity)
        elif deg in ("resolution", "resolution_degradation", "downsample"):
            return cls.apply_resolution_degradation(image, severity)
        else:
            raise ValueError(f"Unknown degradation type: '{degradation_type}'. Supported: gaussian_noise, low_contrast, motion_blur, resolution_degradation")
