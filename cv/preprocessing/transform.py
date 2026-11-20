"""
MRI Image and Mask Preprocessing Module
Implements clinically consistent intensity normalization, resizing,
and format conversions for training, validation, and real-time inference.
"""

from typing import Tuple, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

class MRIPreprocessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        in_channels: int = 1,
        normalize_mode: str = "minmax",
        mask_threshold: float = 127.0
    ):
        """
        Args:
            target_size: (height, width) for model input. Default: (256, 256).
            in_channels: 1 for single-channel grayscale, 3 for RGB.
            normalize_mode: 'minmax' [0, 1] or 'zscore' (mean=0, std=1).
            mask_threshold: Intensity threshold for raw PNG masks (default 127.0).
        """
        self.target_size = target_size
        self.in_channels = in_channels
        self.normalize_mode = normalize_mode
        self.mask_threshold = mask_threshold

    def preprocess_image(self, image: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
        """
        Preprocesses an MRI image into a PyTorch tensor of shape (C, H, W) in float32.
        """
        if isinstance(image, str):
            image = Image.open(image)
            
        if isinstance(image, Image.Image):
            if self.in_channels == 1:
                image = image.convert('L')
            else:
                image = image.convert('RGB')
            arr = np.array(image, dtype=np.float32)
        else:
            arr = image.astype(np.float32)
            if self.in_channels == 1 and arr.ndim == 3:
                arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Resize image using bilinear interpolation
        if (arr.shape[0], arr.shape[1]) != self.target_size:
            arr = cv2.resize(arr, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)

        # Intensity Normalization
        if self.normalize_mode == "minmax":
            img_min, img_max = arr.min(), arr.max()
            if img_max > img_min:
                arr = (arr - img_min) / (img_max - img_min)
            else:
                arr = np.zeros_like(arr)
        elif self.normalize_mode == "zscore":
            mean, std = arr.mean(), arr.std()
            arr = (arr - mean) / (std + 1e-7)

        # Format to (C, H, W)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (2, 0, 1))

        return torch.from_numpy(arr).float()

    def preprocess_mask(self, mask: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
        """
        Preprocesses a segmentation mask into a binary PyTorch tensor of shape (1, H, W) in float32.
        CRITICAL: Uses NEAREST-NEIGHBOR interpolation to strictly avoid interpolating false class labels.
        """
        if isinstance(mask, str):
            mask = Image.open(mask)

        if isinstance(mask, Image.Image):
            mask = mask.convert('L')
            arr = np.array(mask)
        else:
            arr = mask.copy()
            if arr.ndim == 3:
                arr = arr[:, :, 0]

        # Resize mask strictly using INTER_NEAREST
        if (arr.shape[0], arr.shape[1]) != self.target_size:
            arr = cv2.resize(arr, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)

        # Binarize with threshold
        bin_mask = (arr > self.mask_threshold).astype(np.float32)
        bin_mask = np.expand_dims(bin_mask, axis=0) # (1, H, W)

        return torch.from_numpy(bin_mask).float()

    def preprocess_pair(
        self,
        image: Union[np.ndarray, Image.Image, str],
        mask: Union[np.ndarray, Image.Image, str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method to preprocess paired MRI slice and ground truth mask."""
        return self.preprocess_image(image), self.preprocess_mask(mask)

    def denormalize_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Converts a preprocessed (C, H, W) tensor back to uint8 [0, 255] numpy image for visualization.
        """
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                arr = np.transpose(arr, (1, 2, 0))

        if self.normalize_mode == "minmax":
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                arr = ((arr - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)
        return arr
