"""
Data preprocessing. We turn images into tensors.
- read image from my local machine
- convert color format (RGB)
- resize (to feed ai)
- normalize
- binarize masks
- apply light augmentation
"""

from __future__ import annotations

import os
from pathlib import Path
import random

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
import torch


class DataTransformer:

    def __init__(
        self,
        resized_images: tuple[int, int] = (512, 512),
        use_augmentation: bool = False,
    ):
        self.resized_images = resized_images
        self.use_augmentation = use_augmentation

    # Make sure an image is rgb, for ml model
    def read_rgb_image(self, image_path: str | Path) -> np.ndarray:

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def read_binary_mask(self, mask_path: str | Path) -> np.ndarray:

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        # If RGB we make it grayscale
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Make edge more visible by converting feature from 1-white into 255-white
        if mask.max() <= 1:
            mask = (mask > 0).astype(np.uint8) * 255
        else:
            mask = mask.astype(np.uint8)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return mask

    # We resize all images and masks to make sure we can feed it to the AI
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.resized_images, interpolation=cv2.INTER_LINEAR)

    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        return cv2.resize(mask, self.resized_images, interpolation=cv2.INTER_NEAREST)

    def _maybe_flip_segmentation(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

        if self.use_augmentation and random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        if self.use_augmentation and random.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        return image, mask

    def _maybe_flip_change(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        if self.use_augmentation and random.random() < 0.5:
            image1 = np.flip(image1, axis=1).copy()
            image2 = np.flip(image2, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        if self.use_augmentation and random.random() < 0.5:
            image1 = np.flip(image1, axis=0).copy()
            image2 = np.flip(image2, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        return image1, image2, mask

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1)).copy()
        return torch.from_numpy(image)

    def _mask_to_tensor(self, mask: np.ndarray) -> torch.Tensor:

        mask = (mask.astype(np.float32) / 255.0)[None, ...].copy()
        return torch.from_numpy(mask)

    # Orchestration - use all the functions for segmentation
    def process_segmentation_sample(
        self,
        image_path: str | Path,
        mask_path: str | Path,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        image = self.read_rgb_image(image_path)
        mask = self.read_binary_mask(mask_path)

        image = self._resize_image(image)
        mask = self._resize_mask(mask)
        image, mask = self._maybe_flip_segmentation(image, mask)

        return self._image_to_tensor(image), self._mask_to_tensor(mask)

    # Orchestration - use all the functions change detection
    def process_change_sample(
        self,
        image1_path: str | Path,
        image2_path: str | Path,
        mask_path: str | Path,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        image1 = self.read_rgb_image(image1_path)
        image2 = self.read_rgb_image(image2_path)
        mask = self.read_binary_mask(mask_path)

        image1 = self._resize_image(image1)
        image2 = self._resize_image(image2)
        mask = self._resize_mask(mask)
        image1, image2, mask = self._maybe_flip_change(image1, image2, mask)

        return (
            self._image_to_tensor(image1),
            self._image_to_tensor(image2),
            self._mask_to_tensor(mask),
        )
