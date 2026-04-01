"""Visualization helpers.

These functions save side-by-side images that are easy to inspect visually.
That matters a lot in remote sensing because metrics alone do not tell the whole
story, especially for roads and change detection.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def _tensor_to_rgb_image(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized image tensor back into an RGB uint8 image."""

    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def _tensor_to_mask(mask_tensor: torch.Tensor) -> np.ndarray:
    """Convert a mask tensor into a visible grayscale image."""

    mask = mask_tensor.detach().cpu().numpy().squeeze()
    return (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)


def _overlay_prediction(image_rgb: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
    """Blend a predicted mask on top of the source image."""

    overlay = image_rgb.copy()
    colored_mask = np.zeros_like(image_rgb)
    colored_mask[..., 1] = mask_gray

    # OpenCV works more reliably when we blend full images instead of advanced
    # indexed slices, so we create the full blended image first and then copy
    # only the active mask pixels into the overlay.
    blended = cv2.addWeighted(image_rgb, 0.65, colored_mask, 0.35, 0.0)
    active_pixels = mask_gray > 0
    overlay[active_pixels] = blended[active_pixels]

    return overlay


def _title_panel(image_rgb: np.ndarray, title: str) -> np.ndarray:
    """Write a readable label at the top of one panel."""

    panel = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(
        panel,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def save_segmentation_prediction(
    model: torch.nn.Module,
    dataset,
    index: int,
    device: torch.device,
    filename: str | Path = "result.png",
    threshold: float = 0.5,
) -> Path:
    """Save one segmentation comparison panel."""

    sample = dataset[index]
    image = sample["image"]
    target_mask = sample["mask"]

    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        probabilities = torch.sigmoid(logits).squeeze(0).cpu()
        prediction = (probabilities > threshold).float()

    image_rgb = _tensor_to_rgb_image(image)
    target_mask_gray = _tensor_to_mask(target_mask)
    prediction_mask_gray = _tensor_to_mask(prediction)
    overlay_rgb = _overlay_prediction(image_rgb, prediction_mask_gray)

    panel = np.hstack(
        [
            _title_panel(image_rgb, "Image"),
            _title_panel(cv2.cvtColor(target_mask_gray, cv2.COLOR_GRAY2RGB), "Ground Truth"),
            _title_panel(cv2.cvtColor(prediction_mask_gray, cv2.COLOR_GRAY2RGB), "Prediction"),
            _title_panel(overlay_rgb, "Overlay"),
        ]
    )

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(filename), panel)
    return filename


def save_change_detection_prediction(
    model: torch.nn.Module,
    dataset,
    index: int,
    device: torch.device,
    filename: str | Path = "change_result.png",
    threshold: float = 0.5,
) -> Path:
    """Save one change-detection comparison panel."""

    sample = dataset[index]
    image1 = sample["image1"]
    image2 = sample["image2"]
    target_mask = sample["mask"]

    model.eval()
    with torch.no_grad():
        stacked_input = torch.cat([image1.unsqueeze(0), image2.unsqueeze(0)], dim=1).to(device)
        logits = model(stacked_input)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu()
        prediction = (probabilities > threshold).float()

    image1_rgb = _tensor_to_rgb_image(image1)
    image2_rgb = _tensor_to_rgb_image(image2)
    target_mask_gray = _tensor_to_mask(target_mask)
    prediction_mask_gray = _tensor_to_mask(prediction)

    panel = np.hstack(
        [
            _title_panel(image1_rgb, "Time 1"),
            _title_panel(image2_rgb, "Time 2"),
            _title_panel(cv2.cvtColor(target_mask_gray, cv2.COLOR_GRAY2RGB), "Ground Truth"),
            _title_panel(cv2.cvtColor(prediction_mask_gray, cv2.COLOR_GRAY2RGB), "Prediction"),
        ]
    )

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(filename), panel)
    return filename


# Backward-compatible alias for older code that still imports `save_prediction`.
save_prediction = save_segmentation_prediction
