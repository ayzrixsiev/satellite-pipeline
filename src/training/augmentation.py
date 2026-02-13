# src/training/augmentations.py
from typing import Any
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Data to train the model
def get_train_transforms(img_size: int = 224) -> Any:
    """
    Returns augmentations for training.
    Args:
        img_size: Resize to this (e.g., 224 for ResNet/MobileNet input).
    Why: Augmentations simulate real factory variations (e.g., rotated pizzas on belts, varying lights).
         Improves model robustness without more data—key for edge deployment where conditions vary.
    How: Use Albumentations for speed/flexibility (CPU-friendly)
    Interview: "I use random flips/brightness to mimic conveyor belt orientations and lighting, reducing false positives in QC."
    """
    # Using Albumentations
    return A.Compose(
        [
            A.Resize(
                height=img_size, width=img_size
            ),  # Standardize size for model input.
            A.RandomRotate90(p=0.5),  # Simulate random orientations.
            A.Flip(p=0.5),  # Horizontal/vertical flips.
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),  # Lighting variations.
            A.GaussianBlur(p=0.3),  # Simulate camera blur for defective edges.
            A.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),  # ImageNet stats for pretrained models.
            ToTensorV2(),  # To PyTorch tensor.
        ]
    )


# Clean data to validate performance of the model
def get_val_transforms(img_size: int = 224) -> Any:
    """
    Returns lighter transforms for validation (no randomness).
    Why: Val should reflect real inference—clean, deterministic—to accurately measure performance.
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
