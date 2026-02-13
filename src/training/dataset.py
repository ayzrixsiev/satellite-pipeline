# src/training/dataset.py
import os
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import cv2  # ← added
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms  # if you still use torchvision transforms later


class QCDataset(Dataset):
    """
    Custom PyTorch Dataset for loading pizza images and labels.
    Why: In production ML, custom datasets allow efficient loading of large image sets without memory overload.
         This mimics factory data pipelines where images come from cameras and need preprocessing.
    How: Loads images from paths in a CSV (for metadata), applies transforms (passed in), and returns tensor + label.
    Interview tip: Explain "Datasets enable lazy loading and parallel workers in DataLoaders for faster training."
    """

    def __init__(self, csv_file: str, root_dir: str, transform=None):
        """
        Initializes the dataset.
        Args:
            csv_file: Path to CSV with columns 'image_path' (relative to root_dir) and 'label' (0=good, 1=defective).
            root_dir: Base directory for images.
            transform: PyTorch transforms (e.g., from torchvision or albumentations) for preprocessing/augmentation.
        Why: CSV allows metadata management (e.g., add timestamps for factory logs). Transform is flexible for train vs. val.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Data cleaning: Remove rows with missing images
        valid_rows = []
        for idx, row in self.data_frame.iterrows():
            img_path = os.path.join(self.root_dir, row["image_path"])
            if os.path.exists(img_path):
                valid_rows.append(row)
            else:
                print(f"Warning: Missing image {img_path}")
        self.data_frame = pd.DataFrame(valid_rows)

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load one sample using OpenCV → numpy → tensor.
        Important: OpenCV loads BGR → we convert to RGB.
        Returns: (image tensor, label).
        Why: Lazy loading—only load image when needed, efficient for large datasets on edge devices.
        """
        row = self.data_frame.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image_path"])

        # Load with OpenCV
        image = cv2.imread(img_path)  # returns numpy array (H, W, C=BGR)

        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Convert BGR → RGB (critical for pretrained models)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = row["label"]  # 0=good, 1=defective

        # If using transforms (Albumentations or torchvision)
        if self.transform:
            # Albumentations expects numpy array (HWC)
            transformed = self.transform(image=image)
            image = transformed["image"]  # Albumentations returns dict with 'image'

            # If transform already gives tensor → good
            # If not, convert manually below
        else:
            # Manual to tensor if no transform
            image = (
                torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            )  # HWC → CHW, [0,1]

        return image, label
