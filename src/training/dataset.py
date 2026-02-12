from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset


class QualityDataset(Dataset):
    """
    Custom dataset for quality classification.

    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples = []
        self.class_to_idx = {"good": 0, "bad": 1}

        # Collect all image paths + labels
        for class_name, label in self.class_to_idx.items():
            class_folder = self.root_dir / class_name

            for img_path in class_folder.glob("*"):
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label
