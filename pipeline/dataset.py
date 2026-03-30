import torch
import numpy as np
from torch.utils.data import Dataset


class GeoSynthDataset(Dataset):
    def __init__(self, data_pairs, transformer):
        self.data_pairs = data_pairs
        self.transformer = transformer

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, index):
        # Get pairs
        img_path, label_path = self.data_pairs[index]

        # Process each of them
        image = self.transformer.process_images(img_path)
        label = self.transformer.process_labels(label_path)

        # OpenCV loaded images as (Height, Width, Channels), but Pytorch needs (Channels, Height, Width)
        image = np.transpose(image, (2, 0, 1))

        # Convert to Pytorch tensors, which is basically arrays but with a passport to go my GPU
        return torch.from_numpy(image), torch.from_numpy(label)
