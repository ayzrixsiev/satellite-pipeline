"""PyTorch Dataset classes.

The ingestors only pair file paths.
The transformer only knows how to read and preprocess files.
The Dataset classes connect those two ideas to PyTorch.
"""

from __future__ import annotations

from torch.utils.data import Dataset

from data.pipeline.ingest import ChangeDetectionSample, SegmentationSample
from data.pipeline.transform import DataTransformer


class SegmentationDataset(Dataset):
    """Dataset for one-image segmentation tasks such as roads or water-land."""

    def __init__(self, samples: list[SegmentationSample], transformer: DataTransformer):
        self.samples = samples
        self.transformer = transformer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        image_tensor, mask_tensor = self.transformer.process_segmentation_sample(
            image_path=sample.image_path,
            mask_path=sample.mask_path,
        )

        # Returning a dictionary is slightly more verbose than a tuple, but it is
        # much easier to extend later when you want metadata or multiple inputs.
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
        }


class ChangeDetectionDataset(Dataset):
    """Dataset for tasks where one sample has two timestamps and one mask."""

    def __init__(self, samples: list[ChangeDetectionSample], transformer: DataTransformer):
        self.samples = samples
        self.transformer = transformer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        image1_tensor, image2_tensor, mask_tensor = self.transformer.process_change_sample(
            image1_path=sample.image1_path,
            image2_path=sample.image2_path,
            mask_path=sample.mask_path,
        )

        return {
            "image1": image1_tensor,
            "image2": image2_tensor,
            "mask": mask_tensor,
            "sample_id": sample.sample_id,
            "image1_path": str(sample.image1_path),
            "image2_path": str(sample.image2_path),
            "mask_path": str(sample.mask_path),
        }


# This alias keeps the old beginner-friendly name available so your older code
# does not completely break while you transition to the new structure.
GeoSynthDataset = SegmentationDataset

