"""
This is a file responsible for two things
- segmentation: one image + one mask
- change detection: image 1 + image 2 + one change mask
We do this so the rest of the system can work with comfortably paired data
"""

from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path
import random

LOGGER = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True, slots=True)
class SegmentationSample:
    sample_id: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True, slots=True)
class ChangeDetectionSample:
    sample_id: str
    image1_path: Path
    image2_path: Path
    mask_path: Path


# Make sure dir exists
def _validate_directory(directory: str | Path) -> Path:

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a directory but got: {directory}")

    return directory


# Check whether the data we have is supported by the system
def _list_supported_files(directory: str | Path) -> list[Path]:

    directory = _validate_directory(directory)

    return sorted(
        file_path
        for file_path in directory.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES
    )


# Split one list into train and validation (if it was not done)
def _split_list(items: list, val_ratio: float, seed: int) -> tuple[list, list]:

    items = list(items)

    # We shuffle images so that model does not learn the order
    random_generator = random.Random(seed)
    random_generator.shuffle(items)

    # Keep at least one image for validation
    if len(items) <= 1:
        return items, []

    # Split dataset into 20% for val and 80 for training
    val_count = max(1, int(len(items) * val_ratio))
    val_items = items[:val_count]
    train_items = items[val_count:]

    # If the dataset is tiny, make sure training does not become empty
    if not train_items:
        train_items = val_items[:1]
        val_items = val_items[1:]

    return train_items, val_items


class SegmentationIngestor:
    """
    Generic ingestor for segmentation datasets
    This ingestor works for both:
    - road detection
    - land vs water segmentation
    """

    def __init__(
        self,
        train_images_dir: str | Path,
        train_masks_dir: str | Path,
        val_images_dir: str | Path | None = None,
        val_masks_dir: str | Path | None = None,
        split_seed: int = 42,
    ):
        self.train_images_dir = Path(train_images_dir)
        self.train_masks_dir = Path(train_masks_dir)
        self.val_images_dir = (
            Path(val_images_dir) if val_images_dir is not None else None
        )
        self.val_masks_dir = Path(val_masks_dir) if val_masks_dir is not None else None
        self.split_seed = split_seed

    # Pair image files with mask files by matching their stem
    def _pair_directories(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
    ) -> list[SegmentationSample]:

        image_files = _list_supported_files(images_dir)
        mask_files = _list_supported_files(masks_dir)

        mask_map = {mask_path.stem: mask_path for mask_path in mask_files}

        samples: list[SegmentationSample] = []
        missing_masks: list[str] = []

        for image_path in image_files:
            matching_mask = mask_map.get(image_path.stem)

            if matching_mask is None:
                missing_masks.append(image_path.name)
                continue

            if image_path.stat().st_size == 0 or matching_mask.stat().st_size == 0:
                LOGGER.warning(
                    "Skipping empty segmentation sample: %s", image_path.name
                )
                continue

            samples.append(
                SegmentationSample(
                    sample_id=image_path.stem,
                    image_path=image_path,
                    mask_path=matching_mask,
                )
            )

        if missing_masks:
            LOGGER.warning(
                "Skipped %d segmentation images without matching masks in %s.",
                len(missing_masks),
                images_dir,
            )

        return samples

    def build_splits(
        self, val_ratio: float = 0.2
    ) -> tuple[list[SegmentationSample], list[SegmentationSample]]:
        """
        Return train and validation samples.
        If explicit validation folders exist, we use them.
        Otherwise we create a deterministic split from the training folder.
        """

        train_samples = self._pair_directories(
            self.train_images_dir, self.train_masks_dir
        )

        if self.val_images_dir is not None and self.val_masks_dir is not None:
            val_samples = self._pair_directories(
                self.val_images_dir, self.val_masks_dir
            )
            return train_samples, val_samples

        return _split_list(train_samples, val_ratio=val_ratio, seed=self.split_seed)


# Generic ingestor for paired-image change-detection datasets
class ChangeDetectionIngestor:

    def __init__(
        self,
        train_image1_dir: str | Path,
        train_image2_dir: str | Path,
        train_masks_dir: str | Path,
        val_image1_dir: str | Path | None = None,
        val_image2_dir: str | Path | None = None,
        val_masks_dir: str | Path | None = None,
        split_seed: int = 42,
    ):
        self.train_image1_dir = Path(train_image1_dir)
        self.train_image2_dir = Path(train_image2_dir)
        self.train_masks_dir = Path(train_masks_dir)
        self.val_image1_dir = (
            Path(val_image1_dir) if val_image1_dir is not None else None
        )
        self.val_image2_dir = (
            Path(val_image2_dir) if val_image2_dir is not None else None
        )
        self.val_masks_dir = Path(val_masks_dir) if val_masks_dir is not None else None
        self.split_seed = split_seed

    # Pair time-1 images, time-2 images and masks by matching stems
    def _pair_directories(
        self,
        image1_dir: str | Path,
        image2_dir: str | Path,
        masks_dir: str | Path,
    ) -> list[ChangeDetectionSample]:

        image1_files = _list_supported_files(image1_dir)
        image2_map = {
            file_path.stem: file_path for file_path in _list_supported_files(image2_dir)
        }
        mask_map = {
            file_path.stem: file_path for file_path in _list_supported_files(masks_dir)
        }

        samples: list[ChangeDetectionSample] = []
        missing_pairs: list[str] = []

        for image1_path in image1_files:
            image2_path = image2_map.get(image1_path.stem)
            mask_path = mask_map.get(image1_path.stem)

            if image2_path is None or mask_path is None:
                missing_pairs.append(image1_path.name)
                continue

            if (
                image1_path.stat().st_size == 0
                or image2_path.stat().st_size == 0
                or mask_path.stat().st_size == 0
            ):
                LOGGER.warning(
                    "Skipping empty change-detection sample: %s", image1_path.name
                )
                continue

            samples.append(
                ChangeDetectionSample(
                    sample_id=image1_path.stem,
                    image1_path=image1_path,
                    image2_path=image2_path,
                    mask_path=mask_path,
                )
            )

        if missing_pairs:
            LOGGER.warning(
                "Skipped %d change-detection images with incomplete pairs in %s.",
                len(missing_pairs),
                image1_dir,
            )

        return samples

    # Make val and train triplets if was not done manually
    def build_splits(
        self,
        val_ratio: float = 0.2,
    ) -> tuple[list[ChangeDetectionSample], list[ChangeDetectionSample]]:

        train_samples = self._pair_directories(
            self.train_image1_dir,
            self.train_image2_dir,
            self.train_masks_dir,
        )

        if (
            self.val_image1_dir is not None
            and self.val_image2_dir is not None
            and self.val_masks_dir is not None
        ):
            val_samples = self._pair_directories(
                self.val_image1_dir,
                self.val_image2_dir,
                self.val_masks_dir,
            )
            return train_samples, val_samples

        return _split_list(train_samples, val_ratio=val_ratio, seed=self.split_seed)
