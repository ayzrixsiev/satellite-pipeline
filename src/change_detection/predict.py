"""Run inference for change detection and save visual panels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pipeline.dataset import ChangeDetectionDataset
from data.pipeline.ingest import ChangeDetectionIngestor
from data.pipeline.transform import DataTransformer
from src.change_detection.train import CHANGE_DATASETS
from src.shared.models import build_change_detection_model
from src.shared.visualize import save_change_detection_prediction


def load_validation_dataset(
    dataset_name: str,
    image_size: tuple[int, int] = (256, 256),
    val_ratio: float = 0.2,
) -> ChangeDetectionDataset:
    """Build the validation split for demo inference."""

    dataset_config = CHANGE_DATASETS[dataset_name]

    ingestor = ChangeDetectionIngestor(
        train_image1_dir=dataset_config["train_image1"],
        train_image2_dir=dataset_config["train_image2"],
        train_masks_dir=dataset_config["train_masks"],
        val_image1_dir=dataset_config["val_image1"],
        val_image2_dir=dataset_config["val_image2"],
        val_masks_dir=dataset_config["val_masks"],
        split_seed=42,
    )
    _, val_samples = ingestor.build_splits(val_ratio=val_ratio)

    transformer = DataTransformer(resized_images=image_size, use_augmentation=False)
    return ChangeDetectionDataset(val_samples, transformer)


def run_change_detection_prediction(
    dataset_name: str = "changes",
    checkpoint_path: str | Path | None = None,
    image_size: tuple[int, int] = (256, 256),
    output_count: int = 3,
) -> Path:
    """Load a trained model and save prediction panels."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_change_detection_model(encoder_name="resnet18", use_pretrained=False).to(device)

    if checkpoint_path is None:
        checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / f"{dataset_name}_change_best.pth"

    state_dict = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    val_dataset = load_validation_dataset(
        dataset_name=dataset_name,
        image_size=image_size,
    )

    output_dir = PROJECT_ROOT / "outputs" / "predictions" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(min(output_count, len(val_dataset))):
        save_change_detection_prediction(
            model=model,
            dataset=val_dataset,
            index=index,
            device=device,
            filename=output_dir / f"{dataset_name}_prediction_{index}.png",
        )

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run change-detection inference.")
    parser.add_argument("--dataset", choices=sorted(CHANGE_DATASETS), default="changes")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--count", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_change_detection_prediction(
        dataset_name=arguments.dataset,
        checkpoint_path=arguments.checkpoint,
        image_size=(arguments.image_size, arguments.image_size),
        output_count=arguments.count,
    )

