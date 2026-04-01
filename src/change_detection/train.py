"""Train a change-detection model."""

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
from data.pipeline.train import build_dataloaders, choose_device, fit_model, set_seed
from data.pipeline.transform import DataTransformer
from src.shared.models import build_binary_loss, build_change_detection_model, build_optimizer
from src.shared.visualize import save_change_detection_prediction


CHANGE_DATASETS = {
    "changes": {
        "train_image1": PROJECT_ROOT / "dataset" / "changes" / "image1",
        "train_image2": PROJECT_ROOT / "dataset" / "changes" / "image2",
        "train_masks": PROJECT_ROOT / "dataset" / "changes" / "mask",
        "val_image1": None,
        "val_image2": None,
        "val_masks": None,
        "positive_class_weight": None,
        "task_label": "change_detection",
    }
}


def build_change_datasets(
    dataset_name: str,
    image_size: tuple[int, int],
    use_augmentation: bool,
    val_ratio: float,
    split_seed: int,
) -> tuple[ChangeDetectionDataset, ChangeDetectionDataset]:
    """Create train and validation datasets for change detection."""

    if dataset_name not in CHANGE_DATASETS:
        supported = ", ".join(sorted(CHANGE_DATASETS))
        raise KeyError(f"Unknown change-detection dataset '{dataset_name}'. Supported: {supported}")

    dataset_config = CHANGE_DATASETS[dataset_name]

    ingestor = ChangeDetectionIngestor(
        train_image1_dir=dataset_config["train_image1"],
        train_image2_dir=dataset_config["train_image2"],
        train_masks_dir=dataset_config["train_masks"],
        val_image1_dir=dataset_config["val_image1"],
        val_image2_dir=dataset_config["val_image2"],
        val_masks_dir=dataset_config["val_masks"],
        split_seed=split_seed,
    )
    train_samples, val_samples = ingestor.build_splits(val_ratio=val_ratio)

    train_transformer = DataTransformer(
        resized_images=image_size,
        use_augmentation=use_augmentation,
    )
    val_transformer = DataTransformer(
        resized_images=image_size,
        use_augmentation=False,
    )

    train_dataset = ChangeDetectionDataset(train_samples, train_transformer)
    val_dataset = ChangeDetectionDataset(val_samples, val_transformer)
    return train_dataset, val_dataset


def run_change_detection_training(
    dataset_name: str = "changes",
    epochs: int = 8,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    image_size: tuple[int, int] = (256, 256),
    num_workers: int = 0,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    use_augmentation: bool = False,
) -> dict:
    """Full training flow for change detection."""

    set_seed(split_seed)

    train_dataset, val_dataset = build_change_datasets(
        dataset_name=dataset_name,
        image_size=image_size,
        use_augmentation=use_augmentation,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )

    train_loader, val_loader = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    device = choose_device(prefer_cuda=True)
    model = build_change_detection_model(encoder_name="resnet18", use_pretrained=False).to(device)

    positive_class_weight = CHANGE_DATASETS[dataset_name]["positive_class_weight"]
    criterion = build_binary_loss(device=device, positive_class_weight=positive_class_weight)
    optimizer = build_optimizer(model=model, learning_rate=learning_rate)

    output_root = PROJECT_ROOT / "outputs"
    checkpoint_path = output_root / "checkpoints" / f"{dataset_name}_change_best.pth"
    history_path = output_root / "reports" / f"{dataset_name}_change_history.json"
    prediction_dir = output_root / "predictions" / dataset_name
    prediction_dir.mkdir(parents=True, exist_ok=True)

    print(f"Task: {CHANGE_DATASETS[dataset_name]['task_label']}")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        task_type="change_detection",
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        threshold=0.5,
    )

    # We reload the best saved weights before generating preview panels.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    preview_count = min(3, len(val_dataset))
    for index in range(preview_count):
        save_change_detection_prediction(
            model=model,
            dataset=val_dataset,
            index=index,
            device=device,
            filename=prediction_dir / f"{dataset_name}_preview_{index}.png",
        )

    return {
        "model": model,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "history": history,
        "checkpoint_path": checkpoint_path,
        "prediction_dir": prediction_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a change-detection model.")
    parser.add_argument("--dataset", choices=sorted(CHANGE_DATASETS), default="changes")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--augment", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_change_detection_training(
        dataset_name=arguments.dataset,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        image_size=(arguments.image_size, arguments.image_size),
        num_workers=arguments.num_workers,
        val_ratio=arguments.val_ratio,
        use_augmentation=arguments.augment,
    )
