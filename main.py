"""Small project entrypoint.

This file lets you launch the two task families from one place while keeping
the real logic inside `src/segmentation` and `src/change_detection`.
"""

from __future__ import annotations

import argparse

from src.change_detection.predict import run_change_detection_prediction
from src.change_detection.train import run_change_detection_training
from src.segmentation.predict import run_segmentation_prediction
from src.segmentation.train import run_segmentation_training


def parse_args() -> argparse.Namespace:
    """Parse a tiny CLI for the whole project."""

    parser = argparse.ArgumentParser(description="GeoSynth multi-task runner.")
    parser.add_argument("task_family", choices=["segmentation", "change_detection"])
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--augment", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Route the command to the correct task family."""

    arguments = parse_args()

    if arguments.task_family == "segmentation" and arguments.mode == "train":
        run_segmentation_training(
            dataset_name=arguments.dataset or "roads",
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            image_size=(arguments.image_size, arguments.image_size),
            num_workers=arguments.num_workers,
            val_ratio=arguments.val_ratio,
            use_augmentation=arguments.augment,
        )
        return

    if arguments.task_family == "segmentation" and arguments.mode == "predict":
        run_segmentation_prediction(
            dataset_name=arguments.dataset or "roads",
            checkpoint_path=arguments.checkpoint,
            image_size=(arguments.image_size, arguments.image_size),
            output_count=arguments.count,
        )
        return

    if arguments.task_family == "change_detection" and arguments.mode == "train":
        run_change_detection_training(
            dataset_name=arguments.dataset or "changes",
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            image_size=(arguments.image_size, arguments.image_size),
            num_workers=arguments.num_workers,
            val_ratio=arguments.val_ratio,
            use_augmentation=arguments.augment,
        )
        return

    if arguments.task_family == "change_detection" and arguments.mode == "predict":
        run_change_detection_prediction(
            dataset_name=arguments.dataset or "changes",
            checkpoint_path=arguments.checkpoint,
            image_size=(arguments.image_size, arguments.image_size),
            output_count=arguments.count,
        )
        return


if __name__ == "__main__":
    main()

