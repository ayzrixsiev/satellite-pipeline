"""
Shared training utilities
This file contains the reusable "engine" part of the project:
- choose device
- set seeds
- build dataloaders
- run one epoch
- fit a model for multiple epochs
"""

from __future__ import annotations
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.shared.metrics import summarize_binary_batch, summarize_epoch_metrics


# Make random seeds
def set_seed(seed: int = 42) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Pick the device
def choose_device(prefer_cuda: bool = True) -> torch.device:

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Wrap datasets into dataloaders
def build_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int = 4,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def _prepare_inputs_and_targets(
    batch: dict,
    task_type: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one DataLoader batch into model inputs and targets."""

    if task_type == "segmentation":
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        return images, masks

    if task_type == "change_detection":
        image1 = batch["image1"].to(device)
        image2 = batch["image2"].to(device)
        masks = batch["mask"].to(device)

        # Concatenating along the channel axis gives us a simple 6-channel input:
        # [R,G,B from time 1] + [R,G,B from time 2].
        images = torch.cat([image1, image2], dim=1)
        return images, masks

    raise ValueError(f"Unsupported task type: {task_type}")


def run_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    task_type: str,
    threshold: float = 0.5,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Run one full training or validation epoch.

    If `optimizer` is None, we run in evaluation mode.
    If `optimizer` exists, we run in training mode.
    """

    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    sample_count = 0
    true_positives = 0.0
    false_positives = 0.0
    false_negatives = 0.0

    with torch.set_grad_enabled(is_training):
        for batch in loader:
            inputs, targets = _prepare_inputs_and_targets(batch, task_type, device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            sample_count += batch_size

            batch_tp, batch_fp, batch_fn = summarize_binary_batch(
                logits=logits,
                targets=targets,
                threshold=threshold,
            )
            true_positives += batch_tp
            false_positives += batch_fp
            false_negatives += batch_fn

    return summarize_epoch_metrics(
        total_loss=total_loss,
        sample_count=sample_count,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def fit_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_type: str,
    epochs: int,
    checkpoint_path: str | Path,
    history_path: str | Path | None = None,
    threshold: float = 0.5,
) -> list[dict[str, float]]:
    """Train for multiple epochs and save the best checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if history_path is not None:
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            task_type=task_type,
            threshold=threshold,
            optimizer=optimizer,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            task_type=task_type,
            threshold=threshold,
            optimizer=None,
        )

        epoch_record = {"epoch": epoch}
        epoch_record.update(
            {f"train_{key}": value for key, value in train_metrics.items()}
        )
        epoch_record.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f}"
        )

        # Saving the best validation checkpoint keeps later inference simple.
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")

    if history_path is not None:
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return history
