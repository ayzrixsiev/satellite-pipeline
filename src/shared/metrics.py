"""Binary segmentation metrics."""

from __future__ import annotations

import torch


def summarize_binary_batch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Compute TP, FP, and FN counts for one batch.

    We keep the low-level counts here because precision, recall, IoU, and Dice
    can all be derived from them later.
    """

    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > threshold).float()

    true_positives = (predictions * targets).sum().item()
    false_positives = (predictions * (1 - targets)).sum().item()
    false_negatives = ((1 - predictions) * targets).sum().item()

    return true_positives, false_positives, false_negatives


def summarize_epoch_metrics(
    total_loss: float,
    sample_count: int,
    true_positives: float,
    false_positives: float,
    false_negatives: float,
) -> dict[str, float]:
    """Turn accumulated counters into readable metrics."""

    epsilon = 1e-8

    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)
    iou = true_positives / (true_positives + false_positives + false_negatives + epsilon)
    dice = (2 * true_positives) / (
        2 * true_positives + false_positives + false_negatives + epsilon
    )

    return {
        "loss": total_loss / max(sample_count, 1),
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
    }

