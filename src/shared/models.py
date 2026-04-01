"""Shared model, loss, and optimizer builders."""

from __future__ import annotations

import torch
import segmentation_models_pytorch as smp


def _build_unet(
    in_channels: int,
    encoder_name: str = "resnet18",
    use_pretrained: bool = True,
) -> torch.nn.Module:
    """Create one U-Net model.

    We use a single baseline architecture on purpose:
    it keeps the project simple and lets you focus on the full pipeline first.
    """

    encoder_weights = "imagenet" if use_pretrained and in_channels == 3 else None

    try:
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
    except Exception as error:
        # If pretrained weights are unavailable offline, we fall back to a model
        # with randomly initialized encoder weights so the code still runs.
        print(f"Could not load pretrained encoder weights ({error}). Falling back to random init.")
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
        )


def build_segmentation_model(
    encoder_name: str = "resnet18",
    use_pretrained: bool = True,
) -> torch.nn.Module:
    """Build a model for ordinary segmentation tasks."""

    return _build_unet(
        in_channels=3,
        encoder_name=encoder_name,
        use_pretrained=use_pretrained,
    )


def build_change_detection_model(
    encoder_name: str = "resnet18",
    use_pretrained: bool = False,
) -> torch.nn.Module:
    """Build a model for change detection.

    We feed the network a 6-channel tensor:
    - 3 channels from image at time 1
    - 3 channels from image at time 2
    """

    return _build_unet(
        in_channels=6,
        encoder_name=encoder_name,
        use_pretrained=use_pretrained,
    )


def build_binary_loss(
    device: torch.device,
    positive_class_weight: float | None = None,
) -> torch.nn.Module:
    """Create BCE-with-logits loss for binary masks."""

    if positive_class_weight is None:
        return torch.nn.BCEWithLogitsLoss()

    # Positive class weighting is useful for road detection because roads often
    # occupy only a small fraction of pixels.
    pos_weight = torch.tensor([positive_class_weight], device=device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 1e-3,
) -> torch.optim.Optimizer:
    """Use Adam as a clean and reliable baseline optimizer."""

    return torch.optim.Adam(model.parameters(), lr=learning_rate)

