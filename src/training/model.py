# src/training/model.py
import torch
import torch.nn as nn
from torchvision import models
from typing import Any


class QCModel(nn.Module):
    """
    Custom model for binary classification (good vs. defective).
    Why: Uses transfer learning—pretrained on ImageNet for general features, fine-tune for pizza specifics.
         MobileNet for edge efficiency (lightweight, fast on CPU).
    How: Freeze early layers, replace classifier head for binary output.
    Interview: "Transfer learning speeds training and improves accuracy with limited data; I freeze conv layers to retain low-level features like textures for defect detection."
    """

    def __init__(
        self,
        pretrained: bool = True,
        model_name: str = "mobilenet_v3_small",
        num_classes: int = 2,
    ):
        """
        Initializes the model.
        Args:
            pretrained: Load ImageNet weights.
            model_name: 'resnet18', 'efficientnet_b0', or 'mobilenet_v3_small' (light for edge).
            num_classes: 2 for binary.
        """
        super().__init__()
        if model_name == "mobilenet_v3_small":
            self.base = models.mobilenet_v3_small(pretrained=pretrained)
            in_features = self.base.classifier[
                3
            ].in_features  # Get input to final layer.
            self.base.classifier[3] = nn.Linear(
                in_features, num_classes
            )  # Replace head for binary.
        elif model_name == "resnet18":
            self.base = models.resnet18(pretrained=pretrained)
            in_features = self.base.fc.in_features
            self.base.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Freeze early layers for fine-tuning (only train last layers initially).
        self.freeze_layers()

    def freeze_layers(self):
        """Freezes base layers to fine-tune only the head. Why: Prevents overwriting pretrained features, reduces training time/overfitting."""
        for param in self.base.parameters():
            param.requires_grad = False  # Freeze all.
        # Unfreeze last block/head (customize based on model).
        if hasattr(self.base, "classifier"):
            for param in self.base.classifier.parameters():
                param.requires_grad = True
        elif hasattr(self.base, "fc"):
            for param in self.base.fc.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits. Why: Simple for inference compatibility."""
        return self.base(x)

    def unfreeze_all(self):
        """Unfreezes all layers for full fine-tuning after initial epochs. Why: Gradual fine-tuning improves convergence."""
        for param in self.base.parameters():
            param.requires_grad = True


def load_model(
    checkpoint_path: str = None, device: torch.device = torch.device("cpu")
) -> QCModel:
    """
    Loads a saved model.
    Why: For resuming training or inference—production systems often load from checkpoints.
    """
    model = QCModel()
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  # Set to eval mode for inference.
    return model


# Example: model = QCModel(model_name='mobilenet_v3_small')
