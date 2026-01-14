# ==========================================================
# models.py
# Deep Learning Model Architectures for Video Classification
# ==========================================================

import os
import torch
import torch.nn as nn
from torchvision import models


class CNN2DTemporal(nn.Module):
    """
    2D CNN with Temporal Feature Aggregation.

    This model:
    - Extracts frame-level spatial features using ResNet-18
    - Aggregates features across time using mean + max pooling
    - Performs video-level classification

    Input:
        x: Tensor of shape (B, T, C, H, W)

    Output:
        logits: Tensor of shape (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        local_weights_path: str | None = None
    ):
        super().__init__()

        # --------------------------------------------------
        # Initialize ResNet-18 backbone (no auto-download)
        # --------------------------------------------------
        base = models.resnet18(weights=None)

        # --------------------------------------------------
        # Load pretrained weights if provided
        # (Handled silently; logging should be external)
        # --------------------------------------------------
        if local_weights_path and os.path.exists(local_weights_path):
            state_dict = torch.load(local_weights_path, map_location="cpu")
            base.load_state_dict(state_dict)

        # --------------------------------------------------
        # Remove final classification layer
        # Resulting output: (B*T, 512, 1, 1)
        # --------------------------------------------------
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        # --------------------------------------------------
        # Video-level classifier
        # --------------------------------------------------
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor):
                Input video tensor of shape (B, T, C, H, W)

        Returns:
            torch.Tensor:
                Class logits of shape (B, num_classes)
        """

        B, T, C, H, W = x.shape

        # --------------------------------------------------
        # Merge batch and temporal dimensions
        # --------------------------------------------------
        x = x.view(B * T, C, H, W)

        # --------------------------------------------------
        # Extract spatial features per frame
        # --------------------------------------------------
        features = self.backbone(x)          # (B*T, 512, 1, 1)
        features = features.view(B, T, -1)   # (B, T, 512)

        # --------------------------------------------------
        # Temporal aggregation
        # Mean pooling → global appearance
        # Max pooling  → salient motion cues
        # --------------------------------------------------
        pooled = features.mean(dim=1) + features.max(dim=1)[0]

        # --------------------------------------------------
        # Final classification
        # --------------------------------------------------
        return self.classifier(pooled)


class CNN3D(nn.Module):
    """
    3D CNN using R(2+1)D-18 for spatiotemporal modeling.

    This architecture:
    - Applies 3D convolutions across space and time
    - Learns motion and appearance jointly
    - Is computationally heavier but more expressive
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # --------------------------------------------------
        # Load pretrained 3D CNN backbone
        # --------------------------------------------------
        self.model = models.video.r2plus1d_18(pretrained=True)

        # --------------------------------------------------
        # Replace classification head
        # --------------------------------------------------
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor):
                Input video tensor of shape (B, T, C, H, W)

        Returns:
            torch.Tensor:
                Class logits of shape (B, num_classes)
        """

        # --------------------------------------------------
        # Convert to (B, C, T, H, W) for 3D convolutions
        # --------------------------------------------------
        x = x.permute(0, 2, 1, 3, 4)

        return self.model(x)
