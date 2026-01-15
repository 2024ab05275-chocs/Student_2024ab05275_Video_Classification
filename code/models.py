# ==========================================================
# models.py
# Deep Learning Model Architectures for Video Classification
# ==========================================================

from typing import Optional
import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.video import R2Plus1D_18_Weights
from torchvision.models import ResNet18_Weights


# ==========================================================
# 2D CNN + Temporal Aggregation
# ==========================================================

class CNN2DTemporal(nn.Module):
    """
    2D CNN with Temporal Feature Aggregation.

    - ImageNet-pretrained ResNet-18
    - Temporal mean + max pooling
    - Dropout + Normalization for regularization
    """

    def __init__(
        self,
        num_classes: int,
        local_weights_path: Optional[str] = None,
        dropout_p: float = 0.5,
        norm_type: str = "batch"  # "batch" or "layer"
    ):
        super().__init__()

        # --------------------------------------------------
        # Load ResNet-18 (ImageNet pretrained)
        # --------------------------------------------------
        base = models.resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )

        # Optional local weight override
        if local_weights_path and os.path.exists(local_weights_path):
            state_dict = torch.load(
                local_weights_path,
                map_location="cpu"
            )
            base.load_state_dict(state_dict)

        # Backbone (remove FC)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        feat_dim = base.fc.in_features

        # --------------------------------------------------
        # Normalization choice
        # --------------------------------------------------
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(feat_dim)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(feat_dim)
        else:
            raise ValueError("norm_type must be 'batch' or 'layer'")

        # --------------------------------------------------
        # Classifier with Dropout
        # --------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape

        # Frame-level feature extraction
        x = x.reshape(B * T, C, H, W)
        feats = self.backbone(x)              # (B*T, 512, 1, 1)
        feats = feats.reshape(B, T, -1)       # (B, T, 512)

        # Temporal aggregation
        pooled = feats.mean(dim=1) + feats.max(dim=1)[0]

        # Normalization + classification
        pooled = self.norm(pooled)
        return self.classifier(pooled)


# ==========================================================
# 3D CNN (R(2+1)D)
# ==========================================================

class CNN3D(nn.Module):
    """
    3D CNN using R(2+1)D-18 for spatiotemporal modeling.

    - Kinetics-400 pretrained
    - Dropout + normalization
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout_p: float = 0.5
    ):
        super().__init__()

        weights = (
            R2Plus1D_18_Weights.KINETICS400_V1
            if pretrained else None
        )

        self.model = models.video.r2plus1d_18(weights=weights)

        feat_dim = self.model.fc.in_features

        # Replace classifier with regularized head
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        """
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        return self.model(x)


# ==========================================================
# Early Stopping Utility
# ==========================================================

class EarlyStopping:
    """
    Early stopping to terminate training when validation
    loss stops improving.

    Usage:
        early_stopper = EarlyStopping(patience=5)
        stop = early_stopper.step(val_loss)
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        """
        Returns True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience
