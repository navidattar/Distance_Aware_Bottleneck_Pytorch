# coding=utf-8
# Copyright 2024 Ifigeneia Apostolopoulou (original TensorFlow implementation)
# Copyright 2026 PyTorch port authors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file for the full text.
"""Pretrained ResNet-50 with a Distance-Aware Bottleneck head (ImageNet model).

Port of ``models/pretrained_resnet50_dab.py``. The backbone is a torchvision
ResNet-50 truncated before its classifier; the DAB head is a three-layer MLP
with a residual connection, followed by a diagonal-covariance DAB layer and a
linear decoder.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import DABOutput, NormalDiagCovarianceDAB


class ResNet50DAB(nn.Module):
    """ResNet-50 + DAB head.

    Args:
      num_classes: number of output classes.
      dab_dim: bottleneck dimension.
      codebook_size: number of centroids.
      dab_tau: temperature of the distances from the codebook.
      backpropagate: whether gradients reach the backbone. ``False`` freezes it
        (and puts it in eval mode) so only the head and the codebook are
        trained, matching the reference default.
      pretrained: load torchvision's ImageNet weights.
      hidden: width of the head's hidden layers.
      dab_kwargs: extra keyword arguments forwarded to the DAB layer.
    """

    def __init__(self, num_classes: int = 1000, dab_dim: int = 4,
                 codebook_size: int = 1000, dab_tau: float = 2.0,
                 backpropagate: bool = False, pretrained: bool = True,
                 hidden: int = 2048,
                 generator: Optional[torch.Generator] = None, **dab_kwargs):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # -> [B,2048,1,1]
        self.backpropagate = backpropagate
        if not backpropagate:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.fc1 = nn.Linear(2048, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.dab = NormalDiagCovarianceDAB(
            in_features=hidden, dab_dim=dab_dim, codebook_size=codebook_size,
            dab_tau=dab_tau, generator=generator, **dab_kwargs)
        self.decoder = nn.Linear(dab_dim, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.backpropagate:
            self.backbone.eval()      # keep the frozen BatchNorm statistics
        return self

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Head features feeding the bottleneck (``hidden_4 + hidden_1``)."""
        with torch.set_grad_enabled(self.backpropagate and torch.is_grad_enabled()):
            f = torch.flatten(self.backbone(x), 1)
        h1 = F.relu(self.fc1(f))
        h3 = F.relu(self.fc2(h1))
        h4 = F.relu(self.fc3(h3))
        return h4 + h1

    def forward(self, x: torch.Tensor, training: Optional[bool] = None
                ) -> Tuple[torch.Tensor, DABOutput]:
        out = self.dab(self.features(x), training=training)
        return self.decoder(out.latent), out

    def forward_concat(self, x: torch.Tensor,
                       training: Optional[bool] = None) -> torch.Tensor:
        logits, out = self.forward(x, training=training)
        return torch.cat([logits, out.distance.unsqueeze(-1)], dim=-1)


def pretrained_resnet50_dab(num_classes: int = 1000, dab_dim: int = 4,
                            codebook_size: int = 1000, dab_tau: float = 2.0,
                            backpropagate: bool = False,
                            **kwargs) -> ResNet50DAB:
    """Functional alias mirroring the reference ``pretrained_resnet50_dab``."""
    return ResNet50DAB(num_classes=num_classes, dab_dim=dab_dim,
                       codebook_size=codebook_size, dab_tau=dab_tau,
                       backpropagate=backpropagate, **kwargs)


__all__ = ["ResNet50DAB", "pretrained_resnet50_dab"]
