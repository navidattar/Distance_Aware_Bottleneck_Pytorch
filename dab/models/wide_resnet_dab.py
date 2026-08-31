# coding=utf-8
# Copyright 2024 Ifigeneia Apostolopoulou (original TensorFlow implementation)
# Copyright 2026 PyTorch port authors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file for the full text.
"""Wide ResNet with a Distance-Aware Bottleneck (the reference CIFAR-10 model).

Port of ``models/wide_resnet_dab.py``, which itself follows Uncertainty
Baselines (https://github.com/google/uncertainty-baselines/). Three groups of
residual blocks map 32x32 -> 16x16 -> 8x8.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import DABOutput, NormalFullCovarianceDAB


class BasicBlock(nn.Module):
    """Basic residual block of two 3x3 convolutions.

    Args:
      version: 1 for the original ordering (He et al., 2015) or 2 for the
        preactivation ordering (He et al., 2016). The reference model uses 2.
    """

    def __init__(self, in_planes: int, filters: int, stride: int,
                 version: int = 2):
        super().__init__()
        self.version = version
        # BatchNorm defaults chosen to match the reference model, which uses
        # "epsilon and momentum defaults from Torch": eps=1e-5 and a TF momentum
        # of 0.9, i.e. a PyTorch momentum of 0.1.
        bn = lambda c: nn.BatchNorm2d(c, eps=1e-5, momentum=0.1)
        if version == 2:
            self.bn0 = bn(in_planes)
        self.conv1 = nn.Conv2d(in_planes, filters, 3, stride, 1, bias=False)
        self.bn1 = bn(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, 1, 1, bias=False)
        self.bn2 = bn(filters) if version == 1 else None
        self.shortcut = (nn.Conv2d(in_planes, filters, 1, stride, 0, bias=False)
                         if (stride != 1 or in_planes != filters) else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        if self.version == 2:
            y = F.relu(self.bn0(y))
        y = self.conv1(y)
        y = F.relu(self.bn1(y))
        y = self.conv2(y)
        if self.version == 1:
            y = self.bn2(y)
        # The shortcut is taken from the block input, before any preactivation,
        # exactly as in the reference graph.
        s = self.shortcut(x) if self.shortcut is not None else x
        out = s + y
        return F.relu(out) if self.version == 1 else out


class WideResNetDAB(nn.Module):
    """Wide ResNet backbone + DAB bottleneck + linear decoder.

    Args:
      depth: total number of convolutional layers ("n" in WRN-n-k); must satisfy
        ``(depth - 4) % 6 == 0``.
      width_multiplier: "k" in WRN-n-k.
      num_classes: number of output classes.
      dab_dim: bottleneck dimension.
      codebook_size: number of centroids.
      dab_tau: temperature of the distances from the codebook.
      version: residual-block ordering (see :class:`BasicBlock`).
      in_channels: number of input image channels.
      dab_kwargs: extra keyword arguments forwarded to the DAB layer.

    Forward:
      ``x -> (logits [B, num_classes], DABOutput)``. ``DABOutput.distance`` is
      the per-example uncertainty; the reference model returns
      ``concat([logits, distance], -1)``, which :meth:`forward_concat`
      reproduces.
    """

    def __init__(self, depth: int = 28, width_multiplier: int = 10,
                 num_classes: int = 10, dab_dim: int = 8,
                 codebook_size: int = 10, dab_tau: float = 1.0,
                 version: int = 2, in_channels: int = 3,
                 generator: Optional[torch.Generator] = None, **dab_kwargs):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("depth should be 6n+4 (e.g., 16, 22, 28, 40).")
        num_blocks = (depth - 4) // 6
        self.version = version
        widths = [16, 16 * width_multiplier, 32 * width_multiplier,
                  64 * width_multiplier]

        self.conv1 = nn.Conv2d(in_channels, widths[0], 3, 1, 1, bias=False)
        self.bn_stem = (nn.BatchNorm2d(widths[0], eps=1e-5, momentum=0.1)
                        if version == 1 else None)
        self.group1 = self._group(widths[0], widths[1], num_blocks, 1, version)
        self.group2 = self._group(widths[1], widths[2], num_blocks, 2, version)
        self.group3 = self._group(widths[2], widths[3], num_blocks, 2, version)
        self.bn_out = (nn.BatchNorm2d(widths[3], eps=1e-5, momentum=0.1)
                       if version == 2 else None)
        self.feature_dim = widths[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")

        # The reference model passes HeNormal to the DAB dense layer.
        dab_kwargs.setdefault("kernel_initializer", "he_normal")
        self.dab = NormalFullCovarianceDAB(
            in_features=self.feature_dim, dab_dim=dab_dim,
            codebook_size=codebook_size, dab_tau=dab_tau,
            generator=generator, **dab_kwargs)
        self.decoder = nn.Linear(dab_dim, num_classes)
        nn.init.kaiming_normal_(self.decoder.weight, mode="fan_in",
                                nonlinearity="relu")
        nn.init.zeros_(self.decoder.bias)

    @staticmethod
    def _group(in_planes, filters, num_blocks, stride, version):
        layers = [BasicBlock(in_planes, filters, stride, version)]
        layers += [BasicBlock(filters, filters, 1, version)
                   for _ in range(num_blocks - 1)]
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled backbone features, before the bottleneck."""
        x = self.conv1(x)
        if self.bn_stem is not None:
            x = F.relu(self.bn_stem(x))
        x = self.group3(self.group2(self.group1(x)))
        if self.bn_out is not None:
            x = F.relu(self.bn_out(x))
        return torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)

    def forward(self, x: torch.Tensor, training: Optional[bool] = None
                ) -> Tuple[torch.Tensor, DABOutput]:
        out = self.dab(self.features(x), training=training)
        return self.decoder(out.latent), out

    def forward_concat(self, x: torch.Tensor,
                       training: Optional[bool] = None) -> torch.Tensor:
        """``concat([logits, distance], -1)`` -- the reference model's output."""
        logits, out = self.forward(x, training=training)
        return torch.cat([logits, out.distance.unsqueeze(-1)], dim=-1)


def wide_resnet_dab(depth: int = 28, width_multiplier: int = 10,
                    num_classes: int = 10, dab_dim: int = 8,
                    codebook_size: int = 10, dab_tau: float = 1.0,
                    version: int = 2, **kwargs) -> WideResNetDAB:
    """Functional alias mirroring the reference ``wide_resnet_dab(...)``."""
    return WideResNetDAB(depth=depth, width_multiplier=width_multiplier,
                         num_classes=num_classes, dab_dim=dab_dim,
                         codebook_size=codebook_size, dab_tau=dab_tau,
                         version=version, **kwargs)


__all__ = ["WideResNetDAB", "wide_resnet_dab", "BasicBlock"]
