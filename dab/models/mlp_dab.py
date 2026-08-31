# coding=utf-8
# Copyright 2024 Ifigeneia Apostolopoulou (original TensorFlow implementation)
# Copyright 2026 PyTorch port authors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file for the full text.
"""Small MLP with a Distance-Aware Bottleneck.

Port of the network in ``synthetic_regression_demo.py``: two ELU hidden layers,
a full-covariance DAB layer with ``activation=None`` and ``momentum=0.0``, then
a linear decoder. Useful as a minimal, fast end-to-end DAB example and as a
drop-in bottleneck for tabular models.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import DABOutput, NormalFullCovarianceDAB


class MLPDAB(nn.Module):
    """``in_features -> [hidden, ELU] * n -> DAB -> out_features``.

    Args:
      in_features / out_features: input and output widths.
      hidden: hidden width.
      num_hidden: number of hidden layers.
      dab_dim, codebook_size, dab_tau: bottleneck configuration.
      momentum: DAB moving-average momentum. ``0.0`` (the demo's setting) takes
        each batch's estimate outright, which is the right choice when a "batch"
        is the whole dataset.
      dab_activation: non-linearity applied inside the DAB layer before its
        dense encoder. ``None`` in the demo, because the preceding hidden layer
        is already activated.
    """

    def __init__(self, in_features: int = 1, out_features: int = 1,
                 hidden: int = 100, num_hidden: int = 2, dab_dim: int = 8,
                 codebook_size: int = 1, dab_tau: float = 5.0,
                 momentum: float = 0.0, dab_activation=None,
                 generator: Optional[torch.Generator] = None, **dab_kwargs):
        super().__init__()
        layers, width = [], in_features
        for _ in range(num_hidden):
            layers += [nn.Linear(width, hidden), nn.ELU()]
            width = hidden
        self.body = nn.Sequential(*layers)
        self.dab = NormalFullCovarianceDAB(
            in_features=width, dab_dim=dab_dim, codebook_size=codebook_size,
            dab_tau=dab_tau, momentum=momentum, activation=dab_activation,
            generator=generator, **dab_kwargs)
        self.decoder = nn.Linear(dab_dim, out_features)

    def forward(self, x: torch.Tensor, training: Optional[bool] = None
                ) -> Tuple[torch.Tensor, DABOutput]:
        out = self.dab(self.body(x), training=training)
        return self.decoder(out.latent), out


__all__ = ["MLPDAB"]
