# coding=utf-8
# Copyright 2024 Ifigeneia Apostolopoulou (original TensorFlow implementation)
# Copyright 2026 PyTorch port authors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file for the full text.
"""Loss terms for training a model with a Distance-Aware Bottleneck."""
from __future__ import annotations

from typing import Optional

import torch


def codebook_distortion(uncertainty: torch.Tensor,
                        matches: Optional[torch.Tensor] = None,
                        uncertainty_lb: float = 100.0) -> torch.Tensor:
    r"""Reduce the per-example codebook distance to the scalar DAB penalty.

    Two reductions are used by the reference implementation.

    ``matches is None`` (CIFAR-10 / synthetic settings, and the ImageNet setting
    with ``--calibrate=False``) -- the codebook must quantise the encoders of
    *all* training datapoints::

        L = mean_i d(x_i)

    ``matches`` given (the ImageNet setting with ``--calibrate=True``) -- the
    codebook quantises the encoders of *correctly classified* datapoints, while
    misclassified datapoints whose distance is below ``uncertainty_lb`` are
    pushed away from the codebook::

        L = (1/N) sum_i [ m_i d(x_i) + (1 - m_i) max(0, lb - d(x_i)) ]

    where ``m_i in {0, 1}`` indicates a correct prediction. Misclassified points
    that are already further than ``uncertainty_lb`` contribute nothing. This is
    what makes the distance a *calibrated* uncertainty score rather than a pure
    density proxy.

    Args:
      uncertainty: ``[B]`` codebook distances (``DABOutput.distance``).
      matches: optional ``[B]`` float tensor of 0/1 correctness indicators.
        Must be detached -- it is an indicator, not a differentiable quantity.
      uncertainty_lb: the margin ``lb`` above.

    Returns:
      Scalar tensor.
    """
    if matches is None:
        return uncertainty.mean()
    matches = matches.to(uncertainty.dtype).detach()
    batch_size = uncertainty.shape[0]
    correct = (matches * uncertainty).sum() / batch_size
    wrong = ((1.0 - matches)
             * torch.clamp(uncertainty_lb - uncertainty, min=0.0)).sum() \
        / batch_size
    return correct + wrong


def dab_loss(negative_log_likelihood: torch.Tensor,
             uncertainty: torch.Tensor, beta: float,
             matches: Optional[torch.Tensor] = None,
             uncertainty_lb: float = 100.0) -> torch.Tensor:
    r"""The DAB training objective for the encoder/decoder phase.

    ``L = NLL + beta * distortion``. Weight decay (the reference code's explicit
    ``l2`` kernel regulariser) is expected to be handled by the optimiser's
    ``weight_decay``; see :func:`dab.trainer.build_optimizers`.

    Args:
      negative_log_likelihood: scalar task loss (already reduced over the batch).
      uncertainty: ``[B]`` codebook distances.
      beta: Lagrange multiplier on the rate-distortion penalty.
      matches, uncertainty_lb: forwarded to :func:`codebook_distortion`.
    """
    return negative_log_likelihood + beta * codebook_distortion(
        uncertainty, matches, uncertainty_lb)


__all__ = ["codebook_distortion", "dab_loss"]
