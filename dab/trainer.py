# coding=utf-8
# Copyright 2024 Ifigeneia Apostolopoulou (original TensorFlow implementation)
# Copyright 2026 PyTorch port authors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file for the full text.
"""The two-phase DAB training schedule.

Every DAB epoch consists of

  1. an **encoder/decoder phase** -- one pass over the data, updating everything
     *except* the codebook on ``NLL + beta * distortion``; then
  2. an **RDFC phase** (Rate-Distortion with Finite Cardinality) -- the M-step,
     which updates *only* the codebook. It has two sub-passes:

     a. the centroid means are trained by gradient descent on the mean codebook
        distance while the codebook covariances are accumulated in a moving
        average, which is then committed;
     b. a second forward-only pass re-runs the E-step with the freshly committed
        covariances and accumulates the prior centroid probabilities, which are
        then committed.

The phase order and the two sub-passes matter: the priors must be estimated with
the covariances that will actually be used, and the ``initialized`` flag is only
raised at the start of the first RDFC phase, so the very first encoder/decoder
epoch runs with uniform assignments.

References:
  Banerjee, Dhillon, Ghosh & Merugu, "An information theoretic analysis of
  maximum likelihood mixture estimation for exponential families", ICML 2004.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from .layers import DABLayer


# --------------------------------------------------------------------------- #
#  Parameter split
# --------------------------------------------------------------------------- #
def codebook_parameters(module: nn.Module) -> List[nn.Parameter]:
    """Parameters trained by the RDFC phase.

    Mirrors the reference code's ``if "centroid" in var.name`` filter: the only
    trainable codebook tensor is ``centroid_means`` (the covariances and priors
    are buffers updated in closed form, not by gradient descent).
    """
    return [p for n, p in module.named_parameters()
            if "centroid" in n and p.requires_grad]


def network_parameters(module: nn.Module) -> List[nn.Parameter]:
    """Parameters trained by the encoder/decoder phase (everything else)."""
    return [p for n, p in module.named_parameters()
            if "centroid" not in n and p.requires_grad]


def find_dab_layers(module: nn.Module) -> List[DABLayer]:
    """All DAB layers inside ``module`` (unwraps ``DataParallel``/``DDP``).

    Finds every codebook flavour -- the reference ``NormalDiagCovarianceDAB`` /
    ``NormalFullCovarianceDAB`` and the FSQ-grid ``FSQDAB`` alike.
    """
    return [m for m in module.modules() if isinstance(m, DABLayer)]


def build_optimizers(module: nn.Module, base_learning_rate: float = 0.1,
                     rdfc_learning_rate: float = 0.1, momentum: float = 0.9,
                     nesterov: bool = True, weight_decay: float = 2e-4):
    """The reference optimiser pair.

    * network: SGD with Nesterov momentum and ``weight_decay`` standing in for
      the reference code's explicit L2 kernel regulariser;
    * codebook: Adam at ``rdfc_learning_rate``.

    Returns ``(optimizer, codebook_optimizer)``. ``codebook_optimizer`` is
    ``None`` when there is nothing to train in the codebook -- either the module
    has no DAB layer, or it uses an :class:`~dab.fsq_dab.FSQDAB` whose grid is
    fixed and whose variances are fitted in closed form. ``rdfc_epoch`` accepts
    ``None`` and still runs both M-step sub-passes.
    """
    net = network_parameters(module)
    code = codebook_parameters(module)
    optimizer = torch.optim.SGD(net, lr=base_learning_rate, momentum=momentum,
                                nesterov=nesterov, weight_decay=weight_decay)
    codebook_optimizer = (torch.optim.Adam(code, lr=rdfc_learning_rate)
                          if code else None)
    return optimizer, codebook_optimizer


class WarmUpPiecewiseConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup followed by piecewise-constant decay, applied per step.

    Port of ``utils/schedules.WarmUpPiecewiseConstantSchedule`` (Uncertainty
    Baselines). Call ``scheduler.step()`` after every optimiser step.

    Args:
      optimizer: the network optimiser. Its ``lr`` is treated as the base value.
      steps_per_epoch: optimiser steps in one epoch.
      decay_ratio: multiplier applied at each decay epoch.
      decay_epochs: epochs at which to decay.
      warmup_epochs: linear warmup length; ``0`` disables warmup.
    """

    def __init__(self, optimizer, steps_per_epoch: int, decay_ratio: float,
                 decay_epochs: Sequence[int], warmup_epochs: int = 1,
                 last_epoch: int = -1):
        decay_epochs = list(decay_epochs)

        def fn(step: int) -> float:
            lr_epoch = step / steps_per_epoch
            scale = 1.0
            if warmup_epochs >= 1:
                scale = lr_epoch / warmup_epochs
            for index, start_epoch in enumerate([warmup_epochs] + decay_epochs):
                if lr_epoch >= start_epoch:
                    scale = decay_ratio ** index
            return scale

        super().__init__(optimizer, fn, last_epoch=last_epoch)


# --------------------------------------------------------------------------- #
#  RDFC phase
# --------------------------------------------------------------------------- #
def rdfc_epoch(dab_layers, codebook_optimizer,
               batches: Callable[[], Iterable],
               distortion_fn: Callable[[object], torch.Tensor],
               grad_clip: Optional[float] = None) -> float:
    """Run one full RDFC (codebook) phase.

    Args:
      dab_layers: a :class:`~dab.layers._NormalDAB`, a model containing one or
        more of them, or an explicit sequence of them.
      codebook_optimizer: optimiser over :func:`codebook_parameters`, or
        ``None`` when the codebook has no trainable parameters (an FSQ grid).
        Both sub-passes still run: the closed-form covariance and prior M-steps
        are what fit an FSQ codebook.
      batches: zero-argument callable returning a **fresh** iterable over the
        training data. It is called twice, once per sub-pass.
      distortion_fn: ``batch -> scalar`` -- must run a forward pass of the whole
        model with ``training=True`` and return the codebook loss (usually
        ``dab.objectives.codebook_distortion(out.distance)``). Keep the model in
        ``train()`` mode, as the reference code does.
      grad_clip: optional global gradient-norm clip for the centroid means.

    Returns:
      The mean codebook loss over the first sub-pass.
    """
    layers = _as_layers(dab_layers)
    if not layers:
        raise ValueError("no DAB layer found")

    # Flag the network as initialised: from now on the E-step uses the learned
    # priors and distances instead of a uniform assignment.
    for layer in layers:
        layer.mark_initialized()

    # --- sub-pass (a): train the centroids, accumulate the covariances ---
    for layer in layers:
        layer.reset_codebook_covariance()
    total, n = 0.0, 0
    params = ([p for g in codebook_optimizer.param_groups for p in g["params"]]
              if codebook_optimizer is not None else [])
    for batch in batches():
        if codebook_optimizer is None:
            # Nothing to descend on (fixed FSQ grid); the forward pass is still
            # needed so the covariance moving average sees this batch.
            with torch.no_grad():
                loss = distortion_fn(batch)
        else:
            codebook_optimizer.zero_grad(set_to_none=True)
            loss = distortion_fn(batch)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            codebook_optimizer.step()
        total += float(loss.detach())
        n += 1
    for layer in layers:
        layer.set_codebook_covariance()

    # --- sub-pass (b): re-run the E-step, accumulate the priors ---
    for layer in layers:
        layer.reset_centroid_probs()
    with torch.no_grad():
        for batch in batches():
            distortion_fn(batch)
    for layer in layers:
        layer.set_centroid_probs()

    return total / max(n, 1)


def _as_layers(x) -> List[_NormalDAB]:
    if isinstance(x, DABLayer):
        return [x]
    if isinstance(x, nn.Module):
        return find_dab_layers(x)
    return list(x)


__all__ = ["codebook_parameters", "network_parameters", "find_dab_layers",
           "build_optimizers", "WarmUpPiecewiseConstantSchedule", "rdfc_epoch"]
