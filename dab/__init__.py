# coding=utf-8
# Copyright 2024 Ifigeneia Apostolopoulou (original TensorFlow implementation)
# Copyright 2026 PyTorch port authors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file for the full text.
"""Distance-Aware Bottleneck (DAB) -- PyTorch.

A faithful PyTorch port of the reference TensorFlow implementation of

    Apostolopoulou, Eysenbach, Nielsen & Dubrawski,
    "A Rate-Distortion View of Uncertainty Quantification", ICML 2024.
    https://arxiv.org/abs/2406.10775
    https://github.com/ifiaposto/Distance_Aware_Bottleneck

Quick start::

    from dab import NormalFullCovarianceDAB, codebook_distortion, rdfc_epoch

    layer  = NormalFullCovarianceDAB(in_features=640, dab_dim=8, codebook_size=10)
    out    = layer(features)          # -> DABOutput
    logits = head(out.latent)
    score  = out.distance             # per-example uncertainty
"""
from .functional import (fill_triangular, kl_normal_diag, kl_normal_full,
                         symmetrized_fill_triangular)
from .layers import DABOutput, NormalDiagCovarianceDAB, NormalFullCovarianceDAB
from .objectives import codebook_distortion, dab_loss
from .trainer import (WarmUpPiecewiseConstantSchedule, build_optimizers,
                      codebook_parameters, find_dab_layers, network_parameters,
                      rdfc_epoch)

__version__ = "1.0.0"

__all__ = [
    "NormalDiagCovarianceDAB", "NormalFullCovarianceDAB", "DABOutput",
    "codebook_distortion", "dab_loss",
    "rdfc_epoch", "codebook_parameters", "network_parameters",
    "find_dab_layers", "build_optimizers", "WarmUpPiecewiseConstantSchedule",
    "fill_triangular", "symmetrized_fill_triangular",
    "kl_normal_diag", "kl_normal_full",
    "models",
]
