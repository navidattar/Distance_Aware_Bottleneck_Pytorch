# coding=utf-8
# Copyright 2026 PyTorch port authors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file for the full text.
"""One entry point for building any DAB codebook.

``build_bottleneck(kind, ...)`` lets a model take the codebook as a string
option, so switching between the reference DAB codebook and the FSQ product grid
is a one-word change rather than an import change.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

from .fsq_dab import FSQDAB
from .layers import DABLayer, NormalDiagCovarianceDAB, NormalFullCovarianceDAB

#: Codebook kinds accepted by :func:`build_bottleneck`.
BOTTLENECKS = {
    "full": NormalFullCovarianceDAB,   # reference DAB, full-covariance codes
    "diag": NormalDiagCovarianceDAB,   # reference DAB, diagonal-covariance codes
    "fsq": FSQDAB,                     # FSQ product grid of scalar Gaussians
}


def build_bottleneck(kind: str, in_features: int,
                     dab_dim: Optional[int] = None,
                     codebook_size: Optional[int] = None,
                     levels: Union[int, Sequence[int], None] = None,
                     **kwargs) -> DABLayer:
    """Construct a DAB bottleneck by name.

    Args:
      kind: one of

        * ``"full"`` -- the reference DAB codebook, ``K`` full-covariance
          Gaussian centroids (the paper's CIFAR-10 setting);
        * ``"diag"`` -- the reference DAB codebook with diagonal covariances
          (the paper's ImageNet setting);
        * ``"fsq"`` -- a Finite Scalar Quantization product grid; see
          :class:`~dab.fsq_dab.FSQDAB`.

      in_features: size of the incoming feature vector.
      dab_dim: latent dimension. Required for ``"full"``/``"diag"``. For
        ``"fsq"`` it is inferred from ``levels`` when those are given.
      codebook_size: number of centroids ``K``. Required for
        ``"full"``/``"diag"``, ignored by ``"fsq"`` (whose codebook size is
        ``prod(levels)``; use :func:`dab.fsq.recommended_levels` to match a
        target size).
      levels: ``"fsq"`` only -- values per coordinate, or a single int applied
        to every coordinate. Defaults to ``5`` per coordinate, the smallest
        value the FSQ paper's heuristic allows.
      **kwargs: forwarded to the layer (``dab_tau``, ``momentum``,
        ``activation``, and the kind-specific options).

    Returns:
      A :class:`~dab.layers.DABLayer`.

    Example:
      >>> from dab import build_bottleneck, recommended_levels
      >>> build_bottleneck("full", 640, dab_dim=8, codebook_size=10)
      >>> build_bottleneck("fsq", 640, levels=recommended_levels(1024))
    """
    if kind not in BOTTLENECKS:
        raise ValueError(
            f"unknown bottleneck {kind!r}; choose from {sorted(BOTTLENECKS)}")

    if kind == "fsq":
        if levels is None:
            if dab_dim is None:
                raise ValueError("pass `levels` or `dab_dim` for kind='fsq'")
            levels = 5
        if isinstance(levels, int):
            if dab_dim is None:
                raise ValueError("pass `dab_dim` when `levels` is an int")
            return FSQDAB(in_features=in_features, levels=levels,
                          dab_dim=dab_dim, **kwargs)
        return FSQDAB(in_features=in_features, levels=levels, **kwargs)

    if dab_dim is None or codebook_size is None:
        raise ValueError(f"`dab_dim` and `codebook_size` are required for "
                         f"kind={kind!r}")
    return BOTTLENECKS[kind](in_features=in_features, dab_dim=dab_dim,
                             codebook_size=codebook_size, **kwargs)


__all__ = ["build_bottleneck", "BOTTLENECKS"]
