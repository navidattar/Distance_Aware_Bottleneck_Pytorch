# coding=utf-8
# Copyright 2026 PyTorch port authors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file for the full text.
"""iFSQ -- Finite Scalar Quantization with a distribution-matching bound.

Implementation of

    Lin, Li, Niu, Gong, Ge, Lin, Zheng, Zhang, Yang, Zhong, Bo & Yuan,
    "iFSQ: Improving FSQ for Image Generation with 1 Line of Code", 2026.
    https://arxiv.org/abs/2601.17124

The observation is narrow and neat. FSQ bounds the latent with ``tanh``. Written
as a sigmoid, ``tanh(z) = 2 sigma(2z) - 1``: the slope is ``alpha = 2``. Push a
standard-normal latent through that and the result is *bimodal* — mass piles up
near the two ends of the interval, so the middle bins are under-used and the
grid does not carry as many bits as it could. Sweeping the slope, ``alpha = 1.6``
maps a standard normal to an approximately **uniform** distribution on
``[-1, 1]``, which is the distribution that maximises the entropy of the level
assignment. The paper reports ~100% bin utilisation from this one-line change::

    - z = tanh(z)
    + z = 2 * sigmoid(1.6 * z) - 1

Everything downstream — scaling by ``(L-1)/2``, rounding, the straight-through
estimator, the index arithmetic — is unchanged from FSQ.

Two classes are provided, mirroring the FSQ pair:

* :class:`IFSQ` -- the quantizer on its own, a drop-in for :class:`~dab.fsq.FSQ`
  in any VQ-VAE-style model;
* :class:`IFSQDAB` -- the Distance-Aware Bottleneck over an iFSQ grid, a drop-in
  for :class:`~dab.fsq_dab.FSQDAB`.

.. note::
   **Odd levels only.** The paper defines ``L = 2K + 1`` so that an exact zero
   centre exists, and its bounding map has no even-``L`` offset (unlike FSQ's).
   With an even ``L`` the map ``[-1, 1] -> [-(L-1)/2, (L-1)/2]`` rounds to
   ``L + 1`` distinct integers, not ``L``, so :class:`IFSQ` rejects even levels.
   Use ``[5, 5, 5, 5]`` (625 codes) or ``[9, 9, 9]`` (729) where FSQ would have
   used ``[8, 5, 5, 5]``.

.. note::
   **Index convention.** The paper's pseudocode uses a big-endian basis
   (``[L^(d-1), ..., L^0]``), so :class:`IFSQ` defaults to
   ``index_order="big"`` while :class:`~dab.fsq.FSQ` defaults to the Google
   reference's little-endian order. Both are bijections onto
   ``[0, prod(L))``; they just number the codewords differently, so tokens are
   not interchangeable between the two. Pass ``index_order`` explicitly if you
   need a particular numbering.

.. warning::
   The uniformity argument assumes the pre-bound latent is approximately
   standard normal, which is what an unconstrained encoder in a reconstruction
   autoencoder tends to produce. :class:`IFSQDAB` trains that latent under a
   rate-distortion objective with a learned prior instead, so the premise does
   not automatically hold and the benefit is not something this repository has
   measured. Treat ``IFSQDAB`` as an option worth A/B-ing against ``FSQDAB``,
   not as a known improvement.

.. warning::
   **Uniform pre-rounding value does not imply uniform bin occupancy.** The
   paper's step 2 scales the bounded value by ``(L-1)/2`` before rounding. That
   puts the two *outermost* bins at half the width of the interior ones, so even
   a perfectly uniform value on ``[-1, 1]`` produces the level histogram
   ``[1/2, 1, 1, ..., 1, 1/2] / (L-1)`` -- the end bins are under-filled by 2x.
   Pushing ``N(0, 1)`` through and measuring the marginal histogram, vanilla FSQ
   is in fact *closer* to uniform than iFSQ at every ``L`` from 5 to 65, because
   ``tanh``'s bimodality happens to pile mass onto exactly those half-width end
   bins. `alpha` fixes the pre-rounding distribution; it does not fix the grid.

   Passing ``edge_bins="equal"`` scales by ``L/2`` instead, which makes every
   bin the same width and delivers what the uniformity argument promises: at
   ``L=5`` the maximum deviation from a uniform histogram drops from 0.089 to
   0.008 and the realised entropy goes from 2.221 to 2.321 bits (the ceiling is
   2.322). This is an observation of ours, not the paper's, so ``"paper"``
   remains the default. Reproduce both with :func:`level_histogram`.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import torch

from .fsq import FSQ
from .fsq_dab import FSQDAB

#: Sigmoid slope that maps a standard normal to an approximately uniform
#: distribution on [-1, 1]. ``alpha = 2.0`` recovers ``tanh``, i.e. vanilla FSQ.
IFSQ_ALPHA = 1.6

#: The slope at which the iFSQ bound degenerates to FSQ's ``tanh``.
TANH_ALPHA = 2.0

#: The slope that actually minimises the deviation from uniform. The paper
#: sweeps a coarse grid, ``alpha in {1.0, 1.3, 1.6, 2.0, 2.4}``, and reports
#: ``1.6`` as the best of those. Refining the sweep puts the Kolmogorov-Smirnov
#: minimum at ``alpha ~ 1.70`` (KS 0.011 vs 0.017 at 1.6 and 0.046 at 2.0),
#: which is the classic logistic approximation to the Gaussian CDF,
#: ``sigma(1.702 x) ~ Phi(x)`` -- exactly what "match the distribution to a
#: uniform" predicts, since pushing ``z`` through its own CDF gives a uniform.
#: ``IFSQ_ALPHA`` remains the default so the implementation matches the paper;
#: pass ``alpha=IFSQ_ALPHA_KS_OPTIMAL`` if you want the refined value.
#: Reproduce either with :func:`uniformity_error`.
IFSQ_ALPHA_KS_OPTIMAL = 1.702


def ifsq_bound(z: torch.Tensor, alpha: float = IFSQ_ALPHA) -> torch.Tensor:
    """``2 * sigmoid(alpha * z) - 1`` -- the iFSQ distribution-matching bound.

    Maps ``R -> (-1, 1)``. ``alpha = 2`` is exactly ``tanh(z)``; ``alpha = 1.6``
    maps a standard normal to an approximately uniform distribution.
    """
    return 2.0 * torch.sigmoid(alpha * z) - 1.0


class IFSQ(FSQ):
    """iFSQ quantizer: :class:`~dab.fsq.FSQ` with the sigmoid-slope bound.

    Args:
      levels: values per channel. Must all be **odd** (see the module note).
        The paper's heuristic inherits FSQ's ``L_i >= 5``.
      alpha: sigmoid slope. ``1.6`` (default) is the paper's fitted value;
        ``2.0`` degenerates to ``tanh`` and therefore to vanilla FSQ up to the
        ``eps`` shrinkage; :data:`IFSQ_ALPHA_KS_OPTIMAL` (1.702) is the refined
        optimum of the paper's own criterion.
      index_order: ``"big"`` (default, the paper's pseudocode) or ``"little"``
        (the Google FSQ reference).
      edge_bins: ``"paper"`` (default) scales by ``(L-1)/2`` exactly as
        Algorithm 1 does; ``"equal"`` scales by ``L/2`` so that every bin has
        the same width. See the second module warning -- ``"equal"`` is what
        actually makes the level histogram uniform, but it is not the paper's.

    Example:
      >>> q = IFSQ([5, 5, 5, 5])       # 625 codes
      >>> zhat, indices = q(z)         # z: [B, 4]

    Everything except :meth:`bound` is inherited unchanged, so ``quantize``,
    ``codes_to_indices``, ``indices_to_codes``, ``codebook`` and the normalised
    helpers behave exactly as they do for FSQ.
    """

    def __init__(self, levels: Sequence[int], alpha: float = IFSQ_ALPHA,
                 eps: float = 0.0, index_order: str = "big",
                 edge_bins: str = "paper"):
        levels = [int(v) for v in levels]
        if edge_bins not in ("paper", "equal"):
            raise ValueError(
                f"edge_bins must be 'paper' or 'equal', got {edge_bins!r}")
        even = [L for L in levels if L % 2 == 0]
        if even:
            raise ValueError(
                f"iFSQ requires odd levels (L = 2K + 1, for an exact zero "
                f"centre); got even {even} in {levels}. Its bounding map has no "
                f"even-L offset, so an even L would round to L + 1 distinct "
                f"values. Use FSQ for even levels, or round up (e.g. 8 -> 9).")
        super().__init__(levels, eps=eps, index_order=index_order)
        self.alpha = alpha
        self.edge_bins = edge_bins

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """``(2 sigma(alpha z) - 1) * (L - 1) / 2``.

        The paper's step 1 (the sigmoid) and step 2 (scale to the grid) fused,
        so the return value is in the same unnormalised space that FSQ's
        :meth:`~dab.fsq.FSQ.bound` returns and the inherited ``quantize`` can
        round it directly.

        ``eps`` defaults to ``0`` here: FSQ needs it so ``tanh`` can reach the
        outermost bins, but the sigmoid map already spreads mass into them.

        With ``edge_bins="equal"`` the scale is ``L/2`` and the result is clamped
        just inside the rounding range, so every bin -- ends included -- covers
        the same interval.
        """
        if z.shape[-1] != self.num_dimensions:
            raise ValueError(
                f"expected last dimension {self.num_dimensions}, got {z.shape[-1]}")
        levels = self._levels.to(z.dtype)
        y = ifsq_bound(z, self.alpha)
        if self.edge_bins == "paper":
            return y * ((levels - 1) * (1.0 - self.eps) / 2.0)
        lo, hi = self.level_bounds()
        return torch.clamp(y * (levels / 2.0),
                           lo.to(z.dtype) - 0.5 + 1e-4,
                           hi.to(z.dtype) + 0.5 - 1e-4)

    def extra_repr(self) -> str:
        return (f"levels={self.levels}, codebook_size={self.codebook_size}, "
                f"alpha={self.alpha}, index_order={self.index_order!r}, "
                f"edge_bins={self.edge_bins!r}")


class IFSQDAB(FSQDAB):
    """Distance-Aware Bottleneck over an iFSQ grid.

    Identical to :class:`~dab.fsq_dab.FSQDAB` in every respect -- the same
    Gaussian codes, the same rate-distortion objective, the same E-step and
    closed-form M-step, the same soft/hard modes -- except that the encoder mean
    is bounded with ``2 sigma(alpha z) - 1`` instead of FSQ's ``tanh``-based
    ``f``.

    Args:
      ifsq_alpha: sigmoid slope (default ``1.6``). ``2.0`` reproduces
        ``FSQDAB`` up to FSQ's ``eps`` shrinkage and its even-``L`` handling.
      index_order: ``"big"`` (the paper's convention) by default.
      edge_bins: ``"paper"`` (default) or ``"equal"``; see :class:`IFSQ`. This
        matters most with ``hard=True``, where the level histogram *is* the code
        distribution. In soft mode the learned prior absorbs some of the
        non-uniformity on its own.
      **kwargs: everything :class:`~dab.fsq_dab.FSQDAB` accepts.

    Example:
      >>> from dab import IFSQDAB
      >>> layer = IFSQDAB(in_features=640, levels=[5, 5, 5, 5])
      >>> out = layer(features)
      >>> out.latent, out.distance, out.indices

    See the module-level warning before assuming this beats ``FSQDAB`` for
    uncertainty quantification: iFSQ's argument is about the marginal
    distribution of a Gaussian latent, and DAB does not train one.
    """

    def __init__(self, in_features: int,
                 levels: Union[int, Sequence[int]] = 5,
                 dab_dim: Optional[int] = None,
                 ifsq_alpha: float = IFSQ_ALPHA,
                 fsq_eps: float = 0.0, index_order: str = "big",
                 edge_bins: str = "paper", **kwargs):
        if kwargs.pop("quantizer", None) is not None:
            raise TypeError(
                "IFSQDAB builds its own quantizer; pass `ifsq_alpha` instead, "
                "or use FSQDAB(quantizer=...) directly")
        if isinstance(levels, int):
            if dab_dim is None:
                raise ValueError("pass `dab_dim` when `levels` is an int")
            levels = [levels] * dab_dim
        quantizer = IFSQ(levels, alpha=ifsq_alpha, eps=fsq_eps,
                         index_order=index_order, edge_bins=edge_bins)
        super().__init__(in_features=in_features, levels=list(quantizer.levels),
                         quantizer=quantizer, **kwargs)

    @property
    def ifsq_alpha(self) -> float:
        return self.fsq.alpha


def uniformity_error(alpha: float, num_samples: int = 500_000,
                     seed: int = 0) -> dict:
    """How close ``2 sigma(alpha z) - 1`` is to uniform for ``z ~ N(0, 1)``.

    Reproduces the paper's Figure 2(b) sweep. Returns the Kolmogorov-Smirnov
    statistic and the RMSE between the empirical CDF and the uniform CDF on
    ``[-1, 1]``; both are minimised near ``alpha = 1.6``.

    Args:
      alpha: sigmoid slope to evaluate.
      num_samples: number of standard-normal samples (the paper uses 500k).
      seed: RNG seed.

    Returns:
      ``{"ks": float, "rmse": float}``.
    """
    g = torch.Generator().manual_seed(seed)
    y = ifsq_bound(torch.randn(num_samples, generator=g), alpha)
    y, _ = torch.sort(y)
    empirical = torch.arange(1, num_samples + 1, dtype=torch.float64) / num_samples
    uniform = ((y.double() + 1.0) / 2.0).clamp(0.0, 1.0)
    diff = (empirical - uniform).abs()
    return {"ks": float(diff.max()),
            "rmse": float(torch.sqrt((diff ** 2).mean()))}


def level_histogram(quantizer, num_samples: int = 400_000, seed: int = 0,
                    channel: int = 0) -> torch.Tensor:
    """Marginal level occupancy of ``quantizer`` for a standard-normal latent.

    The empirical version of "does this grid use its bins evenly?". Push
    ``N(0, 1)`` through the quantizer and histogram which level each sample
    lands on. A perfectly used channel gives ``1 / L`` everywhere.

    Args:
      quantizer: an :class:`~dab.fsq.FSQ` or :class:`IFSQ`.
      num_samples: number of standard-normal samples.
      seed: RNG seed.
      channel: which channel to report.

    Returns:
      ``[L_channel]`` occupancy fractions, summing to 1.

    Example:
      >>> from dab.fsq import FSQ
      >>> level_histogram(FSQ([5])).max()          # tanh over-fills the ends
      >>> level_histogram(IFSQ([5])).max()         # paper scaling under-fills them
      >>> level_histogram(IFSQ([5], edge_bins="equal")).max()   # ~1/5 everywhere
    """
    g = torch.Generator().manual_seed(seed)
    d = quantizer.num_dimensions
    z = torch.randn(num_samples, d, generator=g)
    zhat, _ = quantizer(z)
    idx = quantizer.per_channel_indices(zhat)[:, channel]
    L = quantizer.levels[channel]
    return torch.bincount(idx, minlength=L).float() / num_samples


__all__ = ["IFSQ", "IFSQDAB", "ifsq_bound", "uniformity_error", "level_histogram",
           "IFSQ_ALPHA", "IFSQ_ALPHA_KS_OPTIMAL", "TANH_ALPHA"]
