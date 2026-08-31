# coding=utf-8
# Copyright 2023 The Google Research Authors (original JAX implementation)
# Copyright 2026 PyTorch port authors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file for the full text.
"""Finite Scalar Quantization (FSQ).

PyTorch port of the reference JAX implementation accompanying

    Mentzer, Minnen, Agustsson & Tschannen,
    "Finite Scalar Quantization: VQ-VAE Made Simple", ICLR 2024.
    https://arxiv.org/abs/2309.15505
    https://github.com/google-research/google-research/tree/master/fsq

FSQ bounds each latent channel with a ``tanh``-based function ``f`` and rounds
to integers, so channel ``i`` takes one of ``L_i`` values. The codebook is the
Cartesian product of the per-channel value sets and is therefore *implicit*:
``|C| = prod_i L_i`` codewords exist but none are stored or learned. Gradients
flow through the rounding with a straight-through estimator. There is no
commitment loss, no EMA on a codebook, and no codebook parameters at all.

This module is standalone -- it is plain FSQ and knows nothing about DAB. See
:class:`dab.fsq_dab.FSQDAB` for the Distance-Aware Bottleneck built on top of
this grid.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

# Recommended level sets from Table 1 of the FSQ paper, mapping a target
# codebook size |C| to levels L that approximately match it.
RECOMMENDED_LEVELS = {
    2 ** 8: [8, 6, 5],
    2 ** 10: [8, 5, 5, 5],
    2 ** 12: [7, 5, 5, 5, 5],
    2 ** 14: [8, 8, 8, 6, 5],
    2 ** 16: [8, 8, 8, 5, 5, 5],
}


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight-through gradients: ``x + sg(round(x) - x)``."""
    return z + (torch.round(z) - z).detach()


def recommended_levels(codebook_size: int) -> List[int]:
    """Levels approximately matching ``codebook_size``, per Table 1 of the paper.

    Exact entries are returned for the tabulated sizes (2^8, 2^10, 2^12, 2^14,
    2^16). For any other target the nearest tabulated size is used, so the
    result honours the paper's ``L_i >= 5`` heuristic; check ``prod(levels)``
    if you need an exact size.
    """
    if codebook_size in RECOMMENDED_LEVELS:
        return list(RECOMMENDED_LEVELS[codebook_size])
    nearest = min(RECOMMENDED_LEVELS,
                  key=lambda c: abs(math.log(c) - math.log(max(codebook_size, 2))))
    return list(RECOMMENDED_LEVELS[nearest])


class FSQ(nn.Module):
    """Finite Scalar Quantizer.

    Args:
      levels: number of values per channel, ``[L_1, ..., L_d]``. The paper's
        heuristic is ``L_i >= 5`` for every channel; see
        :func:`recommended_levels`.
      eps: shrinks the bounding function slightly so the outermost levels stay
        reachable without the ``tanh`` having to saturate exactly.
      index_order: digit order of the mixed-radix codebook index.
        ``"little"`` (default) is the reference implementation's
        ``basis = [1, L_0, L_0 L_1, ...]``; ``"big"`` is
        ``basis = [prod(L_1..), ..., 1]``, which is what the iFSQ paper uses.
        Both are bijections onto ``[0, prod(L))`` -- they simply number the
        codewords differently, so tokens are not interchangeable between them.

    Attributes:
      num_dimensions: ``d = len(levels)``.
      codebook_size: ``prod(levels)``.
      codebook: ``[codebook_size, d]`` -- the implicit codebook, materialised on
        demand for inspection. Never used during training.

    Example:
      >>> fsq = FSQ([8, 5, 5, 5])          # |C| = 1000
      >>> zhat, indices = fsq(z)           # z: [B, 4]
      >>> zhat.shape, indices.shape        # ([B, 4], [B])
      >>> torch.equal(fsq.indices_to_codes(indices), zhat)
      True

    Note:
      Quantized values are renormalised to roughly ``[-1, 1]`` by dividing by
      ``L_i // 2``, matching the reference implementation. For even ``L_i`` the
      grid is asymmetric (``round`` produces integers, so an even count cannot
      be centred on zero) -- this is the paper's intended behaviour.
    """

    def __init__(self, levels: Sequence[int], eps: float = 1e-3,
                 index_order: str = "little"):
        super().__init__()
        levels = [int(v) for v in levels]
        if not levels or any(v < 2 for v in levels):
            raise ValueError(f"levels must all be >= 2, got {levels}")
        if index_order not in ("little", "big"):
            raise ValueError(
                f"index_order must be 'little' or 'big', got {index_order!r}")
        self.levels = levels
        self.eps = eps
        self.index_order = index_order

        lv = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("_levels", lv)
        self.register_buffer("_half_width", lv // 2)

        # Mixed-radix basis. basis[j] must be the product of the levels of every
        # *less significant* digit, so that `floor(i / basis[j]) % L_j` inverts it.
        weights, acc = [0] * len(levels), 1
        order = range(len(levels)) if index_order == "little" \
            else reversed(range(len(levels)))
        for j in order:
            weights[j] = acc
            acc *= levels[j]
        self.register_buffer("_basis", torch.tensor(weights, dtype=torch.long))

    # ------------------------------------------------------------------ #
    @property
    def num_dimensions(self) -> int:
        """Number of channels expected from the input."""
        return len(self.levels)

    @property
    def codebook_size(self) -> int:
        """``prod(levels)`` -- the size of the implicit codebook."""
        return int(math.prod(self.levels))

    @property
    def codebook(self) -> torch.Tensor:
        """``[codebook_size, d]`` -- every codeword, materialised on demand.

        Only for inspection or nearest-code analysis; quantization never needs
        it. The tensor has ``prod(levels)`` rows, so avoid this for large grids.
        """
        idx = torch.arange(self.codebook_size, device=self._basis.device)
        return self.indices_to_codes(idx)

    @property
    def half_width(self) -> torch.Tensor:
        """``L_i // 2`` per channel -- the renormalisation factor, as a tensor."""
        return self._half_width

    def level_bounds(self):
        """``(lo, hi)`` integer bounds of each channel's level set, as tensors.

        Channel ``i`` rounds into ``{lo_i, ..., hi_i}``, which holds ``L_i``
        values. For odd ``L`` this is symmetric; for even ``L`` the grid is
        shifted down by one (see :meth:`bound`).
        """
        lo = -(self._levels // 2)
        return lo, self._levels - 1 + lo

    def channel_codes(self) -> List[torch.Tensor]:
        """The normalised values each channel can take, one 1-D tensor per channel."""
        return [(torch.arange(L, dtype=torch.float32,
                              device=self._levels.device) - (L // 2)) / (L // 2)
                for L in self.levels]

    # ------------------------------------------------------------------ #
    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound ``z`` (``[..., d]``) into the rounding range of each channel.

        ``f(z) = tanh(z + shift) * half_l - offset`` with
        ``half_l = (L - 1) * (1 - eps) / 2``, ``offset = 0.5`` for even ``L``
        (an even number of integers cannot be centred on zero) and ``0``
        otherwise, and ``shift = tan(offset / half_l)`` so that ``z = 0`` still
        maps to the middle of the grid.
        """
        if z.shape[-1] != self.num_dimensions:
            raise ValueError(
                f"expected last dimension {self.num_dimensions}, got {z.shape[-1]}")
        levels = self._levels.to(z.dtype)
        half_l = (levels - 1) * (1.0 - self.eps) / 2.0
        offset = torch.where(levels % 2 == 1, torch.zeros_like(levels),
                             torch.full_like(levels, 0.5))
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize ``z``; returns ``zhat`` of the same shape, in ``[-1, 1]``.

        ``round_ste(bound(z))`` followed by the reference renormalisation
        ``/ (L // 2)``. Gradients pass straight through the rounding.
        """
        quantized = round_ste(self.bound(z))
        return quantized / self._half_width

    def bound_normalized(self, z: torch.Tensor) -> torch.Tensor:
        """:meth:`bound` renormalised to ``[-1, 1]``, *without* rounding.

        The continuous counterpart of :meth:`quantize`, and the space the
        codebook grid lives in. :class:`~dab.fsq_dab.FSQDAB` puts the encoder
        mean here so that it and the codes are directly comparable.
        """
        return self.bound(z) / self._half_width.to(z.dtype)

    def quantize_normalized(self, v: torch.Tensor) -> torch.Tensor:
        """Round a value that is *already* in normalised grid space, with STE.

        Unlike :meth:`quantize` this does not bound first, so ``v`` may lie
        outside ``[-1, 1]`` (for instance after adding encoder noise). The
        result is clamped into the valid level range, and gradients pass
        straight through.
        """
        hw = self._half_width.to(v.dtype)
        lo, hi = self.level_bounds()
        q = torch.clamp(torch.round(v * hw), lo.to(v.dtype), hi.to(v.dtype)) / hw
        return v + (q - v).detach()

    def normalized_to_index(self, v: torch.Tensor) -> torch.Tensor:
        """Nearest level index in ``[0, L_i)`` for values in normalised space.

        Accepts values outside ``[-1, 1]``; they clamp to the outermost level.
        """
        hw = self._half_width.to(v.dtype)
        lo, hi = self.level_bounds()
        return (torch.clamp(torch.round(v * hw), lo.to(v.dtype), hi.to(v.dtype))
                - lo.to(v.dtype)).long()

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        return zhat_normalized * self._half_width + self._half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        return (zhat - self._half_width) / self._half_width

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """Map normalised codewords ``[..., d]`` to codebook indices ``[...]``."""
        if zhat.shape[-1] != self.num_dimensions:
            raise ValueError(
                f"expected last dimension {self.num_dimensions}, got {zhat.shape[-1]}")
        levels = self._scale_and_shift(zhat).round().long()
        return (levels * self._basis).sum(dim=-1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`codes_to_indices`: ``[...]`` -> ``[..., d]``."""
        idx = indices.unsqueeze(-1)
        levels = torch.div(idx, self._basis, rounding_mode="floor") \
            % self._levels.long()
        return self._scale_and_shift_inverse(levels.to(self._levels.dtype))

    def per_channel_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """Per-channel level index in ``[0, L_i)``; shape ``[..., d]``."""
        return self._scale_and_shift(zhat).round().long()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize ``z``. Returns ``(zhat [..., d], indices [...])``."""
        zhat = self.quantize(z)
        return zhat, self.codes_to_indices(zhat)

    def extra_repr(self) -> str:
        return (f"levels={self.levels}, codebook_size={self.codebook_size}, "
                f"eps={self.eps}, index_order={self.index_order!r}")


__all__ = ["FSQ", "round_ste", "recommended_levels", "RECOMMENDED_LEVELS"]
