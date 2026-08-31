# coding=utf-8
# Copyright 2026 PyTorch port authors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file for the full text.
"""Distance-Aware Bottleneck over a Finite Scalar Quantization codebook.

An alternative codebook for DAB. Instead of the reference implementation's ``K``
explicit multivariate Gaussian centroids, the codes are the Cartesian product of
per-coordinate scalar levels, laid out on the FSQ grid of

    Mentzer, Minnen, Agustsson & Tschannen,
    "Finite Scalar Quantization: VQ-VAE Made Simple", ICLR 2024.
    https://arxiv.org/abs/2309.15505

Each product code is still a *Gaussian* (that is what keeps it DAB and not
plain FSQ): coordinate ``j``, level ``l`` carries a scalar
``N(g_jl, s^2_jl)`` where ``g_jl`` is the fixed FSQ grid point and ``s^2_jl``
is fitted by the same closed-form M-step the reference DAB uses.

Why this is exact, not an approximation
---------------------------------------
The encoder covariance is diagonal and the prior factorises over coordinates, so
for a product code ``c = (l_1, ..., l_d)`` with prior ``pi_c = prod_j pi_{j,l_j}``:

* ``KL(encoder || code_c) = sum_j KL_j(l_j)`` -- the KL is additive;
* the Gibbs assignment ``a_c ∝ pi_c exp(-tau KL_c)`` therefore *factorises* into
  ``a_c = prod_j a_{j,l_j}``;
* the expected distortion ``sum_c a_c KL_c`` collapses to
  ``sum_j sum_l a_{j,l} KL_j(l)``.

So the codebook distance computed coordinate-wise here is **identical** to the
expected KL over all ``prod_j L_j`` product codes, at ``O(sum_j L_j)`` cost
instead of ``O(prod_j L_j)``. ``tests/test_fsq_dab.py`` checks this against a
brute-force enumeration.

Trade-offs versus the reference codebook
----------------------------------------
+-------------------+--------------------------------+---------------------------+
|                   | ``NormalFullCovarianceDAB``    | ``FSQDAB``                |
+===================+================================+===========================+
| codes             | ``K`` learned Gaussians        | ``prod_j L_j`` implicit   |
| code parameters   | ``K x d`` means (+ covariance) | none (grid is fixed)      |
| distance cost     | ``O(B K d^2)``                 | ``O(B sum_j L_j)``        |
| code collapse     | possible (dead centroids)      | impossible by construction|
| discrete tokens   | no                             | yes (``out.indices``)     |
| far-OOD distance  | grows without bound            | saturates -- see below    |
+-------------------+--------------------------------+---------------------------+

.. warning::
   **Saturation.** FSQ bounds the encoder mean with a ``tanh``, so an input
   pushed far outside the training manifold lands on the edge of the grid and
   stops moving. The codebook distance then stops growing, and very-far-OOD
   inputs become indistinguishable from moderately-OOD ones. The encoder
   *variance* is unbounded and still responds, but if far-OOD ranking is your
   main goal prefer ``bound=None`` (an unbounded integer lattice; discrete
   indices are then unbounded too) or the reference DAB codebook. The
   ``saturated_frac`` diagnostic reports the fraction of coordinates pinned to
   the outermost levels.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import all_gather_cat, global_batch_size
from .fsq import FSQ, round_ste
from .layers import DABLayer, DABOutput

_TINY = 1e-30


class FSQDAB(DABLayer):
    """DAB layer whose codebook is an FSQ product grid.

    Args:
      in_features: size of the incoming feature vector.
      levels: ``[L_1, ..., L_d]`` values per coordinate, or a single int paired
        with ``dab_dim`` for a uniform grid. The FSQ paper's heuristic is
        ``L_i >= 5``; :func:`dab.fsq.recommended_levels` gives the paper's
        tabulated sets for a target codebook size.
      dab_dim: only needed when ``levels`` is an int.
      dab_tau: temperature multiplying the distance in the E-step softmax.
      momentum: moving-average momentum for the code variances and priors.
      hard: ``False`` (default) keeps DAB's soft Gibbs assignment and a
        continuous latent; ``True`` uses FSQ's straight-through rounding, so the
        prediction head sees the quantized codeword and the distortion is the KL
        to the single selected code.
      bound: ``"fsq"`` (default) applies the paper's bounding function before
        rounding; ``None`` leaves the mean unbounded on an integer lattice,
        which keeps the distance unsaturated at the cost of a codebook that is
        no longer finite.
      code_scale_mode: how the per-level variances are obtained.
        ``"ema"`` (default) fits them with DAB's closed-form M-step -- zero
        codebook parameters; ``"learned"`` makes them a trainable parameter
        updated in the RDFC phase; ``"fixed"`` freezes them at
        ``code_scale_init``.
      code_scale_init: initial per-level standard deviation. ``None`` (default)
        uses half the grid spacing of each coordinate.
      fsq_eps: passed to the quantizer's bounding function.
      quantizer: an already-built :class:`~dab.fsq.FSQ` (or subclass) to use
        instead of constructing one. This is how :class:`~dab.ifsq.IFSQDAB`
        swaps in the iFSQ bounding map; you can also pass a plain ``FSQ`` with a
        non-default ``index_order``.
      activation, use_bias, kernel_initializer, var_shift, var_floor,
      dirichlet_alpha, generator: as in :class:`~dab.layers.DABLayer`.

    .. note::
       ``var_shift`` defaults to the reference DAB value of ``5.0``, which
       starts the encoder standard deviation near ``0.007``. The grid here is
       normalised to ``[-1, 1]``, where the spacing is ``1 / (L // 2)``, so that
       initial noise is very small relative to the grid and the early distance
       is dominated by the log-determinant term. This is harmless -- the scale
       is learned -- but lower ``var_shift`` if you want the encoder to start
       closer to the grid resolution.

    Shape:
      input ``[B, in_features]`` -> :class:`~dab.layers.DABOutput` with
      ``latent [B, d]``, ``distance [B]``, ``indices [B]`` (codebook index) and
      ``per_coord_indices [B, d]``.

    Example:
      >>> from dab import FSQDAB
      >>> from dab.fsq import recommended_levels
      >>> layer = FSQDAB(in_features=640, levels=recommended_levels(1024))
      >>> out = layer(features)
      >>> out.latent, out.distance, out.indices     # [B,4], [B], [B] in [0,1000)
    """

    def __init__(self, in_features: int,
                 levels: Union[int, Sequence[int]] = 5,
                 dab_dim: Optional[int] = None,
                 dab_tau: float = 1.0, momentum: float = 0.99,
                 hard: bool = False, bound: Optional[str] = "fsq",
                 code_scale_mode: str = "ema",
                 code_scale_init: Optional[float] = None,
                 fsq_eps: float = 1e-3, quantizer: Optional[FSQ] = None,
                 activation: Union[str, Callable, None] = "relu",
                 use_bias: bool = False,
                 kernel_initializer: str = "glorot_uniform",
                 var_shift: float = 5.0, var_floor: float = 1e-5,
                 dirichlet_alpha: float = 5.0,
                 generator: Optional[torch.Generator] = None):
        if quantizer is not None:
            levels = quantizer.levels
        if isinstance(levels, int):
            if dab_dim is None:
                raise ValueError("pass `dab_dim` when `levels` is an int")
            levels = [levels] * dab_dim
        levels = [int(v) for v in levels]
        if dab_dim is not None and dab_dim != len(levels):
            raise ValueError(
                f"dab_dim={dab_dim} disagrees with len(levels)={len(levels)}")
        d = len(levels)
        if bound not in ("fsq", None):
            raise ValueError(f"bound must be 'fsq' or None, got {bound!r}")
        if code_scale_mode not in ("ema", "learned", "fixed"):
            raise ValueError(f"unknown code_scale_mode {code_scale_mode!r}")

        super().__init__(in_features=in_features, units=2 * d, dab_dim=d,
                         dab_tau=dab_tau, momentum=momentum,
                         activation=activation, use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         var_shift=var_shift, var_floor=var_floor,
                         dirichlet_alpha=dirichlet_alpha, generator=generator)
        self.levels = levels
        self.Lmax = max(levels)
        self.hard = hard
        self.bound_mode = bound
        self.code_scale_mode = code_scale_mode

        # The grid itself. Kept as a submodule so the bounding function, the
        # rounding and the index arithmetic stay in one place and match their
        # reference implementation exactly.
        self.fsq = quantizer if quantizer is not None else FSQ(levels, eps=fsq_eps)

        # Valid-level mask for ragged grids ([8, 6, 5] etc.): [d, Lmax]
        mask = torch.zeros(d, self.Lmax, dtype=torch.bool)
        for j, L in enumerate(levels):
            mask[j, :L] = True
        self.register_buffer("level_mask", mask)

        # Fixed code means: the normalised FSQ grid points, padded to Lmax.
        means = torch.zeros(d, self.Lmax)
        for j, chan in enumerate(self.fsq.channel_codes()):
            means[j, :levels[j]] = chan
        self.register_buffer("centroid_means", means)

        # Per-coordinate grid spacing (in normalised space), used for the
        # default code scale.
        spacing = 1.0 / self.fsq.half_width
        self.register_buffer("grid_spacing", spacing)
        if code_scale_init is None:
            init_std = 0.5 * spacing.unsqueeze(-1).expand(d, self.Lmax).clone()
        else:
            init_std = torch.full((d, self.Lmax), float(code_scale_init))
        init_var = (init_std ** 2).masked_fill(~mask, 1.0)

        # Codebook variances. Buffer + moving average, exactly like the reference
        # DAB codebook covariance; or a trainable parameter in "learned" mode.
        self.register_buffer("centroid_covariance", init_var.clone())
        self.register_buffer("centroid_covariance_mavg",
                             torch.zeros(d, self.Lmax))
        if code_scale_mode == "learned":
            raw = torch.log(torch.expm1(init_std.clamp_min(1e-4)))
            self.centroid_scale_raw = nn.Parameter(raw)

        # Per-coordinate priors over levels, uniform over the valid ones.
        uniform = self._uniform_prior(torch.zeros(d, self.Lmax))
        self.register_buffer("centroid_probs", uniform.clone())
        self.register_buffer("centroid_probs_mavg", uniform.clone())

    # ------------------------------------------------------------------ #
    @property
    def codebook_size(self) -> int:
        """``prod_j L_j`` -- the size of the implicit product codebook."""
        return self.fsq.codebook_size

    @property
    def num_codebook_parameters(self) -> int:
        """Learned codebook parameters (0 unless ``code_scale_mode='learned'``)."""
        return (self.centroid_scale_raw.numel()
                if self.code_scale_mode == "learned" else 0)

    def _uniform_prior(self, like: torch.Tensor) -> torch.Tensor:
        p = torch.zeros_like(like)
        for j, L in enumerate(self.levels):
            p[j, :L] = 1.0 / L
        return p

    def _code_variance(self) -> torch.Tensor:
        """Per-level code variances ``[d, Lmax]``, valid entries only."""
        if self.code_scale_mode == "learned":
            std = F.softplus(self.centroid_scale_raw) + self.var_floor
            var = std * std
        else:
            var = self.centroid_covariance
        # Invalid levels get variance 1 so the KL stays finite; the mask below
        # removes them from the assignment anyway.
        return var.masked_fill(~self.level_mask, 1.0)

    # ------------------------------------------------------------------ #
    def _bound(self, mu_raw: torch.Tensor) -> torch.Tensor:
        """Encoder mean in normalised grid space (``[-1, 1]`` for ``bound='fsq'``).

        Delegates to the quantizer, so a subclass that changes the bounding map
        (:class:`~dab.ifsq.IFSQDAB`) is picked up automatically.
        """
        if self.bound_mode is None:
            return mu_raw
        return self.fsq.bound_normalized(mu_raw)

    # ------------------------------------------------------------------ #
    def forward(self, inputs: torch.Tensor,
                training: Optional[bool] = None) -> DABOutput:
        if training is None:
            training = self.training

        params = self._call(inputs)
        mu_raw, raw_scale = params.split([self.dab_dim, self.dab_dim], dim=-1)
        mu = self._bound(mu_raw)                            # [B, d]
        scale = self._encoder_scale(raw_scale)              # std
        variance = scale * scale

        code_var = self._code_variance()                    # [d, Lmax]
        code_mean = self.centroid_means                     # [d, Lmax]

        # Per-coordinate, per-level KL(encoder_j || code_jl):  [B, d, Lmax]
        mu_e = mu.unsqueeze(-1)
        var_e = variance.unsqueeze(-1)
        m = code_mean.unsqueeze(0)
        s2 = code_var.unsqueeze(0)
        distances = 0.5 * ((var_e + (mu_e - m) ** 2) / s2 - 1.0
                           + torch.log(s2) - torch.log(var_e))
        mask = self.level_mask.unsqueeze(0)
        distances = distances.masked_fill(~mask, 0.0)

        if self.hard:
            # FSQ semantics: the assignment is deterministic nearest-level
            # rounding, so there is no Gibbs posterior and no `initialized`
            # bootstrap -- the distortion is the KL to that one code.
            idx = self.fsq.normalized_to_index(mu)          # [B, d]
            assignment = F.one_hot(idx, self.Lmax).to(distances.dtype)
        else:
            assignment = self._e_step(distances, mask)      # [B, d, Lmax]
            idx = self.fsq.normalized_to_index(mu.detach())
        distance = (assignment * distances).sum(dim=(-2, -1))   # [B]

        # M-step: accumulate the moving averages.
        self._update_centroid_probs(training, assignment)
        self._update_codebook_covariance(training, assignment, mu.detach(),
                                         variance.detach())

        # Latent. Soft mode keeps DAB's continuous bottleneck; hard mode hands
        # the head the quantized codeword, as FSQ does.
        if training:
            noisy = mu + scale * torch.randn_like(mu)
            latent = self.fsq.quantize_normalized(noisy) if self.hard else noisy
        else:
            latent = self.fsq.quantize_normalized(mu) if self.hard else mu

        with torch.no_grad():
            codes = self.centroid_means.unsqueeze(0).expand(
                idx.shape[0], -1, -1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            indices = self.fsq.codes_to_indices(codes)      # [B]
            diagnostics = self._diagnostics(assignment, idx, mu_raw)

        return DABOutput(
            latent=latent, distance=distance, mean=mu, variance=variance,
            distances_from_centroids=distances, assignment=assignment,
            indices=indices, per_coord_indices=idx, pre_bound_mean=mu_raw,
            diagnostics=diagnostics)

    def _e_step(self, distances: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Per-coordinate Gibbs responsibilities over levels, always detached.

        Identical in form to the reference DAB E-step, applied along the level
        axis of each coordinate. Because the prior factorises, the product of
        these per-coordinate assignments *is* the assignment over the full
        product codebook.
        """
        neg = torch.finfo(distances.dtype).min / 4
        if not bool(self.initialized):
            uniform = mask.to(distances.dtype).expand_as(distances).clone()
            return uniform / uniform.sum(-1, keepdim=True)
        log_prior = torch.log(self.centroid_probs.clamp_min(_TINY)).unsqueeze(0)
        logits = (log_prior - self.dab_tau * distances).detach()
        logits = logits.masked_fill(~mask, neg)
        return torch.softmax(logits, dim=-1)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def set_codebook_covariance(self) -> None:
        """Commit the code-variance moving average.

        A no-op unless ``code_scale_mode == "ema"`` -- in the other modes the
        variances are either frozen or trained by the codebook optimiser.
        """
        if self.code_scale_mode != "ema":
            return
        var = self.centroid_covariance_mavg.clamp_min(self.var_floor ** 2)
        self.centroid_covariance.copy_(var.masked_fill(~self.level_mask, 1.0))

    @torch.no_grad()
    def _calculate_codebook_covariance(self, cond_centroid_probs: torch.Tensor,
                                       mu: torch.Tensor,
                                       covariance: torch.Tensor
                                       ) -> torch.Tensor:
        """Optimal per-level scalar variance on the current batch.

        The reference DAB covariance M-step (Davis & Dhillon eq. 9, Dirichlet(5)
        smoothed) specialised to scalars: a responsibility-weighted average of
        the encoder variances plus the squared distance from the grid point.
        """
        n = global_batch_size(cond_centroid_probs.shape[0], mu.device, mu.dtype)
        mu = all_gather_cat(mu)                              # [N, d]
        covariance = all_gather_cat(covariance)              # [N, d]
        probs = all_gather_cat(cond_centroid_probs)          # [N, d, Lmax]

        w = self._dirichlet_weights(probs, n)                # [N, d, Lmax]
        diff = mu.unsqueeze(-1) - self.centroid_means.unsqueeze(0)
        return (w * (covariance.unsqueeze(-1) + diff * diff)).sum(0)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _diagnostics(self, assignment, idx, mu_raw) -> Dict[str, torch.Tensor]:
        occ = assignment.mean(0).masked_fill(~self.level_mask, 0.0)  # [d, Lmax]
        p = occ / occ.sum(-1, keepdim=True).clamp_min(_TINY)
        entropy = -(p * p.clamp_min(_TINY).log()).sum(-1)            # [d]
        valid = self.level_mask.sum()
        unused = ((occ < 1e-4) & self.level_mask).sum() / valid
        # Coordinates pinned to the outermost level: FSQ's tanh has saturated
        # there, so the distance can no longer grow with distance from the data.
        lo, hi = self.fsq.level_bounds()
        edge = (idx == 0) | (idx == (hi - lo).long())
        return {
            "level_occupancy": occ,
            "level_entropy_mean": entropy.mean(),
            "level_perplexity_mean": entropy.exp().mean(),
            "unused_level_frac": unused.to(occ.dtype),
            "saturated_frac": edge.to(occ.dtype).mean(),
            # Sum of the per-coordinate entropies: the entropy of the product of
            # the marginals, i.e. an upper bound on the joint code entropy.
            "product_code_entropy": entropy.sum(),
        }

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, levels={self.levels}, "
                f"codebook_size={self.codebook_size}, dab_tau={self.dab_tau}, "
                f"momentum={self.momentum}, hard={self.hard}, "
                f"bound={self.bound_mode!r}, "
                f"code_scale_mode={self.code_scale_mode!r}")


__all__ = ["FSQDAB"]
