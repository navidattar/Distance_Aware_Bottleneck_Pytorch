# coding=utf-8
# Copyright 2024 Ifigeneia Apostolopoulou (original TensorFlow implementation)
# Copyright 2026 PyTorch port authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dense layers with a Distance-Aware Bottleneck (DAB).

PyTorch port of ``layers/dab_normal.py``, ``layers/dab_normal_diag_covariance.py``
and ``layers/dab_normal_full_covariance.py`` from

    Apostolopoulou et al., "A Rate-Distortion View of Uncertainty Quantification",
    ICML 2024. https://arxiv.org/abs/2406.10775
    https://github.com/ifiaposto/Distance_Aware_Bottleneck

Attribute names, buffer semantics and the public method surface deliberately
mirror the TensorFlow source so the two can be diffed side by side.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import (all_gather_cat, all_reduce_sum, global_batch_size,
                         kl_normal_diag, kl_normal_full,
                         symmetrized_fill_triangular)

_ACTIVATIONS = {
    "relu": F.relu,
    "elu": F.elu,
    "gelu": F.gelu,
    "tanh": torch.tanh,
    "silu": F.silu,
    "linear": lambda x: x,
    None: lambda x: x,
}


@dataclass
class DABOutput:
    """Everything a DAB layer computes for one batch.

    The reference implementation returns ``concat([latent, distance], -1)``;
    :meth:`as_tensor` reproduces that tensor exactly, while the named fields
    make the layer pleasant to use from PyTorch code.

    Attributes:
      latent:      ``[B, dab_dim]`` features handed to the decoder. A sample from
                   the encoder Gaussian in training mode, the mean at eval time.
      distance:    ``[B]`` distance from the codebook, i.e. the DAB uncertainty
                   score. Higher means further from the training manifold.
      mean:        ``[B, dab_dim]`` encoder mean.
      variance:    ``[B, dab_dim]`` encoder diagonal variance (diag layer), or the
                   diagonal of the full covariance (full layer).
      covariance:  ``[B, dab_dim, dab_dim]`` encoder covariance (full layer only).
      distances_from_centroids: ``[B, K]`` per-centroid ``KL(encoder || centroid)``.
      assignment:  ``[B, K]`` soft E-step responsibilities (always detached).
    """

    latent: torch.Tensor
    distance: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor
    distances_from_centroids: torch.Tensor
    assignment: torch.Tensor
    covariance: Optional[torch.Tensor] = None

    def as_tensor(self) -> torch.Tensor:
        """``concat([latent, distance[:, None]], -1)`` -- the TF layer output."""
        return torch.cat([self.latent, self.distance.unsqueeze(-1)], dim=-1)

    @property
    def uncertainty(self) -> torch.Tensor:
        """Alias for :attr:`distance`."""
        return self.distance


class _NormalDAB(nn.Module):
    """Abstract dense layer with a Distance-Aware Bottleneck.

    Normal distributions are used for both the encoder and the codebook entries.
    Subclasses implement :meth:`forward`, :meth:`set_codebook_covariance` and
    :meth:`_calculate_codebook_covariance`.

    Args:
      in_features: size of the incoming feature vector.
      units: number of encoder outputs (means + covariance parameters).
      dab_dim: dimension ``d`` of the latent features.
      codebook_size: number of centroids ``K``.
      dab_tau: temperature multiplying the distance in the E-step softmax.
      momentum: momentum of the moving averages used to accumulate the codebook
        covariances and prior probabilities in a batched manner. Use ``0.0`` to
        take the current batch's estimate outright (the synthetic-regression
        demo of the reference code does exactly that).
      activation: non-linearity applied to the incoming features *before* the
        encoder's linear layer. ``"relu"`` in the reference layer, ``None`` in
        the synthetic demo.
      use_bias: whether the encoder's linear layer has a bias.
      kernel_initializer: ``"glorot_uniform"`` (the Keras ``Dense`` default and
        therefore the layer default) or ``"he_normal"`` (what the reference
        WideResNet/ResNet-50 models pass in).
      var_shift, var_floor: encoder scale parameterisation,
        ``scale = softplus(raw - var_shift) + var_floor``. The shift keeps the
        singular values small early on and eases convergence (5.0 and 1e-5 in
        the reference code).
      generator: optional ``torch.Generator`` for reproducible initialisation.

    Shape:
      input ``[B, in_features]`` -> :class:`DABOutput` whose ``as_tensor()`` is
      ``[B, dab_dim + 1]``.
    """

    def __init__(
        self,
        in_features: int,
        units: int,
        dab_dim: int,
        codebook_size: int,
        dab_tau: float = 1.0,
        momentum: float = 0.99,
        activation: Union[str, Callable, None] = "relu",
        use_bias: bool = False,
        kernel_initializer: str = "glorot_uniform",
        var_shift: float = 5.0,
        var_floor: float = 1e-5,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.units = units
        self.dab_dim = dab_dim
        self.codebook_size = codebook_size
        self.dab_tau = dab_tau
        self.momentum = momentum
        self.var_shift = var_shift
        self.var_floor = var_floor

        if callable(activation):
            self.dab_activation = activation
        elif activation in _ACTIVATIONS:
            self.dab_activation = _ACTIVATIONS[activation]
        else:
            raise ValueError(f"unknown activation {activation!r}")
        self._activation_name = activation if not callable(activation) else "custom"

        # Encoder: a plain dense layer producing the Gaussian parameters.
        self.dense = nn.Linear(in_features, units, bias=use_bias)
        self._init_kernel(kernel_initializer, generator)

        # Centroid means -- the only *trainable* codebook parameter. The name
        # contains "centroid" so that the RDFC parameter split works (see
        # ``dab.trainer.codebook_parameters``), matching the reference code's
        # ``if "centroid" in var.name`` filter.
        means = torch.empty(codebook_size, dab_dim)
        with torch.no_grad():
            means.normal_(mean=0.0, std=0.1, generator=generator)
        self.centroid_means = nn.Parameter(means)

        # Prior centroid probabilities, two copies:
        #  * ``centroid_probs``      -- committed, used by the E-step this epoch;
        #  * ``centroid_probs_mavg`` -- moving average accumulated for the next.
        uniform = torch.full((codebook_size,), 1.0 / codebook_size)
        self.register_buffer("centroid_probs", uniform.clone())
        self.register_buffer("centroid_probs_mavg", uniform.clone())

        # False until the first RDFC phase completes; while False, datapoints are
        # assigned to centroids uniformly at random.
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    # ------------------------------------------------------------------ #
    def _init_kernel(self, kernel_initializer: str,
                     generator: Optional[torch.Generator]) -> None:
        w = self.dense.weight            # PyTorch stores [out, in]
        with torch.no_grad():
            if kernel_initializer == "glorot_uniform":
                limit = (6.0 / (self.in_features + self.units)) ** 0.5
                w.uniform_(-limit, limit, generator=generator)
            elif kernel_initializer == "he_normal":
                std = (2.0 / self.in_features) ** 0.5
                w.normal_(0.0, std, generator=generator)
            else:
                raise ValueError(
                    f"unknown kernel_initializer {kernel_initializer!r}")
            if self.dense.bias is not None:
                self.dense.bias.zero_()

    def _call(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the encoder parameters (activation, then the dense layer)."""
        return self.dense(self.dab_activation(inputs))

    def _encoder_scale(self, raw: torch.Tensor) -> torch.Tensor:
        """``softplus(raw - var_shift) + var_floor`` -- a standard deviation."""
        return F.softplus(raw - self.var_shift) + self.var_floor

    def forward(self, inputs: torch.Tensor,
                training: Optional[bool] = None) -> DABOutput:
        raise NotImplementedError(
            f"Normal DAB '{type(self).__name__}' must override `forward(...)`.")

    # ------------------------------------------------------------------ #
    #  E-step shared by both covariance variants
    # ------------------------------------------------------------------ #
    def _e_step(self, distances_from_centroids: torch.Tensor) -> torch.Tensor:
        """Conditional assignment probabilities ``p(h | x_i)``.

        ``softmax(log pi - tau * KL)`` over the codebook axis, always detached:
        the responsibilities are treated as constants by the network's
        optimiser, exactly as ``tf.stop_gradient`` does in the reference code.
        Before the first RDFC phase (``initialized == False``) the assignment is
        uniform.
        """
        if not bool(self.initialized):
            return torch.full_like(distances_from_centroids,
                                   1.0 / self.codebook_size)
        log_centroid_probs = torch.log(self.centroid_probs).reshape(
            1, self.codebook_size)
        logits = (log_centroid_probs
                  - self.dab_tau * distances_from_centroids).detach()
        return torch.softmax(logits, dim=-1)

    # ------------------------------------------------------------------ #
    #  Utility functions for the centroid probabilities
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _update_centroid_probs(self, training: bool,
                               cond_centroid_probs: torch.Tensor) -> None:
        """Fold this batch's estimate into ``centroid_probs_mavg``."""
        if not training:
            return
        new = self._calculate_centroid_probs(cond_centroid_probs)
        self.centroid_probs_mavg.mul_(self.momentum).add_(
            new * (1.0 - self.momentum))

    @torch.no_grad()
    def reset_centroid_probs(self) -> None:
        """Reset the prior moving average to uniform.

        Called at the start of the prior sub-phase of the RDFC M-step.
        """
        self.centroid_probs_mavg.fill_(1.0 / self.codebook_size)

    @torch.no_grad()
    def set_centroid_probs(self) -> None:
        """Commit the moving average so the next epoch's E-step uses it.

        Called at the end of the prior sub-phase of the RDFC M-step.
        """
        self.centroid_probs.copy_(self.centroid_probs_mavg)

    @torch.no_grad()
    def _calculate_centroid_probs(
            self, cond_centroid_probs: torch.Tensor) -> torch.Tensor:
        r"""Prior probabilities ``pi(h)`` from this batch's responsibilities.

        Equation for :math:`\pi(h)` with equal datapoint weights (page 1731 of
        Banerjee et al., *Clustering with Bregman divergences*, JMLR 2005):
        the mean responsibility over the (global) batch. ReLU guards against
        floating-point round-off making the mean negative.
        """
        batch = cond_centroid_probs.shape[0]
        local_sum = cond_centroid_probs.sum(dim=0)
        total = all_reduce_sum(local_sum)
        n = global_batch_size(batch, cond_centroid_probs.device,
                              cond_centroid_probs.dtype)
        return F.relu(total / n)

    # ------------------------------------------------------------------ #
    #  Utility functions for the codebook covariances
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def reset_codebook_covariance(self) -> None:
        """Zero the codebook-covariance moving average.

        Called at the start of the covariance sub-phase of the RDFC M-step.
        """
        self.centroid_covariance_mavg.zero_()

    def set_codebook_covariance(self) -> None:
        raise NotImplementedError(
            f"Normal DAB '{type(self).__name__}' must override "
            "`set_codebook_covariance(...)`.")

    @torch.no_grad()
    def _update_codebook_covariance(self, training: bool,
                                    cond_centroid_probs: torch.Tensor,
                                    mu: torch.Tensor,
                                    covariance: torch.Tensor) -> None:
        """Fold this batch's covariance estimate into the moving average."""
        if not training:
            return
        new = self._calculate_codebook_covariance(cond_centroid_probs, mu,
                                                  covariance)
        self.centroid_covariance_mavg.mul_(self.momentum).add_(
            new * (1.0 - self.momentum))

    @torch.no_grad()
    def _calculate_codebook_covariance(self, cond_centroid_probs, mu,
                                       covariance):
        raise NotImplementedError(
            f"Normal DAB '{type(self).__name__}' must override "
            "`_calculate_codebook_covariance(...)`.")

    @torch.no_grad()
    def _dirichlet_weights(self, cond_centroid_probs: torch.Tensor,
                           n: torch.Tensor) -> torch.Tensor:
        r"""Per-datapoint contribution weights for the codebook covariance.

        ``(a_ik + 5) / (sum_i a_ik + 5 N)``: a Dirichlet prior with
        :math:`a_k = 5` that avoids over-concentration and eases training.
        """
        return (cond_centroid_probs + 5.0) / (
            cond_centroid_probs.sum(dim=0, keepdim=True) + 5.0 * n)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def mark_initialized(self, value: bool = True) -> None:
        """Flag the layer as initialised (stop using uniform assignments)."""
        self.initialized.fill_(value)

    def reset_codebook(self) -> None:
        """``reset_codebook_covariance()`` + ``reset_centroid_probs()``."""
        self.reset_codebook_covariance()
        self.reset_centroid_probs()

    def set_codebook(self) -> None:
        """``set_codebook_covariance()`` + ``set_centroid_probs()``."""
        self.set_codebook_covariance()
        self.set_centroid_probs()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, dab_dim={self.dab_dim}, "
                f"codebook_size={self.codebook_size}, dab_tau={self.dab_tau}, "
                f"momentum={self.momentum}, activation={self._activation_name}")


class NormalDiagCovarianceDAB(_NormalDAB):
    """DAB dense layer with diagonal-covariance Normal encoder and codebook.

    The distance from the codebook is the expected
    ``KL(encoder || centroid)`` under the E-step responsibilities.

    This is the variant used by the reference ResNet-50 / ImageNet model.
    See :class:`_NormalDAB` for the arguments.

    Example:
      >>> layer = NormalDiagCovarianceDAB(in_features=640, dab_dim=8,
      ...                                 codebook_size=10, dab_tau=1.0)
      >>> out = layer(features)              # features: [B, 640]
      >>> logits = head(out.latent)          # out.latent: [B, 8]
      >>> uncertainty = out.distance         # [B]
    """

    def __init__(self, in_features: int, dab_dim: int, codebook_size: int,
                 dab_tau: float = 1.0, momentum: float = 0.99,
                 activation: Union[str, Callable, None] = "relu",
                 use_bias: bool = False,
                 kernel_initializer: str = "glorot_uniform",
                 var_shift: float = 5.0, var_floor: float = 1e-5,
                 covariance_floor: float = 0.0,
                 generator: Optional[torch.Generator] = None):
        super().__init__(
            in_features=in_features,
            # dab_dim means + dab_dim diagonal covariance entries
            units=dab_dim + dab_dim,
            dab_dim=dab_dim, codebook_size=codebook_size, dab_tau=dab_tau,
            momentum=momentum, activation=activation, use_bias=use_bias,
            kernel_initializer=kernel_initializer, var_shift=var_shift,
            var_floor=var_floor, generator=generator)
        self.covariance_floor = covariance_floor

        # Moving average supporting the batched RDFC covariance update.
        self.register_buffer("centroid_covariance_mavg",
                             torch.zeros(codebook_size, dab_dim))
        # Codebook variances used for inference; initialised to the identity.
        self.register_buffer("centroid_covariance",
                             torch.ones(codebook_size, dab_dim))

    def forward(self, inputs: torch.Tensor,
                training: Optional[bool] = None) -> DABOutput:
        if training is None:
            training = self.training

        params = self._call(inputs)
        mu, perturb_diag = params.split([self.dab_dim, self.dab_dim], dim=-1)

        # Shift keeps small standard deviations and eases convergence.
        perturb_diag = self._encoder_scale(perturb_diag)      # std
        variance = perturb_diag * perturb_diag

        # distances_from_centroids[i, j]: KL of datapoint i's encoder from
        # centroid j. ``centroid_covariance`` stores variances (the reference
        # code passes ``sqrt(centroid_covariance)`` as ``scale_diag``).
        distances_from_centroids = kl_normal_diag(
            mu.unsqueeze(1), variance.unsqueeze(1),
            self.centroid_means.unsqueeze(0),
            self.centroid_covariance.unsqueeze(0))            # [B, K]

        cond_centroid_probs = self._e_step(distances_from_centroids)

        # The distance from the codebook is the expected distance from centroids.
        distances_from_codebook = (
            cond_centroid_probs * distances_from_centroids).sum(-1)   # [B]

        # M-step: accumulate the moving averages.
        self._update_centroid_probs(training, cond_centroid_probs)
        self._update_codebook_covariance(training, cond_centroid_probs,
                                         mu.detach(), variance.detach())

        # Sample the latent features; the model is deterministic at eval time.
        if training:
            latent = mu + perturb_diag * torch.randn_like(mu)
        else:
            latent = mu

        return DABOutput(latent=latent, distance=distances_from_codebook,
                         mean=mu, variance=variance,
                         distances_from_centroids=distances_from_centroids,
                         assignment=cond_centroid_probs)

    @torch.no_grad()
    def set_codebook_covariance(self) -> None:
        """Commit the covariance moving average for the next inference step."""
        cov = self.centroid_covariance_mavg
        if self.covariance_floor > 0.0:
            cov = cov.clamp_min(self.covariance_floor)
        self.centroid_covariance.copy_(cov)

    @torch.no_grad()
    def _calculate_codebook_covariance(self, cond_centroid_probs: torch.Tensor,
                                       mu: torch.Tensor,
                                       covariance: torch.Tensor
                                       ) -> torch.Tensor:
        """Optimal diagonal codebook covariance on the current batch.

        Equation (9) of Davis & Dhillon, *Differential entropic clustering of
        multivariate Gaussians* (NeurIPS 2006), with the off-diagonal entries
        dropped: a responsibility-weighted average of the encoder variances plus
        the squared distances between encoder and centroid means.
        """
        n = global_batch_size(cond_centroid_probs.shape[0], mu.device, mu.dtype)
        mu = all_gather_cat(mu)                                # [N, d]
        covariance = all_gather_cat(covariance)                # [N, d]
        probs = all_gather_cat(cond_centroid_probs)            # [N, K]

        w = self._dirichlet_weights(probs, n).unsqueeze(-1)    # [N, K, 1]
        diff = mu.unsqueeze(1) - self.centroid_means.unsqueeze(0)   # [N, K, d]
        new_covariance = (w * covariance.unsqueeze(1)).sum(0) \
            + (w * diff * diff).sum(0)                         # [K, d]
        return new_covariance


class NormalFullCovarianceDAB(_NormalDAB):
    """DAB dense layer with full-covariance Normal encoder and codebook.

    The distance from the codebook is the expected ``KL(encoder || centroid)``
    under the E-step responsibilities. This is the variant used by the reference
    WideResNet / CIFAR-10 model and by the synthetic-regression demo.

    In addition to :class:`_NormalDAB`'s arguments:

    Args:
      codebook_jitter: amount of ``eps * I`` added to the codebook covariance
        before inverting it in :meth:`set_codebook_covariance`. ``0.0``
        (the default) reproduces the reference implementation exactly; set a
        small positive value (e.g. ``1e-6``) if you hit ill-conditioning with a
        large ``dab_dim`` or very few RDFC steps per epoch.

    Example:
      >>> layer = NormalFullCovarianceDAB(in_features=640, dab_dim=8,
      ...                                 codebook_size=10, dab_tau=1.0)
      >>> out = layer(features)
      >>> logits, uncertainty = head(out.latent), out.distance
    """

    def __init__(self, in_features: int, dab_dim: int, codebook_size: int,
                 dab_tau: float = 1.0, momentum: float = 0.99,
                 activation: Union[str, Callable, None] = "relu",
                 use_bias: bool = False,
                 kernel_initializer: str = "glorot_uniform",
                 var_shift: float = 5.0, var_floor: float = 1e-5,
                 codebook_jitter: float = 0.0,
                 generator: Optional[torch.Generator] = None):
        super().__init__(
            in_features=in_features,
            # dab_dim means + d(d+1)/2 covariance entries (the matrix is packed
            # triangularly before being symmetrised)
            units=dab_dim + (dab_dim * (dab_dim + 1)) // 2,
            dab_dim=dab_dim, codebook_size=codebook_size, dab_tau=dab_tau,
            momentum=momentum, activation=activation, use_bias=use_bias,
            kernel_initializer=kernel_initializer, var_shift=var_shift,
            var_floor=var_floor, generator=generator)
        self.codebook_jitter = codebook_jitter

        eye = torch.eye(dab_dim).unsqueeze(0).repeat(codebook_size, 1, 1)
        # Moving average supporting the batched RDFC covariance update. The
        # covariance itself is kept here because its update is closed-form; the
        # precision and log-determinant below are what inference actually needs,
        # so they are precomputed once per RDFC phase.
        self.register_buffer("centroid_covariance_mavg",
                             torch.zeros(codebook_size, dab_dim, dab_dim))
        self.register_buffer("centroid_covariance", eye.clone())
        self.register_buffer("centroid_precision", eye.clone())
        self.register_buffer("centroid_covariance_log_abs_det",
                             torch.zeros(codebook_size))

    # ------------------------------------------------------------------ #
    def _encode(self, inputs: torch.Tensor):
        """Encoder mean, covariance, scale matrix and ``log|Sigma|``."""
        params = self._call(inputs)
        mu, perturb_factor = params.split(
            [self.dab_dim, self.units - self.dab_dim], dim=-1)

        # Pack the free parameters into a matrix (see the note in
        # ``symmetrized_fill_triangular``), then take an SVD so that the
        # covariance is positive semi-definite by construction.
        perturb_factor = symmetrized_fill_triangular(perturb_factor)
        u, perturb_diag, _ = torch.linalg.svd(perturb_factor)

        # Shift keeps small singular values and eases convergence.
        perturb_diag = self._encoder_scale(perturb_diag)

        # Scale matrix   S = U L U^T  with L = diag(perturb_diag);
        # covariance     C = S S^T = U L^2 U^T.
        ut = u.transpose(-1, -2)
        scale = u @ torch.diag_embed(perturb_diag) @ ut
        covariance = u @ torch.diag_embed(perturb_diag * perturb_diag) @ ut
        log_det = 2.0 * torch.log(perturb_diag).sum(-1)
        return mu, covariance, scale, log_det

    def forward(self, inputs: torch.Tensor,
                training: Optional[bool] = None) -> DABOutput:
        if training is None:
            training = self.training

        mu, covariance, scale, log_det = self._encode(inputs)

        # distances_from_centroids[i, j]: KL of datapoint i's encoder from
        # centroid j, using the precomputed codebook precisions/log-determinants.
        distances_from_centroids = kl_normal_full(
            mu, covariance, log_det, self.centroid_means,
            self.centroid_precision,
            self.centroid_covariance_log_abs_det)                 # [B, K]

        cond_centroid_probs = self._e_step(distances_from_centroids)

        distances_from_codebook = (
            cond_centroid_probs * distances_from_centroids).sum(-1)

        # M-step: accumulate the moving averages.
        self._update_centroid_probs(training, cond_centroid_probs)
        self._update_codebook_covariance(training, cond_centroid_probs,
                                         mu.detach(), covariance.detach())

        # Sample the latent features; the model is deterministic at eval time.
        if training:
            eps = torch.randn_like(mu).unsqueeze(-1)
            latent = mu + (scale @ eps).squeeze(-1)
        else:
            latent = mu

        return DABOutput(
            latent=latent, distance=distances_from_codebook, mean=mu,
            variance=torch.diagonal(covariance, dim1=-2, dim2=-1),
            covariance=covariance,
            distances_from_centroids=distances_from_centroids,
            assignment=cond_centroid_probs)

    @torch.no_grad()
    def set_codebook_covariance(self) -> None:
        """Commit the covariance moving average and refresh the cached
        precision matrices and log-determinants used by the distance."""
        cov = self.centroid_covariance_mavg
        if self.codebook_jitter > 0.0:
            eye = torch.eye(self.dab_dim, device=cov.device, dtype=cov.dtype)
            cov = cov + self.codebook_jitter * eye
        precision = torch.linalg.inv(cov)
        sign, log_abs_det = torch.linalg.slogdet(cov)
        if not torch.isfinite(precision).all() or (sign <= 0).any():
            raise RuntimeError(
                "DAB codebook covariance is singular or non-positive-definite. "
                "This usually means the RDFC covariance phase ran for too few "
                "steps relative to `momentum` (the moving average starts at "
                "zero, so it needs roughly 1/(1-momentum) steps to converge). "
                "Lower `momentum`, run more RDFC steps per epoch, or set a "
                "small `codebook_jitter`.")
        self.centroid_covariance.copy_(cov)
        self.centroid_precision.copy_(precision)
        self.centroid_covariance_log_abs_det.copy_(log_abs_det)

    @torch.no_grad()
    def _calculate_codebook_covariance(self, cond_centroid_probs: torch.Tensor,
                                       mu: torch.Tensor,
                                       covariance: torch.Tensor
                                       ) -> torch.Tensor:
        """Optimal codebook covariance on the current batch.

        Equation (9) of Davis & Dhillon, *Differential entropic clustering of
        multivariate Gaussians* (NeurIPS 2006): a responsibility-weighted
        average of the encoder covariances plus rank-one updates from the
        differences between the encoder and centroid means.
        """
        n = global_batch_size(cond_centroid_probs.shape[0], mu.device, mu.dtype)
        mu = all_gather_cat(mu)                                # [N, d]
        covariance = all_gather_cat(covariance)                # [N, d, d]
        probs = all_gather_cat(cond_centroid_probs)            # [N, K]

        diff = mu.unsqueeze(1) - self.centroid_means.unsqueeze(0)   # [N, K, d]
        diff_corr = diff.unsqueeze(-1) * diff.unsqueeze(-2)         # [N, K, d, d]
        new_covariance = covariance.unsqueeze(1) + diff_corr

        w = self._dirichlet_weights(probs, n)[..., None, None]      # [N,K,1,1]
        return (w * new_covariance).sum(0)                          # [K, d, d]


__all__ = ["DABOutput", "NormalDiagCovarianceDAB", "NormalFullCovarianceDAB"]
