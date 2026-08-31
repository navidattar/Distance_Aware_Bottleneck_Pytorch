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
"""Pure tensor primitives used by the Distance-Aware Bottleneck layers.

Every function here has a one-to-one counterpart in the reference TensorFlow /
TensorFlow-Probability implementation; the mapping is documented in
``docs/FAITHFULNESS.md``.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.distributed as dist


# --------------------------------------------------------------------------- #
#  Triangular packing (port of ``tfp.math.fill_triangular``)
# --------------------------------------------------------------------------- #
def fill_triangular(x: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """Pack the last axis of ``x`` into a triangular matrix.

    Bit-exact port of ``tensorflow_probability.math.fill_triangular``, including
    its (non-obvious) element ordering::

        fill_triangular([1, 2, 3, 4, 5, 6])              fill_triangular(..., upper=True)
            [[4, 0, 0],                                      [[1, 2, 3],
             [6, 5, 0],                                       [0, 5, 6],
             [3, 2, 1]]                                       [0, 0, 4]]

    Note that the ``upper=True`` result is *not* the transpose of the
    ``upper=False`` result. The reference DAB encoder relies on exactly this
    ordering (see :func:`symmetrized_fill_triangular`).

    Args:
      x: ``[..., n * (n + 1) // 2]``.
      upper: return the upper-triangular packing instead of the lower one.

    Returns:
      ``[..., n, n]``.
    """
    m = x.shape[-1]
    n = int((math.isqrt(1 + 8 * m) - 1) // 2)
    if n * (n + 1) // 2 != m:
        raise ValueError(
            f"last dimension {m} is not a triangular number n*(n+1)/2")
    batch = x.shape[:-1]
    if upper:
        packed = torch.cat([x, x[..., n:].flip(-1)], dim=-1)
        return torch.triu(packed.reshape(*batch, n, n))
    packed = torch.cat([x[..., n:], x.flip(-1)], dim=-1)
    return torch.tril(packed.reshape(*batch, n, n))


def symmetrized_fill_triangular(x: torch.Tensor) -> torch.Tensor:
    """``0.5 * (fill_triangular(x, upper=True) + fill_triangular(x))``.

    This is exactly what the reference full-covariance DAB encoder computes
    before its SVD. Because the two ``fill_triangular`` orderings are not
    transposes of one another the result is *not* symmetric in general, despite
    the comment in the original source. It does not matter: the matrix is only
    ever consumed by an SVD whose left singular vectors ``U`` are used to build
    ``cov = U diag(s^2) U^T``, which is symmetric positive definite for any
    input. We reproduce the original arithmetic bit-for-bit so that a model
    ported from the TensorFlow code behaves identically.
    """
    return 0.5 * (fill_triangular(x, upper=True) + fill_triangular(x))


# --------------------------------------------------------------------------- #
#  Gaussian KL divergences
# --------------------------------------------------------------------------- #
def kl_normal_diag(mu_p: torch.Tensor, var_p: torch.Tensor,
                   mu_q: torch.Tensor, var_q: torch.Tensor) -> torch.Tensor:
    r"""``KL( N(mu_p, diag(var_p)) || N(mu_q, diag(var_q)) )``.

    .. math::
        \tfrac12 \sum_j \Big[ \frac{\sigma_{p,j}^2 + (\mu_{p,j}-\mu_{q,j})^2}
                                   {\sigma_{q,j}^2}
                              - 1 + \log\sigma_{q,j}^2 - \log\sigma_{p,j}^2 \Big]

    Shapes broadcast over leading dimensions; the last axis is the latent
    dimension and is summed out. Equivalent to
    ``tfp.distributions.MultivariateNormalDiag.kl_divergence``.
    """
    per_coord = (var_p + (mu_p - mu_q) ** 2) / var_q - 1.0 \
        + torch.log(var_q) - torch.log(var_p)
    return 0.5 * per_coord.sum(-1)


def kl_normal_full(mu_p: torch.Tensor, cov_p: torch.Tensor,
                   log_det_p: torch.Tensor, mu_q: torch.Tensor,
                   precision_q: torch.Tensor,
                   log_det_q: torch.Tensor) -> torch.Tensor:
    r"""``KL( N(mu_p, cov_p) || N(mu_q, cov_q) )`` from precomputed codebook state.

    .. math::
        \tfrac12\Big[\operatorname{tr}(\Sigma_q^{-1}\Sigma_p)
                     + (\mu_p-\mu_q)^\top \Sigma_q^{-1} (\mu_p-\mu_q) - d
                     + \log|\Sigma_q| - \log|\Sigma_p|\Big]

    The codebook side is supplied as a precision matrix and a log-determinant
    because those are the quantities the reference implementation caches at the
    end of every RDFC M-step (``centroid_precision`` /
    ``centroid_covariance_log_abs_det``). This keeps the pairwise distance an
    ``O(B K d^2)`` batched matmul instead of ``B * K`` Cholesky factorisations.

    Args:
      mu_p:        ``[B, d]``       encoder means.
      cov_p:       ``[B, d, d]``    encoder covariances (symmetric).
      log_det_p:   ``[B]``          ``log|Sigma_p|``.
      mu_q:        ``[K, d]``       centroid means.
      precision_q: ``[K, d, d]``    centroid precisions (symmetric).
      log_det_q:   ``[K]``          ``log|Sigma_q|``.

    Returns:
      ``[B, K]``.
    """
    d = mu_p.shape[-1]
    diff = mu_p.unsqueeze(1) - mu_q.unsqueeze(0)                    # [B, K, d]
    maha = torch.einsum("bkd,kde,bke->bk", diff, precision_q, diff)
    # tr(P_q Sigma_p); both operands are symmetric so the transpose is free.
    trace = torch.einsum("kij,bij->bk", precision_q, cov_p)
    return 0.5 * (trace + maha - d
                  + log_det_q.unsqueeze(0) - log_det_p.unsqueeze(1))


# --------------------------------------------------------------------------- #
#  Distributed reductions (ports of the TF ``replica_ctx`` branches)
# --------------------------------------------------------------------------- #
def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    """Sum ``t`` across all ranks (no-op for single-process training)."""
    if not is_distributed():
        return t
    t = t.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_gather_cat(t: torch.Tensor) -> torch.Tensor:
    """Concatenate ``t`` from all ranks along axis 0.

    Mirrors ``replica_ctx.all_gather(..., axis=0)``. Requires an identical
    per-rank shape, which is the case for the drop-last loaders used by DAB.
    """
    if not is_distributed():
        return t
    world = dist.get_world_size()
    buf = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(buf, t.contiguous())
    return torch.cat(buf, dim=0)


def global_batch_size(local_batch_size: int, device,
                      dtype=torch.float32) -> torch.Tensor:
    """Total number of examples in the current step across all ranks."""
    bs = torch.tensor(float(local_batch_size), device=device, dtype=dtype)
    return all_reduce_sum(bs)


__all__ = [
    "fill_triangular", "symmetrized_fill_triangular",
    "kl_normal_diag", "kl_normal_full",
    "is_distributed", "all_reduce_sum", "all_gather_cat", "global_batch_size",
]
