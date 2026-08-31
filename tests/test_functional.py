"""Numerical validation of the tensor primitives against independent references."""
import math

import pytest
import torch
from torch.distributions import MultivariateNormal, Normal, kl_divergence

from dab.functional import (fill_triangular, kl_normal_diag, kl_normal_full,
                            symmetrized_fill_triangular)


def test_fill_triangular_matches_tfp_ordering():
    # The exact examples from the tfp.math.fill_triangular docstring.
    x = torch.arange(1.0, 7.0)
    torch.testing.assert_close(
        fill_triangular(x),
        torch.tensor([[4., 0., 0.], [6., 5., 0.], [3., 2., 1.]]))
    torch.testing.assert_close(
        fill_triangular(x, upper=True),
        torch.tensor([[1., 2., 3.], [0., 5., 6.], [0., 0., 4.]]))


@pytest.mark.parametrize("d", [1, 2, 3, 5, 8])
def test_fill_triangular_shapes_and_batching(d):
    m = d * (d + 1) // 2
    x = torch.randn(4, 7, m)
    lower, upper = fill_triangular(x), fill_triangular(x, upper=True)
    assert lower.shape == upper.shape == (4, 7, d, d)
    torch.testing.assert_close(lower, torch.tril(lower))
    torch.testing.assert_close(upper, torch.triu(upper))
    # Every free parameter appears exactly once in each packing.
    assert lower.abs().sum() == pytest.approx(float(x.abs().sum()), rel=1e-5)


def test_symmetrized_fill_triangular_is_the_reference_combination():
    x = torch.randn(3, 6)
    torch.testing.assert_close(
        symmetrized_fill_triangular(x),
        0.5 * (fill_triangular(x, upper=True) + fill_triangular(x)))


def test_kl_normal_diag_matches_torch_distributions():
    mu_p, mu_q = torch.randn(6, 4), torch.randn(6, 4)
    var_p, var_q = torch.rand(6, 4) + 0.1, torch.rand(6, 4) + 0.1
    ref = kl_divergence(Normal(mu_p, var_p.sqrt()),
                        Normal(mu_q, var_q.sqrt())).sum(-1)
    torch.testing.assert_close(kl_normal_diag(mu_p, var_p, mu_q, var_q), ref)


def test_kl_normal_diag_broadcasts_over_the_codebook_axis():
    mu, var = torch.randn(5, 3), torch.rand(5, 3) + 0.1
    m, s = torch.randn(4, 3), torch.rand(4, 3) + 0.1
    got = kl_normal_diag(mu.unsqueeze(1), var.unsqueeze(1),
                         m.unsqueeze(0), s.unsqueeze(0))
    assert got.shape == (5, 4)
    for i in range(5):
        for k in range(4):
            torch.testing.assert_close(
                got[i, k], kl_normal_diag(mu[i], var[i], m[k], s[k]))


def _random_spd(n, d, generator=None):
    a = torch.randn(n, d, d, generator=generator)
    return a @ a.transpose(-1, -2) + d * torch.eye(d)


def test_kl_normal_full_matches_torch_distributions():
    torch.manual_seed(0)
    B, K, d = 5, 4, 3
    mu_p, cov_p = torch.randn(B, d), _random_spd(B, d)
    mu_q, cov_q = torch.randn(K, d), _random_spd(K, d)
    got = kl_normal_full(mu_p, cov_p, torch.logdet(cov_p), mu_q,
                         torch.linalg.inv(cov_q), torch.logdet(cov_q))
    assert got.shape == (B, K)
    for i in range(B):
        for k in range(K):
            ref = kl_divergence(MultivariateNormal(mu_p[i], cov_p[i]),
                                MultivariateNormal(mu_q[k], cov_q[k]))
            torch.testing.assert_close(got[i, k], ref, rtol=1e-4, atol=1e-5)


def test_kl_is_zero_for_identical_distributions():
    d = 4
    mu, cov = torch.randn(3, d), _random_spd(3, d)
    got = kl_normal_full(mu, cov, torch.logdet(cov), mu,
                         torch.linalg.inv(cov), torch.logdet(cov))
    torch.testing.assert_close(torch.diagonal(got), torch.zeros(3),
                               rtol=0, atol=1e-4)


def test_full_kl_reduces_to_diagonal_kl_for_diagonal_covariances():
    B, K, d = 4, 3, 5
    mu_p, var_p = torch.randn(B, d), torch.rand(B, d) + 0.5
    mu_q, var_q = torch.randn(K, d), torch.rand(K, d) + 0.5
    cov_p, cov_q = torch.diag_embed(var_p), torch.diag_embed(var_q)
    full = kl_normal_full(mu_p, cov_p, torch.log(var_p).sum(-1), mu_q,
                          torch.diag_embed(1.0 / var_q),
                          torch.log(var_q).sum(-1))
    diag = kl_normal_diag(mu_p.unsqueeze(1), var_p.unsqueeze(1),
                          mu_q.unsqueeze(0), var_q.unsqueeze(0))
    torch.testing.assert_close(full, diag, rtol=1e-5, atol=1e-6)
