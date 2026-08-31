"""Behavioural tests for the DAB layers, checked against the reference semantics."""
import math

import pytest
import torch
import torch.nn.functional as F

from dab import NormalDiagCovarianceDAB, NormalFullCovarianceDAB
from dab.functional import kl_normal_diag, symmetrized_fill_triangular

LAYERS = [NormalDiagCovarianceDAB, NormalFullCovarianceDAB]


def make(cls, **kw):
    kw.setdefault("in_features", 12)
    kw.setdefault("dab_dim", 4)
    kw.setdefault("codebook_size", 5)
    return cls(**kw)


# --------------------------------------------------------------------------- #
#  Shapes, output contract, encoder parameterisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", LAYERS)
def test_output_contract(cls):
    layer = make(cls)
    out = layer(torch.randn(9, 12), training=True)
    assert out.latent.shape == (9, 4)
    assert out.distance.shape == (9,)
    assert out.distances_from_centroids.shape == (9, 5)
    assert out.assignment.shape == (9, 5)
    # The reference layer returns concat([latent, distance], -1).
    assert out.as_tensor().shape == (9, 5)
    torch.testing.assert_close(out.as_tensor()[:, :4], out.latent)
    torch.testing.assert_close(out.as_tensor()[:, 4], out.distance)


@pytest.mark.parametrize("cls,units", [(NormalDiagCovarianceDAB, 8),
                                       (NormalFullCovarianceDAB, 4 + 10)])
def test_encoder_units(cls, units):
    """d means + (d | d(d+1)/2) covariance parameters, no bias by default."""
    layer = make(cls)
    assert layer.units == units
    assert layer.dense.weight.shape == (units, 12)
    assert layer.dense.bias is None


@pytest.mark.parametrize("cls", LAYERS)
def test_eval_mode_is_deterministic_and_train_mode_is_not(cls):
    layer = make(cls)
    layer.mark_initialized()
    x = torch.randn(16, 12)
    a, b = layer(x, training=False), layer(x, training=False)
    torch.testing.assert_close(a.latent, b.latent)
    torch.testing.assert_close(a.latent, a.mean)     # eval returns the mean
    torch.testing.assert_close(a.distance, b.distance)
    c, d = layer(x, training=True), layer(x, training=True)
    assert not torch.allclose(c.latent, d.latent)


@pytest.mark.parametrize("cls", LAYERS)
def test_training_flag_defaults_to_module_mode(cls):
    layer = make(cls).eval()
    out = layer(torch.randn(4, 12))
    torch.testing.assert_close(out.latent, out.mean)


@pytest.mark.parametrize("cls", LAYERS)
def test_distances_are_nonnegative(cls):
    layer = make(cls)
    out = layer(torch.randn(32, 12) * 5.0, training=True)
    assert (out.distances_from_centroids >= -1e-4).all()
    assert (out.distance >= -1e-4).all()


def test_encoder_scale_parameterisation():
    """scale = softplus(raw - 5) + 1e-5, and the diag variance is its square."""
    layer = make(NormalDiagCovarianceDAB)
    x = torch.randn(6, 12)
    params = layer.dense(F.relu(x))
    _, raw = params.split([4, 4], dim=-1)
    expected = F.softplus(raw - 5.0) + 1e-5
    out = layer(x, training=True)
    torch.testing.assert_close(out.variance, expected * expected)


def test_full_covariance_is_spd_and_matches_the_reference_construction():
    layer = make(NormalFullCovarianceDAB)
    x = torch.randn(6, 12)
    out = layer(x, training=True)
    cov = out.covariance
    torch.testing.assert_close(cov, cov.transpose(-1, -2), rtol=1e-5, atol=1e-6)
    assert (torch.linalg.eigvalsh(cov) > 0).all()

    # Reproduce cov = U diag(softplus(s-5)+1e-5)^2 U^T from the raw parameters.
    params = layer.dense(F.relu(x))
    _, raw = params.split([4, layer.units - 4], dim=-1)
    u, s, _ = torch.linalg.svd(symmetrized_fill_triangular(raw))
    s = F.softplus(s - 5.0) + 1e-5
    ref = u @ torch.diag_embed(s * s) @ u.transpose(-1, -2)
    torch.testing.assert_close(cov, ref, rtol=1e-4, atol=1e-6)


def test_full_covariance_sampling_uses_the_full_covariance():
    """Regression test: sampling must not collapse to the diagonal.

    Draw many latents for a single fixed input and check that the empirical
    covariance matches the encoder's *full* covariance, off-diagonals included.
    """
    torch.manual_seed(0)
    layer = make(NormalFullCovarianceDAB, dab_dim=3)
    # Push the raw scale parameters up so the sampling noise is measurable
    # (the -5.0 softplus shift makes the default scales ~1e-2).
    with torch.no_grad():
        layer.dense.weight.mul_(60.0)
    x = torch.randn(1, 12)
    with torch.no_grad():
        cov = layer(x, training=False).covariance[0]
        samples = torch.cat([layer(x, training=True).latent
                             for _ in range(20000)], dim=0)
    empirical = torch.cov(samples.T)
    assert cov.abs().max() > 1e-3, "test would be vacuous with a tiny covariance"
    torch.testing.assert_close(empirical, cov, rtol=0.12,
                               atol=0.05 * float(cov.abs().max()))
    off = cov - torch.diag_embed(torch.diagonal(cov))
    assert off.abs().max() > 0.05 * cov.abs().max(), "covariance is ~diagonal"


# --------------------------------------------------------------------------- #
#  E-step
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", LAYERS)
def test_uniform_assignment_before_initialization(cls):
    layer = make(cls)
    assert not bool(layer.initialized)
    out = layer(torch.randn(8, 12), training=True)
    torch.testing.assert_close(out.assignment,
                               torch.full((8, 5), 0.2))
    # ... and the distance is then the plain average over centroids.
    torch.testing.assert_close(out.distance,
                               out.distances_from_centroids.mean(-1))


@pytest.mark.parametrize("cls", LAYERS)
def test_assignment_is_softmax_of_log_prior_minus_tau_distance(cls):
    layer = make(cls, dab_tau=2.5)
    layer.mark_initialized()
    with torch.no_grad():
        layer.centroid_probs.copy_(torch.softmax(torch.randn(5), 0))
    out = layer(torch.randn(8, 12), training=True)
    ref = torch.softmax(torch.log(layer.centroid_probs).reshape(1, 5)
                        - 2.5 * out.distances_from_centroids.detach(), dim=-1)
    torch.testing.assert_close(out.assignment, ref)
    torch.testing.assert_close(out.assignment.sum(-1), torch.ones(8))


@pytest.mark.parametrize("cls", LAYERS)
def test_assignment_carries_no_gradient(cls):
    layer = make(cls)
    layer.mark_initialized()
    out = layer(torch.randn(8, 12), training=True)
    assert not out.assignment.requires_grad


@pytest.mark.parametrize("cls", LAYERS)
def test_distortion_gradient_reaches_centroid_means_and_the_encoder(cls):
    layer = make(cls)
    layer.mark_initialized()
    out = layer(torch.randn(8, 12), training=True)
    out.distance.mean().backward()
    assert layer.centroid_means.grad is not None
    assert layer.centroid_means.grad.abs().sum() > 0
    assert layer.dense.weight.grad is not None
    assert layer.dense.weight.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
#  M-step: moving averages and the closed-form covariance update
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", LAYERS)
def test_moving_averages_only_update_in_training_mode(cls):
    layer = make(cls)
    layer.mark_initialized()
    before_p = layer.centroid_probs_mavg.clone()
    before_c = layer.centroid_covariance_mavg.clone()
    layer(torch.randn(8, 12), training=False)
    torch.testing.assert_close(layer.centroid_probs_mavg, before_p)
    torch.testing.assert_close(layer.centroid_covariance_mavg, before_c)
    layer(torch.randn(8, 12), training=True)
    assert not torch.allclose(layer.centroid_probs_mavg, before_p)
    assert not torch.allclose(layer.centroid_covariance_mavg, before_c)


@pytest.mark.parametrize("cls", LAYERS)
def test_moving_averages_update_without_grad_enabled(cls):
    """The prior sub-pass of RDFC runs under ``torch.no_grad()``."""
    layer = make(cls)
    layer.mark_initialized()
    before = layer.centroid_probs_mavg.clone()
    with torch.no_grad():
        layer(torch.randn(8, 12), training=True)
    assert not torch.allclose(layer.centroid_probs_mavg, before)


@pytest.mark.parametrize("cls", LAYERS)
def test_prior_moving_average_formula(cls):
    layer = make(cls, momentum=0.7)
    layer.mark_initialized()
    before = layer.centroid_probs_mavg.clone()
    out = layer(torch.randn(8, 12), training=True)
    expected = 0.7 * before + 0.3 * out.assignment.mean(0)
    torch.testing.assert_close(layer.centroid_probs_mavg, expected)


@pytest.mark.parametrize("cls", LAYERS)
def test_reset_and_set_prior_round_trip(cls):
    layer = make(cls, momentum=0.0)
    layer.mark_initialized()
    layer.reset_centroid_probs()
    torch.testing.assert_close(layer.centroid_probs_mavg, torch.full((5,), 0.2))
    out = layer(torch.randn(64, 12), training=True)
    layer.set_centroid_probs()
    torch.testing.assert_close(layer.centroid_probs, out.assignment.mean(0))
    assert float(layer.centroid_probs.sum()) == pytest.approx(1.0, abs=1e-5)


def test_diag_codebook_covariance_closed_form():
    """Weighted encoder variance + weighted squared mean differences."""
    torch.manual_seed(0)
    layer = make(NormalDiagCovarianceDAB, momentum=0.0)
    layer.mark_initialized()
    layer.reset_codebook_covariance()
    x = torch.randn(32, 12)
    out = layer(x, training=True)
    layer.set_codebook_covariance()

    a, mu, var = out.assignment, out.mean.detach(), out.variance.detach()
    w = (a + 5.0) / (a.sum(0, keepdim=True) + 5.0 * 32)          # Dirichlet(5)
    expected = torch.zeros(5, 4)
    for k in range(5):
        diff = mu - layer.centroid_means[k]
        expected[k] = (w[:, k:k + 1] * (var + diff * diff)).sum(0)
    torch.testing.assert_close(layer.centroid_covariance, expected,
                               rtol=1e-4, atol=1e-6)


def test_full_codebook_covariance_closed_form():
    """Weighted encoder covariance + rank-one mean-difference updates."""
    torch.manual_seed(0)
    layer = make(NormalFullCovarianceDAB, momentum=0.0)
    layer.mark_initialized()
    layer.reset_codebook_covariance()
    x = torch.randn(32, 12)
    out = layer(x, training=True)

    a, mu, cov = out.assignment, out.mean.detach(), out.covariance.detach()
    w = (a + 5.0) / (a.sum(0, keepdim=True) + 5.0 * 32)
    expected = torch.zeros(5, 4, 4)
    for k in range(5):
        diff = (mu - layer.centroid_means[k]).unsqueeze(-1)
        expected[k] = (w[:, k, None, None]
                       * (cov + diff @ diff.transpose(-1, -2))).sum(0)
    torch.testing.assert_close(layer.centroid_covariance_mavg, expected,
                               rtol=1e-4, atol=1e-6)

    # set_codebook_covariance caches a consistent precision and log-determinant.
    layer.set_codebook_covariance()
    torch.testing.assert_close(
        layer.centroid_precision @ layer.centroid_covariance,
        torch.eye(4).expand(5, 4, 4), rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(layer.centroid_covariance_log_abs_det,
                               torch.logdet(layer.centroid_covariance),
                               rtol=1e-4, atol=1e-5)


def test_codebook_covariance_starts_at_the_identity():
    diag = make(NormalDiagCovarianceDAB)
    torch.testing.assert_close(diag.centroid_covariance, torch.ones(5, 4))
    full = make(NormalFullCovarianceDAB)
    torch.testing.assert_close(full.centroid_precision,
                               torch.eye(4).expand(5, 4, 4).contiguous())
    torch.testing.assert_close(full.centroid_covariance_log_abs_det,
                               torch.zeros(5))


def test_full_distance_uses_the_cached_precision_and_logdet():
    """The layer's distance must equal an explicit KL against the codebook."""
    torch.manual_seed(0)
    layer = make(NormalFullCovarianceDAB, momentum=0.0)
    layer.mark_initialized()
    layer.reset_codebook_covariance()
    layer(torch.randn(64, 12), training=True)
    layer.set_codebook_covariance()

    from torch.distributions import MultivariateNormal, kl_divergence
    with torch.no_grad():
        out = layer(torch.randn(6, 12), training=False)
        for i in range(6):
            p = MultivariateNormal(out.mean[i], out.covariance[i])
            for k in range(5):
                q = MultivariateNormal(layer.centroid_means[k],
                                       layer.centroid_covariance[k])
                torch.testing.assert_close(
                    out.distances_from_centroids[i, k], kl_divergence(p, q),
                    rtol=1e-3, atol=1e-4)


def test_diag_distance_equals_explicit_kl():
    layer = make(NormalDiagCovarianceDAB)
    with torch.no_grad():
        layer.centroid_covariance.copy_(torch.rand(5, 4) + 0.5)
    out = layer(torch.randn(6, 12), training=False)
    ref = kl_normal_diag(out.mean.unsqueeze(1), out.variance.unsqueeze(1),
                         layer.centroid_means.unsqueeze(0),
                         layer.centroid_covariance.unsqueeze(0))
    torch.testing.assert_close(out.distances_from_centroids, ref)


# --------------------------------------------------------------------------- #
#  Misc
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", LAYERS)
def test_state_dict_round_trip(cls):
    torch.manual_seed(0)
    a = make(cls, momentum=0.0)
    a.mark_initialized()
    a.reset_codebook()
    a(torch.randn(32, 12), training=True)
    a.set_codebook()
    b = make(cls)
    b.load_state_dict(a.state_dict())
    assert bool(b.initialized)
    x = torch.randn(5, 12)
    torch.testing.assert_close(a(x, training=False).distance,
                               b(x, training=False).distance)


@pytest.mark.parametrize("cls", LAYERS)
def test_custom_activation_and_bias(cls):
    layer = make(cls, activation=None, use_bias=True)
    assert layer.dense.bias is not None
    x = torch.randn(4, 12)
    torch.testing.assert_close(layer._call(x), layer.dense(x))


@pytest.mark.parametrize("cls", LAYERS)
def test_reproducible_initialisation(cls):
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    torch.testing.assert_close(make(cls, generator=g1).centroid_means,
                               make(cls, generator=g2).centroid_means)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("cls", LAYERS)
def test_runs_on_cuda(cls):
    layer = make(cls).cuda()
    layer.mark_initialized()
    out = layer(torch.randn(8, 12, device="cuda"), training=True)
    assert out.distance.device.type == "cuda"
    layer.set_codebook()
