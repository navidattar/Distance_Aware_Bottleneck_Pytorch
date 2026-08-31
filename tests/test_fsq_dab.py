"""Tests for the FSQ-codebook DAB layer.

The central one is :func:`test_factorized_distance_equals_product_codebook`:
the per-coordinate computation must be *exactly* the expected KL over all
prod_j L_j product codes, not an approximation of it.
"""
import itertools

import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from dab import (FSQDAB, build_optimizers, codebook_distortion,
                 codebook_parameters, find_dab_layers, rdfc_epoch)
from dab.layers import DABLayer


def make(**kw):
    kw.setdefault("in_features", 12)
    kw.setdefault("levels", [5, 4, 3])
    return FSQDAB(**kw)


# --------------------------------------------------------------------------- #
#  The factorization identity
# --------------------------------------------------------------------------- #
def _brute_force_distance(layer, mu, var, tau):
    """Expected KL over every product code, enumerated explicitly."""
    d = layer.dab_dim
    code_var = layer._code_variance()
    prior = layer.centroid_probs
    combos = list(itertools.product(*[range(L) for L in layer.levels]))
    kls, log_priors = [], []
    for combo in combos:
        m = torch.stack([layer.centroid_means[j, combo[j]] for j in range(d)])
        s2 = torch.stack([code_var[j, combo[j]] for j in range(d)])
        lp = sum(torch.log(prior[j, combo[j]]) for j in range(d))
        kls.append(kl_divergence(Normal(mu, var.sqrt()),
                                 Normal(m, s2.sqrt())).sum(-1))
        log_priors.append(lp)
    kl = torch.stack(kls, dim=-1)                       # [B, K]
    lp = torch.stack(log_priors).reshape(1, -1)         # [1, K]
    a = torch.softmax(lp - tau * kl, dim=-1)
    return (a * kl).sum(-1)


@pytest.mark.parametrize("levels", [[3, 3], [5, 4, 3], [4, 4, 4]])
@pytest.mark.parametrize("tau", [0.5, 1.0, 3.0])
def test_factorized_distance_equals_product_codebook(levels, tau):
    """O(sum L_j) coordinate-wise == O(prod L_j) explicit enumeration."""
    torch.manual_seed(0)
    layer = make(levels=levels, dab_tau=tau, momentum=0.0)
    layer.mark_initialized()
    # Give the priors and the code variances non-uniform values, otherwise the
    # identity could hold for trivial reasons.
    with torch.no_grad():
        for j, L in enumerate(levels):
            layer.centroid_probs[j, :L] = torch.softmax(torch.randn(L), 0)
            layer.centroid_covariance[j, :L] = torch.rand(L) * 0.2 + 0.05
    x = torch.randn(6, 12)
    with torch.no_grad():
        out = layer(x, training=False)
        ref = _brute_force_distance(layer, out.mean, out.variance, tau)
    torch.testing.assert_close(out.distance, ref, rtol=1e-4, atol=1e-5)


def test_assignment_factorises_into_the_product_assignment():
    """a_c = prod_j a_{j,l_j} over the full codebook."""
    torch.manual_seed(0)
    levels = [3, 3]
    layer = make(levels=levels, dab_tau=1.5)
    layer.mark_initialized()
    with torch.no_grad():
        for j, L in enumerate(levels):
            layer.centroid_probs[j, :L] = torch.softmax(torch.randn(L), 0)
    with torch.no_grad():
        out = layer(torch.randn(4, 12), training=False)
        code_var = layer._code_variance()
        kls, lps = [], []
        for combo in itertools.product(range(3), range(3)):
            m = torch.stack([layer.centroid_means[j, combo[j]] for j in range(2)])
            s2 = torch.stack([code_var[j, combo[j]] for j in range(2)])
            kls.append(kl_divergence(Normal(out.mean, out.variance.sqrt()),
                                     Normal(m, s2.sqrt())).sum(-1))
            lps.append(sum(torch.log(layer.centroid_probs[j, combo[j]])
                           for j in range(2)))
        a_full = torch.softmax(torch.stack(lps).reshape(1, -1)
                               - 1.5 * torch.stack(kls, -1), dim=-1)
    a_prod = torch.stack([out.assignment[:, 0, c0] * out.assignment[:, 1, c1]
                          for c0, c1 in itertools.product(range(3), range(3))], -1)
    torch.testing.assert_close(a_prod, a_full, rtol=1e-4, atol=1e-6)


# --------------------------------------------------------------------------- #
#  Layer contract
# --------------------------------------------------------------------------- #
def test_output_contract_and_codebook_size():
    layer = make(levels=[8, 6, 5])
    out = layer(torch.randn(9, 12), training=True)
    assert layer.codebook_size == 240
    assert out.latent.shape == (9, 3)
    assert out.distance.shape == (9,)
    assert out.distances_from_centroids.shape == (9, 3, 8)   # padded to Lmax
    assert out.indices.shape == (9,)
    assert out.per_coord_indices.shape == (9, 3)
    assert out.pre_bound_mean.shape == (9, 3)
    assert int(out.indices.min()) >= 0 and int(out.indices.max()) < 240
    assert out.as_tensor().shape == (9, 4)


def test_it_is_a_dab_layer_and_is_discoverable():
    layer = make()
    assert isinstance(layer, DABLayer)
    assert find_dab_layers(torch.nn.Sequential(layer)) == [layer]


def test_ragged_levels_are_masked_out():
    layer = make(levels=[5, 4, 3])
    out = layer(torch.randn(8, 12), training=True)
    invalid = ~layer.level_mask.unsqueeze(0).expand_as(out.assignment)
    assert float(out.assignment[invalid].abs().max()) == 0.0
    torch.testing.assert_close(out.assignment.sum(-1), torch.ones(8, 3))
    for j, L in enumerate(layer.levels):
        assert float(layer.centroid_probs[j, L:].sum()) == 0.0


def test_mean_is_bounded_to_the_grid_range():
    layer = make(levels=[5, 5, 5])
    out = layer(torch.randn(64, 12) * 100, training=False)
    assert float(out.mean.abs().max()) <= 1.0 + 1e-5


def test_unbounded_mode_lets_the_distance_keep_growing():
    """`bound=None` trades finite tokens for an unsaturated OOD score."""
    torch.manual_seed(0)
    bounded, unbounded = make(levels=[5, 5, 5]), make(levels=[5, 5, 5], bound=None)
    for layer in (bounded, unbounded):
        layer.mark_initialized()
    with torch.no_grad():
        near = torch.randn(32, 12)
        far = near * 50
        d_b = (bounded(far, training=False).distance.mean()
               / bounded(near, training=False).distance.mean())
        d_u = (unbounded(far, training=False).distance.mean()
               / unbounded(near, training=False).distance.mean())
    assert d_u > d_b


def test_saturation_diagnostic_fires_on_extreme_inputs():
    layer = make(levels=[5, 5, 5])
    layer.mark_initialized()
    with torch.no_grad():
        mild = layer(torch.randn(64, 12) * 0.01, training=False)
        wild = layer(torch.randn(64, 12) * 100, training=False)
    assert float(wild.diagnostics["saturated_frac"]) > \
        float(mild.diagnostics["saturated_frac"])
    assert float(wild.diagnostics["saturated_frac"]) > 0.9


def test_uniform_assignment_before_initialization():
    layer = make(levels=[5, 4, 3])
    assert not bool(layer.initialized)
    out = layer(torch.randn(8, 12), training=True)
    for j, L in enumerate(layer.levels):
        torch.testing.assert_close(out.assignment[:, j, :L],
                                   torch.full((8, L), 1.0 / L))


def test_eval_is_deterministic_and_training_samples():
    layer = make()
    layer.mark_initialized()
    x = torch.randn(16, 12)
    torch.testing.assert_close(layer(x, training=False).latent,
                               layer(x, training=False).latent)
    assert not torch.allclose(layer(x, training=True).latent,
                              layer(x, training=True).latent)


# --------------------------------------------------------------------------- #
#  Hard (FSQ straight-through) mode
# --------------------------------------------------------------------------- #
def test_hard_mode_assignment_is_one_hot_at_the_nearest_level():
    layer = make(levels=[5, 4, 3], hard=True)
    layer.mark_initialized()
    out = layer(torch.randn(10, 12), training=False)
    assert set(out.assignment.unique().tolist()) <= {0.0, 1.0}
    torch.testing.assert_close(out.assignment.sum(-1), torch.ones(10, 3))
    torch.testing.assert_close(out.assignment.argmax(-1), out.per_coord_indices)


def test_hard_mode_distortion_is_the_kl_to_the_selected_code():
    layer = make(levels=[5, 4, 3], hard=True)
    layer.mark_initialized()
    with torch.no_grad():
        out = layer(torch.randn(6, 12), training=False)
        code_var = layer._code_variance()
        idx = out.per_coord_indices
        m = layer.centroid_means.unsqueeze(0).expand(6, -1, -1) \
            .gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        s2 = code_var.unsqueeze(0).expand(6, -1, -1) \
            .gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        ref = kl_divergence(Normal(out.mean, out.variance.sqrt()),
                            Normal(m, s2.sqrt())).sum(-1)
    torch.testing.assert_close(out.distance, ref, rtol=1e-4, atol=1e-6)


def test_hard_mode_latent_is_a_codeword_and_matches_the_indices():
    layer = make(levels=[8, 6, 5], hard=True)
    with torch.no_grad():
        out = layer(torch.randn(12, 12), training=False)
    torch.testing.assert_close(layer.fsq.indices_to_codes(out.indices),
                               out.latent)


def test_hard_mode_passes_gradients_through_the_rounding():
    layer = make(levels=[5, 5, 5], hard=True)
    out = layer(torch.randn(4, 12), training=True)
    out.latent.sum().backward()
    g = layer.dense.weight.grad
    assert g is not None and torch.isfinite(g).all() and (g.abs() > 0).any()


def test_soft_mode_latent_is_continuous():
    layer = make(levels=[5, 5, 5], hard=False)
    with torch.no_grad():
        out = layer(torch.randn(64, 12), training=False)
    on_grid = (out.latent.unsqueeze(-1)
               - layer.centroid_means.unsqueeze(0)).abs().min(-1).values
    assert float(on_grid.max()) > 1e-4      # not snapped to the grid


# --------------------------------------------------------------------------- #
#  Codebook fitting (RDFC)
# --------------------------------------------------------------------------- #
def test_grid_has_no_trainable_parameters_by_default():
    layer = make()
    assert codebook_parameters(layer) == []
    _, code_opt = build_optimizers(layer)
    assert code_opt is None
    assert layer.num_codebook_parameters == 0


def test_learned_scale_mode_exposes_a_codebook_parameter():
    layer = make(code_scale_mode="learned")
    names = [n for n, _ in layer.named_parameters() if "centroid" in n]
    assert names == ["centroid_scale_raw"]
    _, code_opt = build_optimizers(layer)
    assert code_opt is not None


def test_rdfc_epoch_fits_the_codebook_without_an_optimizer():
    """The FSQ grid is fixed; the M-step still fits variances and priors."""
    torch.manual_seed(0)
    layer = make(levels=[5, 5, 5], momentum=0.0)
    _, code_opt = build_optimizers(layer)
    assert code_opt is None
    before_var = layer.centroid_covariance.clone()
    before_prior = layer.centroid_probs.clone()
    data = [torch.randn(32, 12) for _ in range(4)]
    rdfc_epoch(layer, code_opt, lambda: iter(data),
               lambda b: codebook_distortion(layer(b, training=True).distance))
    assert bool(layer.initialized)
    assert not torch.allclose(layer.centroid_covariance, before_var)
    assert not torch.allclose(layer.centroid_probs, before_prior)
    torch.testing.assert_close(layer.centroid_probs.sum(-1), torch.ones(3))


def test_fixed_scale_mode_never_changes_the_variances():
    layer = make(code_scale_mode="fixed", momentum=0.0)
    before = layer.centroid_covariance.clone()
    rdfc_epoch(layer, None, lambda: iter([torch.randn(16, 12)]),
               lambda b: codebook_distortion(layer(b, training=True).distance))
    torch.testing.assert_close(layer.centroid_covariance, before)


def test_code_variance_closed_form():
    """Responsibility-weighted encoder variance + squared distance to the grid."""
    torch.manual_seed(0)
    layer = make(levels=[5, 5, 5], momentum=0.0)
    layer.mark_initialized()
    layer.reset_codebook_covariance()
    out = layer(torch.randn(32, 12), training=True)
    layer.set_codebook_covariance()

    a, mu, var = out.assignment, out.mean.detach(), out.variance.detach()
    w = (a + 5.0) / (a.sum(0, keepdim=True) + 5.0 * 32)
    diff = mu.unsqueeze(-1) - layer.centroid_means.unsqueeze(0)
    expected = (w * (var.unsqueeze(-1) + diff * diff)).sum(0)
    torch.testing.assert_close(layer.centroid_covariance,
                               expected.masked_fill(~layer.level_mask, 1.0),
                               rtol=1e-4, atol=1e-6)


def test_prior_reflects_level_usage_after_rdfc():
    """The committed prior is the mean responsibility of the last M-step batch.

    Captured with a hook rather than recomputed, because committing the prior
    changes the E-step that would produce it.
    """
    torch.manual_seed(0)
    layer = make(levels=[5, 5, 5], momentum=0.0)
    data = [torch.randn(64, 12) for _ in range(3)]
    seen = []
    h = layer.register_forward_hook(
        lambda mod, inp, out: seen.append(out.assignment.detach().clone()))
    rdfc_epoch(layer, None, lambda: iter(data),
               lambda b: codebook_distortion(layer(b, training=True).distance))
    h.remove()
    assert len(seen) == 2 * len(data)
    torch.testing.assert_close(layer.centroid_probs, seen[-1].mean(0),
                               rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(layer.centroid_probs.sum(-1), torch.ones(3))


def test_state_dict_round_trip():
    torch.manual_seed(0)
    a = make(levels=[5, 4, 3], momentum=0.0)
    rdfc_epoch(a, None, lambda: iter([torch.randn(32, 12)]),
               lambda b: codebook_distortion(a(b, training=True).distance))
    b = make(levels=[5, 4, 3])
    b.load_state_dict(a.state_dict())
    assert bool(b.initialized)
    x = torch.randn(5, 12)
    torch.testing.assert_close(a(x, training=False).distance,
                               b(x, training=False).distance)


def test_rejects_inconsistent_configuration():
    with pytest.raises(ValueError):
        FSQDAB(in_features=8, levels=5)                    # dab_dim missing
    with pytest.raises(ValueError):
        FSQDAB(in_features=8, levels=[5, 5], dab_dim=3)
    with pytest.raises(ValueError):
        FSQDAB(in_features=8, levels=[5], bound="tanh")
    with pytest.raises(ValueError):
        FSQDAB(in_features=8, levels=[5], code_scale_mode="nope")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_runs_on_cuda():
    layer = make(levels=[8, 6, 5]).cuda()
    layer.mark_initialized()
    out = layer(torch.randn(8, 12, device="cuda"), training=True)
    assert out.distance.device.type == "cuda"
    layer.set_codebook()
