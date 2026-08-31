"""Tests for iFSQ, checked against the paper's Algorithm 1 and Figure 2."""
import pytest
import torch

from dab import build_bottleneck
from dab.fsq import FSQ
from dab.ifsq import (IFSQ, IFSQ_ALPHA, IFSQ_ALPHA_KS_OPTIMAL, IFSQDAB,
                      TANH_ALPHA, ifsq_bound, level_histogram,
                      uniformity_error)
from dab.layers import DABLayer


# --------------------------------------------------------------------------- #
#  The bounding map
# --------------------------------------------------------------------------- #
def test_alpha_two_is_exactly_tanh():
    """tanh(z) = 2*sigmoid(2z) - 1, the identity the paper's argument rests on."""
    z = torch.randn(1000) * 3
    torch.testing.assert_close(ifsq_bound(z, TANH_ALPHA), torch.tanh(z),
                               rtol=1e-6, atol=1e-6)


def test_bound_maps_into_the_open_unit_interval():
    z = torch.tensor([-1e6, -3.0, 0.0, 3.0, 1e6])
    y = ifsq_bound(z, IFSQ_ALPHA)
    assert float(y.min()) > -1.0 - 1e-6 and float(y.max()) < 1.0 + 1e-6
    assert float(ifsq_bound(torch.zeros(1), IFSQ_ALPHA)) == pytest.approx(0.0)


def test_bound_is_monotonic():
    z = torch.linspace(-10, 10, 5000)
    y = ifsq_bound(z, IFSQ_ALPHA)
    assert (y[1:] - y[:-1] >= 0).all()          # ties only where float32 saturates
    z = torch.linspace(-4, 4, 5000)
    y = ifsq_bound(z, IFSQ_ALPHA)
    assert (y[1:] - y[:-1] > 0).all()


# --------------------------------------------------------------------------- #
#  Figure 2(b): the uniformity sweep
# --------------------------------------------------------------------------- #
def test_alpha_1_6_is_more_uniform_than_tanh():
    """The paper's core claim: 1.6 beats tanh's 2.0 at matching a uniform."""
    at_16 = uniformity_error(IFSQ_ALPHA, 200_000)
    at_20 = uniformity_error(TANH_ALPHA, 200_000)
    assert at_16["ks"] < at_20["ks"]
    assert at_16["rmse"] < at_20["rmse"]


def test_uniformity_is_unimodal_in_alpha_with_a_minimum_near_1_7():
    """Reproduces Figure 2(b), and locates the optimum of its own criterion.

    The paper sweeps {1.0, 1.3, 1.6, 2.0, 2.4} and picks 1.6. On a finer sweep
    the Kolmogorov-Smirnov minimum sits at ~1.70 -- the classic logistic
    approximation to the Gaussian CDF, sigma(1.702 x) ~ Phi(x), which is exactly
    what "match the bounded latent to a uniform" predicts.
    """
    grid = [1.0, 1.3, 1.6, 1.7, 2.0, 2.4]
    ks = [uniformity_error(a, 200_000)["ks"] for a in grid]
    best = grid[min(range(len(grid)), key=lambda i: ks[i])]
    assert best == pytest.approx(1.7)
    assert ks == sorted(ks[:grid.index(best) + 1], reverse=True) + \
        sorted(ks[grid.index(best) + 1:])          # decreasing then increasing
    assert uniformity_error(IFSQ_ALPHA_KS_OPTIMAL, 200_000)["ks"] < \
        uniformity_error(IFSQ_ALPHA, 200_000)["ks"]


def test_uniformity_error_is_deterministic():
    assert uniformity_error(1.6, 10_000, seed=3) == \
        uniformity_error(1.6, 10_000, seed=3)


# --------------------------------------------------------------------------- #
#  The quantizer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("levels", [[3], [5], [9], [5, 5, 5, 5], [9, 5, 5]])
def test_each_channel_takes_exactly_L_values(levels):
    q = IFSQ(levels)
    z = torch.linspace(-40, 40, 40001).reshape(-1, 1).repeat(1, len(levels))
    zhat, _ = q(z)
    for j, L in enumerate(levels):
        assert len(torch.unique(zhat[:, j])) == L


def test_grid_is_symmetric_with_an_exact_zero_centre():
    """L = 2K + 1 exists precisely so that 0 is a codeword."""
    codes = IFSQ([5]).channel_codes()[0]
    torch.testing.assert_close(codes, torch.tensor([-1., -.5, 0., .5, 1.]))
    torch.testing.assert_close(codes, -codes.flip(0))


def test_even_levels_are_rejected_with_an_explanation():
    with pytest.raises(ValueError, match="odd levels"):
        IFSQ([8, 5, 5, 5])
    with pytest.raises(ValueError, match="odd levels"):
        IFSQDAB(in_features=8, levels=[4, 4])


def test_extreme_levels_are_reachable_without_eps():
    """FSQ needs eps so tanh can reach the end bins; the sigmoid map does not."""
    q = IFSQ([5])
    assert q.eps == 0.0
    zhat, _ = q(torch.tensor([[-1e4], [1e4]]))
    torch.testing.assert_close(zhat.flatten(), torch.tensor([-1.0, 1.0]))


def test_alpha_two_reproduces_plain_fsq_for_odd_levels():
    """With alpha=2 and eps=0 the two quantizers must agree exactly."""
    ifsq = IFSQ([5, 5, 5], alpha=TANH_ALPHA, eps=0.0, index_order="little")
    fsq = FSQ([5, 5, 5], eps=0.0)
    z = torch.randn(500, 3) * 2
    torch.testing.assert_close(ifsq.quantize(z), fsq.quantize(z))


@pytest.mark.parametrize("order", ["big", "little"])
def test_index_round_trip(order):
    q = IFSQ([5, 5, 3], index_order=order)
    idx = torch.arange(q.codebook_size)
    torch.testing.assert_close(q.codes_to_indices(q.indices_to_codes(idx)), idx)


def test_big_endian_index_matches_the_papers_worked_example():
    """The paper's example: digits (2, 2, 1, 0) with L=3 give index 75."""
    q = IFSQ([3, 3, 3, 3], index_order="big")
    codes = q.indices_to_codes(torch.tensor([75]))
    assert q.per_channel_indices(codes).flatten().tolist() == [2, 2, 1, 0]


def test_index_order_defaults_differ_between_fsq_and_ifsq():
    assert IFSQ([5, 5]).index_order == "big"
    assert FSQ([5, 5]).index_order == "little"


# --------------------------------------------------------------------------- #
#  Bin occupancy: what the paper's scaling actually delivers
# --------------------------------------------------------------------------- #
def _maxdev(q):
    h = level_histogram(q, 200_000)
    return float((h - 1.0 / q.levels[0]).abs().max())


@pytest.mark.parametrize("L", [5, 9, 17, 33])
def test_paper_scaling_underfills_the_end_bins(L):
    """Uniform pre-rounding value does not give a uniform level histogram.

    Scaling by (L-1)/2 makes the two outermost bins half as wide as the
    interior ones, so they collect about half the mass. The consequence is
    counter-intuitive and worth pinning: on this measure vanilla FSQ is closer
    to uniform than iFSQ, because tanh's bimodality happens to compensate.
    """
    h = level_histogram(IFSQ([L]), 200_000)
    interior = h[1:-1].mean()
    assert h[0] < 0.62 * interior and h[-1] < 0.62 * interior
    assert _maxdev(IFSQ([L])) > _maxdev(FSQ([L]))


@pytest.mark.parametrize("L", [5, 7, 9, 17])
def test_equal_edge_bins_deliver_a_uniform_histogram(L):
    """`edge_bins="equal"` scales by L/2, making every bin the same width."""
    q = IFSQ([L], edge_bins="equal")
    h = level_histogram(q, 200_000)
    assert len(h) == L and (h > 0).all()
    assert float((h - 1.0 / L).abs().max()) < 0.02
    assert _maxdev(q) < _maxdev(IFSQ([L]))          # better than the paper's
    assert _maxdev(q) < _maxdev(FSQ([L]))           # ... and better than FSQ


@pytest.mark.parametrize("L", [5, 7, 9, 17])
def test_equal_edge_bins_still_produce_exactly_L_values(L):
    """L//2 can be odd, where round-half-to-even would overflow the range."""
    q = IFSQ([L], edge_bins="equal")
    z = torch.linspace(-40, 40, 40001).reshape(-1, 1)
    zhat, idx = q(z)
    assert len(torch.unique(zhat)) == L
    assert int(idx.min()) >= 0 and int(idx.max()) < L


def test_equal_edge_bins_raise_the_realised_entropy():
    import math
    L = 5
    ents = {}
    for name, q in (("fsq", FSQ([L])), ("paper", IFSQ([L])),
                    ("equal", IFSQ([L], edge_bins="equal"))):
        h = level_histogram(q, 200_000)
        ents[name] = float(-(h[h > 0] * h[h > 0].log2()).sum())
    assert ents["equal"] > ents["fsq"] > ents["paper"]
    assert ents["equal"] == pytest.approx(math.log2(L), abs=0.01)


def test_edge_bins_is_validated():
    with pytest.raises(ValueError, match="edge_bins"):
        IFSQ([5], edge_bins="nope")


def test_level_histogram_sums_to_one_and_is_deterministic():
    h = level_histogram(IFSQ([5, 9]), 20_000, channel=1)
    assert len(h) == 9
    assert float(h.sum()) == pytest.approx(1.0, abs=1e-5)
    torch.testing.assert_close(h, level_histogram(IFSQ([5, 9]), 20_000, channel=1))


def test_gradients_flow_through_quantize():
    q = IFSQ([5, 5])
    z = torch.randn(6, 2, requires_grad=True)
    q.quantize(z).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert (z.grad.abs() > 0).any()


# --------------------------------------------------------------------------- #
#  The bottleneck
# --------------------------------------------------------------------------- #
def test_ifsq_dab_is_a_dab_layer_with_the_same_contract():
    layer = IFSQDAB(in_features=12, levels=[5, 5, 5])
    assert isinstance(layer, DABLayer)
    assert layer.codebook_size == 125
    assert layer.ifsq_alpha == IFSQ_ALPHA
    out = layer(torch.randn(7, 12), training=True)
    assert out.latent.shape == (7, 3)
    assert out.distance.shape == (7,)
    assert out.indices.shape == (7,)
    assert int(out.indices.max()) < 125


def test_ifsq_dab_uses_the_sigmoid_bound_not_tanh():
    torch.manual_seed(0)
    layer = IFSQDAB(in_features=6, levels=[5], ifsq_alpha=IFSQ_ALPHA)
    x = torch.randn(32, 6)
    with torch.no_grad():
        params = layer._call(x)
        mu_raw, _ = params.split([1, 1], dim=-1)
        expected = ifsq_bound(mu_raw, IFSQ_ALPHA)     # half_width == L//2 here
        got = layer(x, training=False).mean
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_ifsq_dab_alpha_two_matches_fsq_dab():
    """alpha=2, eps=0 makes the two bottlenecks numerically identical."""
    from dab import FSQDAB
    torch.manual_seed(0)
    a = IFSQDAB(in_features=8, levels=[5, 5], ifsq_alpha=TANH_ALPHA,
                index_order="little")
    b = FSQDAB(in_features=8, levels=[5, 5], fsq_eps=0.0)
    b.load_state_dict(a.state_dict(), strict=False)
    with torch.no_grad():
        b.dense.weight.copy_(a.dense.weight)
    x = torch.randn(9, 8)
    with torch.no_grad():
        torch.testing.assert_close(a(x, training=False).distance,
                                   b(x, training=False).distance,
                                   rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("hard", [False, True])
def test_ifsq_dab_rdfc_fits_the_codebook(hard):
    from dab import build_optimizers, codebook_distortion, rdfc_epoch
    torch.manual_seed(0)
    layer = IFSQDAB(in_features=10, levels=[5, 5, 5], momentum=0.0, hard=hard)
    _, code_opt = build_optimizers(layer)
    assert code_opt is None                       # the grid is fixed
    before = layer.centroid_covariance.clone()
    data = [torch.randn(32, 10) for _ in range(3)]
    rdfc_epoch(layer, code_opt, lambda: iter(data),
               lambda b: codebook_distortion(layer(b, training=True).distance))
    assert bool(layer.initialized)
    assert not torch.allclose(layer.centroid_covariance, before)
    torch.testing.assert_close(layer.centroid_probs.sum(-1), torch.ones(3))


def test_ifsq_dab_rejects_an_external_quantizer():
    with pytest.raises(TypeError, match="builds its own quantizer"):
        IFSQDAB(in_features=8, levels=[5], quantizer=FSQ([5]))


def test_build_bottleneck_supports_ifsq():
    b = build_bottleneck("ifsq", 32, levels=[5, 5, 5, 5])
    assert isinstance(b, IFSQDAB) and b.codebook_size == 625
    assert isinstance(build_bottleneck("ifsq", 32, dab_dim=3), IFSQDAB)


def test_models_accept_the_ifsq_bottleneck():
    from dab.models import wide_resnet_dab
    m = wide_resnet_dab(depth=10, width_multiplier=1, num_classes=4,
                        bottleneck="ifsq", levels=[5, 5, 5])
    logits, out = m(torch.randn(2, 3, 32, 32), training=True)
    assert logits.shape == (2, 4) and out.indices.shape == (2,)


def test_state_dict_round_trip():
    torch.manual_seed(0)
    a = IFSQDAB(in_features=8, levels=[5, 5], momentum=0.0)
    a.mark_initialized(); a.reset_codebook()
    a(torch.randn(32, 8), training=True)
    a.set_codebook()
    b = IFSQDAB(in_features=8, levels=[5, 5])
    b.load_state_dict(a.state_dict())
    x = torch.randn(5, 8)
    torch.testing.assert_close(a(x, training=False).distance,
                               b(x, training=False).distance)


def test_ifsq_dab_accepts_equal_edge_bins():
    layer = IFSQDAB(in_features=8, levels=[5, 5], edge_bins="equal")
    assert layer.fsq.edge_bins == "equal"
    out = layer(torch.randn(6, 8), training=True)
    assert out.indices.shape == (6,) and int(out.indices.max()) < 25
