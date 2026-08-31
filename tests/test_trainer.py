"""Tests for the two-phase (encoder/decoder + RDFC) training schedule."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dab import (NormalFullCovarianceDAB, build_optimizers,
                 codebook_distortion, codebook_parameters, dab_loss,
                 find_dab_layers, network_parameters, rdfc_epoch)
from dab.models import MLPDAB, WideResNetDAB
from dab.trainer import WarmUpPiecewiseConstantSchedule


class Tiny(nn.Module):
    def __init__(self, k=4, d=3):
        super().__init__()
        self.body = nn.Linear(6, 16)
        self.dab = NormalFullCovarianceDAB(16, dab_dim=d, codebook_size=k,
                                           momentum=0.0)
        self.head = nn.Linear(d, 2)

    def forward(self, x, training=None):
        out = self.dab(self.body(x), training=training)
        return self.head(out.latent), out


# --------------------------------------------------------------------------- #
def test_parameter_split_matches_the_reference_name_filter():
    m = Tiny()
    code = codebook_parameters(m)
    net = network_parameters(m)
    code_ids = {id(p) for p in code}
    assert [n for n, p in m.named_parameters()
            if id(p) in code_ids] == ["dab.centroid_means"]
    assert len(code) + len(net) == len(list(m.parameters()))
    assert not ({id(p) for p in code} & {id(p) for p in net})


def test_find_dab_layers():
    assert len(find_dab_layers(Tiny())) == 1
    assert len(find_dab_layers(WideResNetDAB(depth=10, width_multiplier=1))) == 1


def test_build_optimizers_uses_sgd_for_the_network_and_adam_for_the_codebook():
    opt, code_opt = build_optimizers(Tiny(), base_learning_rate=0.1,
                                     rdfc_learning_rate=0.05)
    assert isinstance(opt, torch.optim.SGD) and opt.defaults["nesterov"]
    assert isinstance(code_opt, torch.optim.Adam)
    assert code_opt.param_groups[0]["lr"] == 0.05
    assert len(code_opt.param_groups[0]["params"]) == 1


def test_codebook_optimizer_is_none_without_a_dab_layer():
    _, code_opt = build_optimizers(nn.Linear(3, 3))
    assert code_opt is None


# --------------------------------------------------------------------------- #
def _batches(n=8, bs=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    data = [torch.randn(bs, 6, generator=g) for _ in range(n)]
    return lambda: iter(data)


def test_rdfc_epoch_raises_initialized_and_commits_the_codebook():
    torch.manual_seed(0)
    m = Tiny()
    _, code_opt = build_optimizers(m, rdfc_learning_rate=0.05)
    batches = _batches()
    assert not bool(m.dab.initialized)
    before_cov = m.dab.centroid_covariance.clone()
    before_probs = m.dab.centroid_probs.clone()

    rdfc_epoch(m, code_opt, batches,
               lambda b: codebook_distortion(m(b, training=True)[1].distance))

    assert bool(m.dab.initialized)
    assert not torch.allclose(m.dab.centroid_covariance, before_cov)
    assert not torch.allclose(m.dab.centroid_probs, before_probs)
    assert float(m.dab.centroid_probs.sum()) == pytest.approx(1.0, abs=1e-4)
    assert (m.dab.centroid_probs >= 0).all()


def test_rdfc_priors_are_estimated_with_the_committed_covariance():
    """The second sub-pass must run *after* ``set_codebook_covariance``.

    The reference RDFC M-step has two passes: the first trains the centroids and
    accumulates the covariances, the second re-runs the E-step with the freshly
    committed covariances to estimate the priors. Collapsing them into a single
    pass would estimate the priors under a stale codebook.
    """
    torch.manual_seed(0)
    m = Tiny()
    _, code_opt = build_optimizers(m, rdfc_learning_rate=0.1)
    data = [torch.randn(16, 6) for _ in range(3)]

    seen = []
    handle = m.dab.register_forward_pre_hook(
        lambda mod, inp: seen.append(mod.centroid_covariance.clone()))
    rdfc_epoch(m, code_opt, lambda: iter(data),
               lambda b: codebook_distortion(m(b, training=True)[1].distance))
    handle.remove()

    assert len(seen) == 2 * len(data)
    final = m.dab.centroid_covariance
    # Every forward of the prior sub-pass saw the final, committed covariance.
    for cov in seen[len(data):]:
        torch.testing.assert_close(cov, final)
    # ... which is not the one the centroid sub-pass ran against.
    assert not torch.allclose(seen[0], final)


def test_rdfc_epoch_only_moves_the_codebook():
    torch.manual_seed(0)
    m = Tiny()
    _, code_opt = build_optimizers(m, rdfc_learning_rate=0.1)
    frozen = {n: p.clone() for n, p in m.named_parameters()
              if "centroid" not in n}
    means = m.dab.centroid_means.clone()
    rdfc_epoch(m, code_opt, _batches(),
               lambda b: codebook_distortion(m(b, training=True)[1].distance))
    for n, p in m.named_parameters():
        if "centroid" not in n:
            torch.testing.assert_close(p, frozen[n])
    assert not torch.allclose(m.dab.centroid_means, means)


def test_rdfc_epoch_reduces_the_codebook_distortion():
    torch.manual_seed(0)
    m = Tiny()
    _, code_opt = build_optimizers(m, rdfc_learning_rate=0.05)
    batches = _batches(n=12)
    fn = lambda b: codebook_distortion(m(b, training=True)[1].distance)
    first = rdfc_epoch(m, code_opt, batches, fn)
    for _ in range(4):
        last = rdfc_epoch(m, code_opt, batches, fn)
    assert last < first


def test_full_two_phase_epoch_runs_and_trains():
    """One realistic epoch: encoder/decoder phase, then RDFC phase."""
    torch.manual_seed(0)
    m = Tiny()
    opt, code_opt = build_optimizers(m, base_learning_rate=0.05,
                                     rdfc_learning_rate=0.05,
                                     weight_decay=1e-4)
    g = torch.Generator().manual_seed(1)
    xs = [torch.randn(32, 6, generator=g) for _ in range(10)]
    ys = [(x.sum(-1) > 0).long() for x in xs]

    losses = []
    for epoch in range(6):
        m.train()
        total = 0.0
        for x, y in zip(xs, ys):                       # phase 1
            opt.zero_grad(set_to_none=True)
            logits, out = m(x, training=True)
            loss = dab_loss(F.cross_entropy(logits, y), out.distance, beta=1e-3)
            loss.backward()
            opt.step()
            total += float(loss)
        losses.append(total / len(xs))
        rdfc_epoch(m, code_opt, lambda: iter(xs),   # phase 2
                   lambda b: codebook_distortion(m(b, training=True)[1].distance))
    assert losses[-1] < losses[0]
    assert torch.isfinite(m.dab.centroid_means).all()


def test_uncertainty_separates_in_and_out_of_distribution_inputs():
    """The core DAB claim, on a toy problem: far-away inputs score higher."""
    torch.manual_seed(0)
    m = MLPDAB(in_features=1, out_features=1, hidden=64, dab_dim=4,
               codebook_size=1, dab_tau=5.0, momentum=0.0)
    opt, code_opt = build_optimizers(m, base_learning_rate=0.0,
                                     rdfc_learning_rate=0.05, weight_decay=0.0)
    opt = torch.optim.Adam(network_parameters(m), lr=1e-3)
    x_in = torch.rand(512, 1) * 2 - 1                     # train on [-1, 1]
    y_in = x_in ** 3

    for _ in range(300):
        m.train()
        opt.zero_grad(set_to_none=True)
        pred, out = m(x_in, training=True)
        loss = dab_loss(F.mse_loss(pred, y_in), out.distance, beta=1e-2)
        loss.backward()
        opt.step()
        rdfc_epoch(m, code_opt, lambda: iter([x_in]),
                   lambda b: codebook_distortion(m(b, training=True)[1].distance))

    m.eval()
    with torch.no_grad():
        d_in = m(x_in, training=False)[1].distance.mean()
        d_out = m(torch.linspace(6, 10, 512).reshape(-1, 1),
                  training=False)[1].distance.mean()
    assert d_out > d_in


# --------------------------------------------------------------------------- #
def test_warmup_piecewise_constant_schedule():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=0.4)
    sched = WarmUpPiecewiseConstantSchedule(opt, steps_per_epoch=10,
                                            decay_ratio=0.1,
                                            decay_epochs=[3, 5],
                                            warmup_epochs=1)
    lrs = []
    for _ in range(70):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.0)          # warmup starts at 0
    assert lrs[5] == pytest.approx(0.2)          # halfway through warmup
    assert lrs[10] == pytest.approx(0.4)         # base lr after warmup
    assert lrs[29] == pytest.approx(0.4)
    assert lrs[30] == pytest.approx(0.04)        # first decay at epoch 3
    assert lrs[50] == pytest.approx(0.004)       # second decay at epoch 5
