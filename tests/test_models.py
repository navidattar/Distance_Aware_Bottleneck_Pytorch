"""Smoke tests for the reference architectures."""
import pytest
import torch

from dab.models import MLPDAB, WideResNetDAB, wide_resnet_dab


@pytest.mark.parametrize("version", [1, 2])
def test_wide_resnet_dab_forward(version):
    m = wide_resnet_dab(depth=10, width_multiplier=1, num_classes=7, dab_dim=3,
                        codebook_size=4, version=version)
    x = torch.randn(2, 3, 32, 32)
    logits, out = m(x, training=True)
    assert logits.shape == (2, 7)
    assert out.distance.shape == (2,)
    assert m.forward_concat(x, training=False).shape == (2, 8)


def test_wide_resnet_depth_validation():
    with pytest.raises(ValueError):
        wide_resnet_dab(depth=27)


def test_wide_resnet_dab_28_10_shapes():
    m = wide_resnet_dab(depth=28, width_multiplier=10, dab_dim=8,
                        codebook_size=10)
    assert m.feature_dim == 640
    assert m.dab.units == 8 + 8 * 9 // 2
    with torch.no_grad():
        logits, out = m(torch.randn(2, 3, 32, 32), training=False)
    assert logits.shape == (2, 10) and out.distance.shape == (2,)


def test_mlp_dab_forward_and_backward():
    m = MLPDAB(in_features=1, out_features=1, dab_dim=4, codebook_size=1)
    pred, out = m(torch.randn(5, 1), training=True)
    assert pred.shape == (5, 1)
    (pred.sum() + out.distance.sum()).backward()
    assert m.dab.centroid_means.grad is not None


def test_wide_resnet_batchnorm_defaults_match_the_reference():
    m = wide_resnet_dab(depth=10, width_multiplier=1)
    bns = [x for x in m.modules() if isinstance(x, torch.nn.BatchNorm2d)]
    assert bns and all(b.eps == 1e-5 and b.momentum == 0.1 for b in bns)
