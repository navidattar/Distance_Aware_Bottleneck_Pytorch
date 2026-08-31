# coding=utf-8
"""Reference architectures wired up with a Distance-Aware Bottleneck."""
from .mlp_dab import MLPDAB
from .wide_resnet_dab import BasicBlock, WideResNetDAB, wide_resnet_dab

__all__ = ["MLPDAB", "WideResNetDAB", "wide_resnet_dab", "BasicBlock",
           "ResNet50DAB", "pretrained_resnet50_dab"]


def __getattr__(name):
    # torchvision is only needed for the ResNet-50 model; import it lazily so
    # the rest of the package works without it.
    if name in ("ResNet50DAB", "pretrained_resnet50_dab"):
        from . import resnet50_dab as _m
        return getattr(_m, name)
    raise AttributeError(name)
