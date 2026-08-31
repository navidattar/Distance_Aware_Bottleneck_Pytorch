# coding=utf-8
# Copyright 2024 Ifigeneia Apostolopoulou (original TensorFlow implementation)
# Copyright 2026 PyTorch port authors
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Train and evaluate WideResNet-28-10 + DAB on CIFAR-10.

Port of ``run_cifar.py``. Defaults reproduce the reference hyper-parameters
(``hparams/cifar_hparams.py``). Reports in-distribution accuracy/NLL and OOD
detection AUROC against SVHN and CIFAR-100 using the codebook distance as the
uncertainty score.

Usage::

    python examples/train_cifar10.py --data_root ./data
    python examples/train_cifar10.py --epochs 5 --width_multiplier 2   # quick check

Swap the reference codebook for a Finite Scalar Quantization grid with
``--bottleneck fsq`` (optionally ``--levels 8 5 5 5``)::

    python examples/train_cifar10.py --bottleneck fsq --levels 8 5 5 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dab import (WarmUpPiecewiseConstantSchedule, build_optimizers,
                 codebook_distortion, dab_loss)
from dab.models import wide_resnet_dab
from dab.trainer import rdfc_epoch

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def loaders(data_root: str, batch_size: int, workers: int = 8):
    from torchvision import datasets, transforms
    norm = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])

    def dl(ds, shuffle, drop_last):
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
            pin_memory=True, drop_last=drop_last, persistent_workers=workers > 0)

    train = datasets.CIFAR10(data_root, train=True, download=True,
                             transform=train_tf)
    test = datasets.CIFAR10(data_root, train=False, download=True,
                            transform=test_tf)
    ood = {
        "cifar100": datasets.CIFAR100(data_root, train=False, download=True,
                                      transform=test_tf),
        "svhn": datasets.SVHN(data_root, split="test", download=True,
                              transform=test_tf),
    }
    return (dl(train, True, True), dl(test, False, False),
            {k: dl(v, False, False) for k, v in ood.items()})


def auroc(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    """Rank-based AUROC for "OOD scores higher than ID scores"."""
    s = np.concatenate([scores_id, scores_ood])
    y = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits_all, unc_all, labels_all = [], [], []
    for x, y in loader:
        logits, out = model(x.to(device, non_blocking=True), training=False)
        logits_all.append(logits.float().cpu())
        unc_all.append(out.distance.float().cpu())
        labels_all.append(y)
    return (torch.cat(logits_all), torch.cat(unc_all), torch.cat(labels_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--base_learning_rate", type=float, default=0.1,
                    help="scaled by batch_size/128, as in the reference code")
    ap.add_argument("--rdfc_learning_rate", type=float, default=0.1)
    ap.add_argument("--lr_decay_ratio", type=float, default=0.2)
    ap.add_argument("--lr_decay_epochs", type=int, nargs="*",
                    default=[60, 120, 160])
    ap.add_argument("--lr_warmup_epochs", type=int, default=1)
    ap.add_argument("--one_minus_momentum", type=float, default=0.1)
    ap.add_argument("--l2", type=float, default=2e-4)
    ap.add_argument("--beta", type=float, default=0.001)
    ap.add_argument("--dab_tau", type=float, default=1.0)
    ap.add_argument("--dab_dim", type=int, default=8)
    ap.add_argument("--codebook_size", type=int, default=10)
    ap.add_argument("--bottleneck", default="full", choices=("full", "diag", "fsq"),
                    help="codebook: 'full'/'diag' are the reference DAB codebook, "
                         "'fsq' is the Finite Scalar Quantization product grid")
    ap.add_argument("--levels", type=int, nargs="*", default=None,
                    help="--bottleneck fsq only: values per coordinate "
                         "(default: 5 per coordinate)")
    ap.add_argument("--depth", type=int, default=28)
    ap.add_argument("--width_multiplier", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = args.device

    train_loader, test_loader, ood_loaders = loaders(
        args.data_root, args.batch_size, args.workers)

    model = wide_resnet_dab(depth=args.depth,
                            width_multiplier=args.width_multiplier,
                            num_classes=10, dab_dim=args.dab_dim,
                            codebook_size=args.codebook_size,
                            dab_tau=args.dab_tau,
                            bottleneck=args.bottleneck,
                            levels=args.levels).to(device)
    print(model.dab)

    # The reference code scales the learning rate and the decay epochs linearly
    # from a batch size of 128.
    base_lr = args.base_learning_rate * args.batch_size / 128
    decay_epochs = [(e * args.epochs) // 200 for e in args.lr_decay_epochs]
    optimizer, codebook_optimizer = build_optimizers(
        model, base_learning_rate=base_lr,
        rdfc_learning_rate=args.rdfc_learning_rate,
        momentum=1.0 - args.one_minus_momentum, nesterov=True,
        weight_decay=args.l2)
    scheduler = WarmUpPiecewiseConstantSchedule(
        optimizer, steps_per_epoch=len(train_loader),
        decay_ratio=args.lr_decay_ratio, decay_epochs=decay_epochs,
        warmup_epochs=args.lr_warmup_epochs)

    def distortion(batch):
        x, _ = batch
        _, out = model(x.to(device, non_blocking=True), training=True)
        return codebook_distortion(out.distance)

    for epoch in range(args.epochs):
        t0 = time.time()

        # ---- phase 1: train the encoder and the decoder -------------------
        model.train()
        total, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, out = model(x, training=True)
            loss = dab_loss(F.cross_entropy(logits, y), out.distance, args.beta)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss) * y.numel()
            correct += int((logits.argmax(-1) == y).sum())
            seen += y.numel()

        # ---- phase 2: fit the codebook (RDFC) -----------------------------
        # codebook_optimizer is None for an FSQ grid -- it has no trainable
        # parameters, but the closed-form covariance/prior M-steps still run.
        rdfc_loss = rdfc_epoch(model, codebook_optimizer,
                               lambda: iter(train_loader), distortion)

        print(f"epoch {epoch:3d}  loss {total / seen:.4f}  "
              f"acc {100 * correct / seen:5.2f}%  rdfc {rdfc_loss:.4f}  "
              f"lr {optimizer.param_groups[0]['lr']:.4f}  "
              f"{time.time() - t0:.0f}s")

    # ---- evaluation --------------------------------------------------------
    logits, unc_id, labels = evaluate(model, test_loader, device)
    acc = float((logits.argmax(-1) == labels).float().mean())
    nll = float(F.cross_entropy(logits, labels))
    print(f"\ntest accuracy {100 * acc:.2f}%   test NLL {nll:.4f}")

    correct = (logits.argmax(-1) == labels).numpy().astype(bool)
    print(f"misclassification-detection AUROC "
          f"{auroc(unc_id.numpy()[correct], unc_id.numpy()[~correct]):.4f}")
    for name, loader in ood_loaders.items():
        _, unc_ood, _ = evaluate(model, loader, device)
        print(f"OOD AUROC ({name}) {auroc(unc_id.numpy(), unc_ood.numpy()):.4f}")


if __name__ == "__main__":
    main()
