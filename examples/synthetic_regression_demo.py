# coding=utf-8
# Copyright 2024 Ifigeneia Apostolopoulou (original TensorFlow implementation)
# Copyright 2026 PyTorch port authors
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Synthetic 1-D regression demo -- the smallest complete DAB training loop.

Port of ``synthetic_regression_demo.py`` from the reference repository. A small
MLP is fitted to ``y = x^3 + noise`` on data that covers only part of the input
range; the DAB codebook distance is then plotted as an uncertainty band. It
should widen exactly where there is no training data.

Usage::

    python examples/synthetic_regression_demo.py --example 1 --out demo1.png
    python examples/synthetic_regression_demo.py --example 2 --out demo2.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dab import codebook_distortion, dab_loss, network_parameters, rdfc_epoch
from dab.models import MLPDAB


def create_dataset(num_train_examples: int, example: int, rng):
    """Two disjoint input regions, so the gap is genuinely out-of-distribution."""
    lo, hi = ((-4.0, 4.0) if example == 1 else (-5.0, 5.0))
    inner = 0.0 if example == 1 else 2.0
    x1 = rng.uniform(lo, -inner, size=num_train_examples // 2)
    x2 = rng.uniform(inner, hi, size=num_train_examples // 2)
    x = np.concatenate([x1, x2]).reshape(-1, 1)
    y = x ** 3 + rng.normal(0.0, 9.0, size=x.shape)
    x_test = np.linspace(-5, 5, 50).reshape(-1, 1)
    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(x_test, dtype=torch.float32),
            torch.tensor(x_test ** 3, dtype=torch.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--num_train_examples", type=int, default=50)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--rdfc_learning_rate", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--dab_tau", type=float, default=5.0)
    ap.add_argument("--dab_dim", type=int, default=8)
    ap.add_argument("--codebook_size", type=int, default=1)
    ap.add_argument("--example", type=int, default=1, choices=(1, 2))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="synthetic_regression_demo.png")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    x_train, y_train, x_test, y_test = create_dataset(
        args.num_train_examples, args.example, rng)

    # The demo uses full-batch steps, so momentum=0.0: each "batch" is the whole
    # dataset and the moving average would only slow the codebook down.
    model = MLPDAB(in_features=1, out_features=1, hidden=100, num_hidden=2,
                   dab_dim=args.dab_dim, codebook_size=args.codebook_size,
                   dab_tau=args.dab_tau, momentum=0.0, dab_activation=None)

    optimizer = torch.optim.Adam(network_parameters(model),
                                 lr=args.learning_rate)
    codebook_optimizer = torch.optim.Adam(
        [model.dab.centroid_means], lr=args.rdfc_learning_rate)

    def distortion(batch):
        return codebook_distortion(model(batch, training=True)[1].distance)

    for epoch in range(args.epochs):
        model.train()

        # -- phase 1: encoder & decoder --------------------------------------
        optimizer.zero_grad(set_to_none=True)
        pred, out = model(x_train, training=True)
        # The reference demo models p(y|x) as N(prediction, 1), so the NLL is
        # 0.5 * squared error up to an additive constant.
        nll = 0.5 * F.mse_loss(pred, y_train, reduction="none").sum(-1).mean()
        loss = dab_loss(nll, out.distance, beta=args.beta)
        loss.backward()
        optimizer.step()

        # -- phase 2: RDFC (the codebook) ------------------------------------
        rdfc_epoch(model, codebook_optimizer, lambda: iter([x_train]),
                   distortion)

        if epoch % 500 == 0:
            print(f"epoch {epoch:5d}  loss {float(loss):9.3f}  "
                  f"distance {float(out.distance.mean()):9.3f}")

    model.eval()
    with torch.no_grad():
        y_pred, out = model(x_test, training=False)
    y_pred = y_pred.squeeze(-1).numpy()
    uncertainty = out.distance.numpy()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the plot")
        print("uncertainty over x in [-5, 5]:", np.round(uncertainty, 2))
        return

    xt = x_test.squeeze(-1).numpy()
    plt.figure(figsize=(5, 4))
    plt.fill_between(xt, y_pred - 2 * uncertainty, y_pred + 2 * uncertainty,
                     color="coral", alpha=0.5, label="Uncertainty")
    plt.plot(xt, y_pred, c="royalblue", label="Prediction", linewidth=2)
    plt.scatter(x_train.squeeze(-1), y_train.squeeze(-1), c="navy", s=12,
                label="Train datapoint")
    plt.plot(xt, y_test.squeeze(-1), c="grey", label="Ground truth", linewidth=2)
    plt.legend(loc="upper center")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
