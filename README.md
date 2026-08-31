# Distance-Aware Bottleneck — PyTorch

A faithful, dependency-light PyTorch implementation of the **Distance-Aware
Bottleneck (DAB)** from

> Ifigeneia Apostolopoulou, Benjamin Eysenbach, Frank Nielsen, Artur Dubrawski.
> **A Rate-Distortion View of Uncertainty Quantification.** ICML 2024.
> [arXiv:2406.10775](https://arxiv.org/abs/2406.10775)

ported from the authors' reference TensorFlow code at
[`ifiaposto/Distance_Aware_Bottleneck`](https://github.com/ifiaposto/Distance_Aware_Bottleneck).

DAB gives a deterministic network a **single-forward-pass uncertainty score**.
It learns a small codebook of Gaussians that summarises the training data in
latent space; the distance from an input's encoder distribution to that codebook
is the uncertainty. No ensembles, no MC dropout, no second forward pass, and no
change to the prediction head.

The port is line-by-line auditable against the original: attribute names, buffer
semantics, update equations and the two-phase training schedule all match. Every
deviation is listed in [`docs/FAITHFULNESS.md`](docs/FAITHFULNESS.md).

---

## Contents

- [Install](#install)
- [30-second example](#30-second-example)
- [How DAB works](#how-dab-works)
- [Adding DAB to your own model](#adding-dab-to-your-own-model)
- [The training loop](#the-training-loop)
- [API reference](#api-reference)
- [Choosing hyper-parameters](#choosing-hyper-parameters)
- [Using the uncertainty score](#using-the-uncertainty-score)
- [Reference models and examples](#reference-models-and-examples)
- [Multi-GPU](#multi-gpu)
- [Checkpointing](#checkpointing)
- [Troubleshooting](#troubleshooting)
- [Tests](#tests)
- [Citation and license](#citation-and-license)

---

## Install

Requires Python ≥ 3.8 and PyTorch ≥ 1.13. There are no other required
dependencies — `torchvision` is needed only for the ResNet-50 model and the
CIFAR-10 example, `matplotlib` only for the synthetic demo's plot.

```bash
git clone https://github.com/navidattar/Distance_Aware_Bottleneck_Pytorch.git
cd Distance_Aware_Bottleneck_Pytorch
pip install -e .
```

Or just copy the `dab/` directory into your project — it is self-contained.

---

## 30-second example

```python
import torch
from dab import NormalFullCovarianceDAB

layer = NormalFullCovarianceDAB(in_features=640, dab_dim=8, codebook_size=10)

features = torch.randn(32, 640)      # pooled backbone features
out = layer(features)                # -> DABOutput

out.latent      # [32, 8]  feed this to your prediction head
out.distance    # [32]     per-example uncertainty (higher = more novel)
```

That is the whole inference story: one forward pass, one extra scalar per
example. Training needs the two-phase loop described below — without it the
codebook never moves and the distance is meaningless.

---

## How DAB works

DAB replaces the usual deterministic bottleneck with a *distributional* one and
adds a learned codebook of Gaussians.

**1. The encoder is stochastic.** A dense layer maps features to the parameters
of a Gaussian `N(μ(x), Σ(x))` over the `d`-dimensional latent. Two
parameterisations are provided:

| Layer | Encoder covariance | Encoder outputs | Used in the paper for |
| --- | --- | --- | --- |
| `NormalDiagCovarianceDAB` | diagonal | `2d` | ImageNet / ResNet-50 |
| `NormalFullCovarianceDAB` | full | `d + d(d+1)/2` | CIFAR-10 / WideResNet, synthetic |

Scales are parameterised as `softplus(raw − 5) + 1e-5`; the shift starts the
model near-deterministic and lets the noise grow only if it helps.

**2. A codebook of `K` Gaussians summarises the training set.** Each centroid
`k` has a mean `m_k` (a trainable parameter), a covariance `Σ_k` and a prior
probability `π_k` (both updated in closed form, not by gradient descent).

**3. Distance is a KL divergence.** For each centroid,
`D_k(x) = KL(N(μ(x), Σ(x)) ‖ N(m_k, Σ_k))`. Soft assignments come from a Gibbs
posterior,

```
a_k(x) = softmax_k( log π_k − τ · D_k(x) )
```

and the reported uncertainty is the expected distance from the codebook:

```
d(x) = Σ_k a_k(x) · D_k(x)
```

The assignment is treated as a constant (stop-gradient) — it is the E-step of an
EM procedure, not something the network optimises through.

**4. The loss is rate–distortion.** The network minimises
`NLL + β · mean(d(x))`, so it must both predict well and keep its encoders close
to a *finite* set of codes. That pressure is what makes `d(x)` informative: an
input unlike anything in training cannot be squeezed near any code, so its
distance is large.

**5. The codebook is fitted by an EM-style M-step (RDFC).** Once per epoch, the
centroid means are updated by gradient descent on the mean distance, and the
covariances and priors are recomputed in closed form (Davis & Dhillon's optimal
Gaussian codebook, with a Dirichlet(5) smoothing of the responsibilities).

---

## Adding DAB to your own model

Put the layer where your pooled feature vector meets your prediction head. It
replaces that projection — you do not need a separate bottleneck.

```python
import torch.nn as nn
from dab import NormalFullCovarianceDAB

class MyModel(nn.Module):
    def __init__(self, num_classes=10, dab_dim=8, codebook_size=10):
        super().__init__()
        self.backbone = my_backbone()                     # -> [B, F]
        self.dab = NormalFullCovarianceDAB(
            in_features=self.backbone.out_dim,
            dab_dim=dab_dim,
            codebook_size=codebook_size,
            dab_tau=1.0,
        )
        self.head = nn.Linear(dab_dim, num_classes)

    def forward(self, x, training=None):
        out = self.dab(self.backbone(x), training=training)
        return self.head(out.latent), out
```

Notes:

- **`training`** controls two things: whether the latent is *sampled* from the
  encoder Gaussian (train) or set to the mean (eval), and whether the codebook's
  moving averages accumulate. Leave it `None` and the layer follows
  `module.training`, like any other PyTorch module. Pass it explicitly when you
  want the reference code's behaviour of running the E-step accumulation from
  inside a `torch.no_grad()` block.
- **The layer applies its own activation** to the incoming features
  (`activation="relu"` by default, matching the reference layer). If the
  preceding layer is already activated, pass `activation=None`.
- **No bias by default** (`use_bias=False`), matching the reference.
- The layer works for regression, classification, or anything else — it only
  ever sees features and returns a latent plus a scalar.

---

## The training loop

DAB alternates two phases per epoch. **The order matters**, and so does running
the codebook phase as two sub-passes.

```python
from dab import build_optimizers, codebook_distortion, dab_loss, rdfc_epoch

model = MyModel().cuda()
optimizer, codebook_optimizer = build_optimizers(
    model, base_learning_rate=0.1, rdfc_learning_rate=0.1, weight_decay=2e-4)

def distortion(batch):
    """Codebook loss for one batch. Must run a full forward with training=True."""
    x, _ = batch
    _, out = model(x.cuda(), training=True)
    return codebook_distortion(out.distance)

for epoch in range(num_epochs):
    model.train()

    # ---- phase 1: encoder + decoder, on NLL + beta * distortion ----
    for x, y in train_loader:
        optimizer.zero_grad(set_to_none=True)
        logits, out = model(x.cuda(), training=True)
        loss = dab_loss(F.cross_entropy(logits, y.cuda()), out.distance, beta=1e-3)
        loss.backward()
        optimizer.step()

    # ---- phase 2: the codebook (RDFC M-step) ----
    rdfc_epoch(model, codebook_optimizer, lambda: iter(train_loader), distortion)
```

`build_optimizers` splits the parameters exactly as the reference code does:
everything whose name contains `centroid` goes to the codebook optimiser (Adam),
everything else to the network optimiser (SGD with Nesterov momentum). The two
never touch each other's parameters.

### What `rdfc_epoch` does

1. Sets `initialized = True`. Until this happens — i.e. for the whole of the
   first encoder/decoder epoch — assignments are **uniform**, which bootstraps
   the codebook from a random-assignment state.
2. **Sub-pass (a):** resets the covariance moving average, walks the data
   training only the centroid means on `mean(d(x))` while accumulating the
   closed-form covariance estimate, then commits the covariances (and refreshes
   the cached precision matrices and log-determinants).
3. **Sub-pass (b):** resets the prior moving average, walks the data again under
   `torch.no_grad()` — re-running the E-step against the *freshly committed*
   covariances — and commits the priors.

Collapsing (a) and (b) into a single pass estimates the priors under a stale
codebook and is the most common way to get a port of DAB subtly wrong. `batches`
is therefore a *callable* returning a fresh iterator, because it is consumed
twice.

If you want to drive the phases yourself, the primitives are public:
`mark_initialized()`, `reset_codebook_covariance()`, `set_codebook_covariance()`,
`reset_centroid_probs()`, `set_centroid_probs()` (plus `reset_codebook()` /
`set_codebook()` which do both).

### Calibrated distortion (the ImageNet variant)

The reference ImageNet run adds a margin term: the codebook should quantise the
encoders of *correctly classified* points, while misclassified points closer
than `uncertainty_lb` are pushed away. Pass a correctness indicator:

```python
matches = (logits.argmax(-1) == y).float()
loss = dab_loss(nll, out.distance, beta=0.01,
                matches=matches, uncertainty_lb=100.0)
```

Use the same `matches` in the RDFC `distortion` function. This turns the
distance into a *calibrated* score (it also flags likely mistakes on
in-distribution data) rather than a pure novelty score.

---

## API reference

### `NormalDiagCovarianceDAB` / `NormalFullCovarianceDAB`

```python
NormalFullCovarianceDAB(
    in_features,            # size of the incoming feature vector
    dab_dim,                # latent dimension d
    codebook_size,          # number of centroids K
    dab_tau=1.0,            # temperature multiplying the distance in the E-step
    momentum=0.99,          # moving-average momentum for covariances and priors
    activation="relu",      # applied to inputs before the encoder's dense layer
    use_bias=False,
    kernel_initializer="glorot_uniform",   # or "he_normal"
    var_shift=5.0,          # scale = softplus(raw - var_shift) + var_floor
    var_floor=1e-5,
    codebook_jitter=0.0,    # full-cov only; eps*I added before inverting
    covariance_floor=0.0,   # diag-cov only; lower bound on codebook variances
    generator=None,         # torch.Generator for reproducible init
)
```

`forward(inputs, training=None) -> DABOutput`.

### `DABOutput`

| Field | Shape | Meaning |
| --- | --- | --- |
| `latent` | `[B, d]` | features for the head; a sample when training, the mean at eval |
| `distance` (alias `uncertainty`) | `[B]` | distance from the codebook — **the uncertainty score** |
| `mean` | `[B, d]` | encoder mean `μ(x)` |
| `variance` | `[B, d]` | encoder variances (diagonal of `Σ(x)` for the full-cov layer) |
| `covariance` | `[B, d, d]` | encoder covariance (full-cov layer only) |
| `distances_from_centroids` | `[B, K]` | per-centroid `KL(encoder ‖ centroid)` |
| `assignment` | `[B, K]` | E-step responsibilities (always detached) |

`out.as_tensor()` returns `concat([latent, distance[:, None]], -1)` — the exact
tensor the reference Keras layer emits, if you are porting downstream code.

### Codebook state (buffers, saved in `state_dict`)

`centroid_means` (parameter), `centroid_probs`, `centroid_probs_mavg`,
`centroid_covariance`, `centroid_covariance_mavg`, `initialized`, and for the
full-covariance layer `centroid_precision`,
`centroid_covariance_log_abs_det`.

### Functions

| Symbol | Purpose |
| --- | --- |
| `dab_loss(nll, uncertainty, beta, matches=None, uncertainty_lb=100.0)` | `NLL + β · distortion` |
| `codebook_distortion(uncertainty, matches=None, uncertainty_lb=100.0)` | the distortion term alone (the RDFC loss) |
| `rdfc_epoch(model, codebook_optimizer, batches, distortion_fn, grad_clip=None)` | one full codebook phase |
| `build_optimizers(model, ...)` | the reference SGD/Adam optimiser pair |
| `codebook_parameters(model)` / `network_parameters(model)` | the RDFC parameter split |
| `find_dab_layers(model)` | every DAB layer in a module tree |
| `WarmUpPiecewiseConstantSchedule(optimizer, ...)` | the reference LR schedule (step it per optimiser step) |
| `kl_normal_diag`, `kl_normal_full`, `fill_triangular` | the underlying tensor primitives |

---

## Choosing hyper-parameters

The reference settings, for orientation:

| | CIFAR-10 (WRN-28-10) | ImageNet (ResNet-50) | Synthetic 1-D |
| --- | --- | --- | --- |
| `dab_dim` | 8 | 4 | 8 |
| `codebook_size` | 10 | 1000 | 1 |
| `dab_tau` | 1.0 | 2.0 | 5.0 |
| `beta` | 0.001 | 0.01 | 1.0 |
| `momentum` | 0.99 | 0.99 | 0.0 |
| covariance | full | diagonal | full |
| network optimiser | SGD, nesterov, lr 0.1·(B/128) | SGD, nesterov, lr 0.01 | Adam, lr 1e-3 |
| codebook optimiser | Adam, lr 0.1 | Adam, lr 0.1 | Adam, lr 1e-3 |

Practical guidance:

- **`beta`** is the main knob. Too small and the encoders spread out, so the
  distance stops discriminating; too large and accuracy suffers. Start at
  `1e-3` and tune on a validation OOD proxy, not on accuracy alone.
- **`codebook_size`** can be small. Ten codes are enough for CIFAR-10; the codes
  are Gaussians, not points, so they cover a lot of ground. Scale it up only for
  genuinely multi-modal data (the ImageNet run uses one per class).
- **`dab_dim`** is small on purpose — the bottleneck is what forces the
  quantisation to be informative. 4–16 is the useful range.
- **`dab_tau`** sharpens the assignment. `τ → ∞` is hard nearest-code
  assignment; `τ → 0` is uniform. `1.0`–`5.0` is the tested range.
- **`momentum`** must be matched to the number of RDFC steps per epoch. The
  moving average restarts from zero each phase, so it needs roughly
  `1/(1 − momentum)` steps to converge — `0.99` needs ~100+ steps per epoch. For
  full-batch or very small datasets use `momentum=0.0`.
- **Diagonal vs full covariance:** full is more expressive and costs
  `O(B·K·d²)` per forward plus a `d×d` SVD per example; diagonal is
  `O(B·K·d)`. Use full for small `d`, diagonal for large `K`.

---

## Using the uncertainty score

`out.distance` is a positive, unbounded, **relative** score — it is a KL
divergence in nats, not a probability. Use it by ranking, not by absolute value:

```python
model.eval()
with torch.no_grad():
    logits, out = model(x, training=False)
    prediction = logits.argmax(-1)
    novelty = out.distance          # rank these
```

Typical uses:

- **OOD detection.** Threshold or rank `distance`; report AUROC against an OOD
  set. `examples/train_cifar10.py` does this against SVHN and CIFAR-100.
- **Misclassification detection / selective prediction.** Rank test points by
  `distance` and abstain on the tail. Train with the `matches` variant of
  `codebook_distortion` if this is the primary use case.
- **Active learning / dataset curation.** High distance marks inputs the
  training set does not cover.

Pick the threshold on a held-out in-distribution split (e.g. the 95th percentile
of ID distances gives a 5% false-positive rate). It is not transferable between
runs, models, or `beta` settings.

---

## Reference models and examples

```python
from dab.models import wide_resnet_dab, pretrained_resnet50_dab, MLPDAB

# CIFAR-10 model from the paper: WRN-28-10 + full-covariance DAB
model = wide_resnet_dab(depth=28, width_multiplier=10, num_classes=10,
                        dab_dim=8, codebook_size=10, dab_tau=1.0)
logits, out = model(images)

# ImageNet model: frozen pretrained ResNet-50 + MLP head + diagonal DAB
model = pretrained_resnet50_dab(num_classes=1000, dab_dim=4,
                                codebook_size=1000, dab_tau=2.0,
                                backpropagate=False)
```

All three expose `forward(x, training=None) -> (logits, DABOutput)`, and the
image models also expose `forward_concat(x)` returning
`concat([logits, distance], -1)` for parity with the reference Keras models.

**Runnable examples:**

```bash
# 1-D regression: uncertainty widens exactly where there is no training data
python examples/synthetic_regression_demo.py --example 1 --out demo.png

# CIFAR-10 with the paper's hyper-parameters (200 epochs)
python examples/train_cifar10.py --data_root ./data

# quick sanity run
python examples/train_cifar10.py --epochs 5 --depth 16 --width_multiplier 4
```

---

## Multi-GPU

The codebook statistics are reduced across ranks, mirroring the reference
implementation's `all_reduce` / `all_gather`. Wrap the model in
`DistributedDataParallel` as usual, and:

- use `drop_last=True` so every rank sees the same batch size (the gather
  assumes it, as the reference does);
- reach the DAB layer through `model.module` for the phase calls, or use
  `find_dab_layers(model)`, which unwraps DDP for you;
- `rdfc_epoch` accepts a DDP-wrapped module directly.

Sub-pass (b) runs under `no_grad`, so mark it accordingly if you use
`no_sync()` optimisations elsewhere.

---

## Checkpointing

Everything the codebook needs lives in the module's `state_dict` — parameters
*and* buffers, including the `initialized` flag. `torch.save(model.state_dict())`
and `load_state_dict` round-trip exactly; a resumed run continues with the
committed codebook rather than restarting from uniform assignments.

If you fine-tune a DAB model on new data, remember the codebook describes the
*old* training distribution until you run RDFC phases on the new one.

---

## Troubleshooting

**`RuntimeError: DAB codebook covariance is singular…`**
The covariance moving average had not converged when it was committed. Lower
`momentum`, run more RDFC steps per epoch (larger dataset or smaller batch), or
set a small `codebook_jitter` (e.g. `1e-6`).

**The distance is the same for every input (AUROC ≈ 0.5).**
Usually the codebook phase never ran. Check that `initialized` is `True` after
the first epoch and that `centroid_probs` is no longer uniform. Also check that
your `distortion_fn` really runs the model with `training=True` — the moving
averages only accumulate then.

**Accuracy collapses.**
`beta` is too large, or the codebook optimiser is touching the network. Verify
the split with `codebook_parameters(model)` — it should return exactly the
`centroid_means` tensors.

**The distance is huge (1e4+) early in training.**
Expected. The codebook covariance starts at the identity while encoder scales
start near `1e-2`, so the initial KL is dominated by the log-determinant ratio.
It drops sharply after the first RDFC phase.

**NaNs in the full-covariance layer.**
The encoder's SVD backward is unstable when singular values coincide, which the
reference implementation shares. Try the diagonal layer, a smaller `dab_dim`, or
gradient clipping via `rdfc_epoch(..., grad_clip=...)`.

---

## Tests

```bash
pip install pytest
pytest tests/ -q
```

71 tests covering: `fill_triangular` against the tfp docstring values, both KL
divergences against `torch.distributions`, the encoder parameterisation, the
E-step formula and its stop-gradient, uniform assignment before initialisation,
both closed-form covariance updates against explicit loops, the moving-average
formulas, the RDFC phase ordering (including that the priors are estimated with
the committed covariance), the parameter split, full-covariance sampling against
its empirical covariance, `state_dict` round-trips, and the LR schedule.

---

## Citation and license

```bibtex
@inproceedings{apostolopoulou2024rate,
  title     = {A Rate-Distortion View of Uncertainty Quantification},
  author    = {Apostolopoulou, Ifigeneia and Eysenbach, Benjamin and
               Nielsen, Frank and Dubrawski, Artur},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024}
}
```

Apache License 2.0, matching the original repository. The original TensorFlow
implementation is © 2024 Ifigeneia Apostolopoulou; this port keeps that notice
in every ported file. The WideResNet backbone follows
[Uncertainty Baselines](https://github.com/google/uncertainty-baselines/).
