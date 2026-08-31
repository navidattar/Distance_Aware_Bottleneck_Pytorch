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

**Three codebooks, one interface.** Alongside the paper's explicit Gaussian
centroids, the package ships two optional quantized codebooks in which the codes
become the Cartesian product of per-coordinate scalar levels — *zero* codebook
parameters, no possibility of collapse, and discrete tokens for free:

- **FSQ** — [Mentzer et al., ICLR 2024](https://arxiv.org/abs/2309.15505);
- **iFSQ** — [Lin et al., 2026](https://arxiv.org/abs/2601.17124), FSQ with a
  distribution-matching bound.

Switching is one word — `bottleneck="fsq"` or `"ifsq"` — and everything else
(the objective, the E-step, the RDFC schedule, the training loop) is unchanged.
See [Choosing a codebook](#choosing-a-codebook).

---

## Contents

- [Install](#install)
- [30-second example](#30-second-example)
- [How DAB works](#how-dab-works)
- [Choosing a codebook](#choosing-a-codebook)
- [The FSQ codebook](#the-fsq-codebook)
- [The iFSQ codebook](#the-ifsq-codebook)
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

The same thing with an FSQ codebook instead — no codebook parameters, plus a
discrete token per example:

```python
from dab import FSQDAB, recommended_levels

layer = FSQDAB(in_features=640, levels=recommended_levels(1024))  # [8, 5, 5, 5]

out = layer(features)
out.latent      # [32, 4]   d is len(levels)
out.distance    # [32]      same uncertainty score
out.indices     # [32]      code index in [0, 1000)
```

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

## Choosing a codebook

Both codebooks implement the same DAB objective, the same E-step, and the same
two-phase schedule. They differ only in *what the codes are*.

|                          | `"full"` / `"diag"` (reference DAB) | `"fsq"` | `"ifsq"` |
| --- | --- | --- | --- |
| codes | `K` learned Gaussians in `R^d` | `∏ⱼ Lⱼ` implicit product codes | same |
| codebook parameters | `K × d` means (+ covariance state) | **none** | **none** |
| distance cost / example | `O(K·d²)` full, `O(K·d)` diag | `O(Σⱼ Lⱼ)` | `O(Σⱼ Lⱼ)` |
| codebook collapse | possible (centroids can die) | impossible | impossible |
| discrete tokens | ✗ | ✓ `out.indices` | ✓ `out.indices` |
| bounding map | — | `tanh`-based `f` | `2σ(αz) − 1` |
| even levels | — | ✓ | ✗ (odd only) |
| far-OOD distance | grows without bound | **saturates** | **saturates** |
| faithful to the DAB paper | ✓ | extension | extension |

**Use the reference codebook when** you want to reproduce the paper, or when
ranking far-out-of-distribution inputs is the point. It is the default.

**Use FSQ when** you want a large codebook cheaply (`[8,5,5,5]` is 1000 codes for
zero parameters), when you need discrete tokens downstream (a transformer over
the latent, retrieval, caching), or when you saw dead centroids with a large `K`.

```python
from dab import build_bottleneck, recommended_levels

build_bottleneck("full", in_features=640, dab_dim=8, codebook_size=10)
build_bottleneck("fsq",  in_features=640, levels=recommended_levels(1024))
build_bottleneck("ifsq", in_features=640, levels=[5, 5, 5, 5])
```

Every model takes the same switch:

```python
from dab.models import wide_resnet_dab

wide_resnet_dab(bottleneck="full", dab_dim=8, codebook_size=10)      # default
wide_resnet_dab(bottleneck="fsq",  levels=[8, 5, 5, 5])
wide_resnet_dab(bottleneck="ifsq", levels=[5, 5, 5, 5])
```

---

## The FSQ codebook

FSQ bounds each latent coordinate with a `tanh`-based function and rounds to
integers, so coordinate `j` takes one of `Lⱼ` values. The codebook is the
Cartesian product of those value sets — `|C| = ∏ⱼ Lⱼ` codewords exist, but none
are stored or learned.

`FSQDAB` keeps that grid and makes each product code a **Gaussian**: coordinate
`j`, level `l` carries `N(gⱼₗ, s²ⱼₗ)` where `gⱼₗ` is the fixed grid point and the
variance is fitted by the same closed-form M-step the reference DAB uses. That
is what keeps it DAB rather than plain FSQ — the distance is still a KL
divergence between distributions, not a Euclidean distance between points.

### Why the per-coordinate computation is exact

The encoder covariance is diagonal and the prior factorises over coordinates, so
for a product code `c = (l₁,…,l_d)` with `π_c = ∏ⱼ πⱼ,ₗⱼ`:

- `KL(encoder ‖ code_c) = Σⱼ KLⱼ(lⱼ)` — the KL is additive;
- the Gibbs assignment `a_c ∝ π_c exp(−τ·KL_c)` therefore **factorises** into
  `a_c = ∏ⱼ aⱼ,ₗⱼ`;
- and the expected distortion `Σ_c a_c KL_c` collapses to `Σⱼ Σₗ aⱼₗ KLⱼ(l)`.

So the coordinate-wise distance is *identical* to the expected KL over all
`∏ⱼ Lⱼ` product codes, at `O(Σⱼ Lⱼ)` cost instead of `O(∏ⱼ Lⱼ)`. This is checked
against brute-force enumeration in `tests/test_fsq_dab.py`, for several level
sets and temperatures.

### Soft and hard modes

```python
FSQDAB(in_features=640, levels=[8, 5, 5, 5], hard=False)   # default
```

- **`hard=False` (soft, default)** — DAB semantics. The assignment is the Gibbs
  posterior over levels, the distortion is the expected KL, and the head sees a
  continuous latent. `out.indices` is still reported (the nearest code), so you
  get tokens without giving up the soft objective.
- **`hard=True`** — FSQ semantics. The mean is rounded with a straight-through
  estimator, the head sees the **quantized codeword**, and the distortion is the
  KL to that single selected code. Use this when the latent must genuinely be
  discrete; `out.latent` is then exactly `fsq.indices_to_codes(out.indices)`.

### Code variances

```python
FSQDAB(..., code_scale_mode="ema")       # default
```

| mode | behaviour | codebook parameters |
| --- | --- | --- |
| `"ema"` | fitted by DAB's closed-form M-step | 0 |
| `"fixed"` | frozen at `code_scale_init` (purest FSQ — nothing is learned) | 0 |
| `"learned"` | a trainable parameter, updated in the RDFC phase | `Σⱼ Lⱼ` |

With `"ema"` or `"fixed"` the codebook has **no trainable parameters at all**, so
`build_optimizers` returns `codebook_optimizer=None`. `rdfc_epoch` accepts that
and still runs both M-step sub-passes — fitting the variances and priors is what
adapts an FSQ codebook to your data.

### Choosing levels

The FSQ paper's heuristic is `Lⱼ ≥ 5` for every coordinate, with `d < 10`.
`recommended_levels(size)` returns the paper's Table 1 entries:

| target `\|C\|` | levels |
| --- | --- |
| 2⁸ = 256 | `[8, 6, 5]` |
| 2¹⁰ = 1024 | `[8, 5, 5, 5]` |
| 2¹² = 4096 | `[7, 5, 5, 5, 5]` |
| 2¹⁴ | `[8, 8, 8, 6, 5]` |
| 2¹⁶ | `[8, 8, 8, 5, 5, 5]` |

Note that `d = len(levels)`, so the level set also fixes the bottleneck
dimension. Ragged sets like `[8, 6, 5]` are fully supported.

### Saturation: read this before using FSQ for OOD

FSQ bounds the encoder mean with a `tanh`. An input far outside the training
manifold lands on the **edge** of the grid and stops moving, so its codebook
distance stops growing: a wildly-OOD input can score the same as a mildly-OOD
one. The encoder *variance* is unbounded and still responds, which is why the
score does not collapse entirely — but the ranking degrades in the far tail.

Mitigations, in order of preference:

1. **Use the reference codebook** (`bottleneck="full"`) if far-OOD ranking is the
   primary goal. This is why it stays the default.
2. **`bound=None`** — drops the `tanh` and puts the grid on an unbounded integer
   lattice, so the distance grows without limit. You keep the DAB uncertainty
   and lose FSQ's guarantee of a finite token set (indices are clamped to the
   grid and are no longer a faithful encoding).
3. **Watch `out.diagnostics["saturated_frac"]`** — the fraction of coordinates
   pinned to an outermost level. If it is high on your in-distribution data the
   grid is too small or the encoder is over-driven, and the score is degraded for
   everything, not just the tail.

### Plain FSQ, without DAB

`dab.FSQ` is a standalone, faithful port of the reference JAX quantizer, usable
in any VQ-VAE-style model:

```python
from dab import FSQ

fsq = FSQ([8, 5, 5, 5])
zhat, indices = fsq(z)            # z: [B, 4] -> zhat [B, 4], indices [B]
fsq.codebook_size                 # 1000
fsq.indices_to_codes(indices)     # == zhat
```

It has no parameters, no commitment loss and no EMA — that is the point of the
paper.

---

## The iFSQ codebook

iFSQ changes exactly one thing about FSQ: the bounding function. Written as a
sigmoid, FSQ's bound is `tanh(z) = 2σ(2z) − 1` — slope `α = 2`. Push a
standard-normal latent through it and the output is *bimodal*, piling up near
the ends of the interval. Sweeping the slope, `α = 1.6` maps a standard normal
to an approximately **uniform** distribution on `[-1, 1]`:

```diff
- z = tanh(z)
+ z = 2 * sigmoid(1.6 * z) - 1
```

Everything downstream — scaling, rounding, the STE, the index arithmetic — is
unchanged, so `IFSQ` is a drop-in for `FSQ` and `IFSQDAB` for `FSQDAB`:

```python
from dab import IFSQ, IFSQDAB

quantizer = IFSQ([5, 5, 5, 5])                       # plain iFSQ, 625 codes
layer = IFSQDAB(in_features=640, levels=[5, 5, 5, 5])   # iFSQ + DAB
```

### Odd levels only

The paper defines `L = 2K + 1` so an exact zero centre exists, and its bound has
no even-`L` offset (FSQ's does). With an even `L` the map would round to `L + 1`
distinct integers, so `IFSQ` rejects even levels with an explanation. Where FSQ
would use `[8, 5, 5, 5]`, use `[5, 5, 5, 5]` (625 codes) or `[9, 5, 5, 5]` (1125).

### Verified: α = 1.6 does improve uniformity, and ≈1.70 is better still

`uniformity_error(alpha)` reproduces the paper's Figure 2(b) sweep — the KS
statistic and RMSE between the bounded value's CDF and the uniform CDF:

| α | KS | RMSE |
| --- | --- | --- |
| 1.0 | 0.119 | 0.083 |
| 1.3 | 0.060 | 0.042 |
| **1.6** (paper) | 0.017 | 0.009 |
| **1.70** | **0.011** | **0.007** |
| 2.0 (`tanh`, FSQ) | 0.046 | 0.031 |
| 2.4 | 0.087 | 0.061 |

The paper's claim holds: 1.6 is ~2.7× closer to uniform than `tanh`. It sweeps a
coarse grid `{1.0, 1.3, 1.6, 2.0, 2.4}`; refining it puts the minimum at
`α ≈ 1.70`, which is the classic logistic approximation to the Gaussian CDF,
`σ(1.702x) ≈ Φ(x)` — exactly what "match the distribution to a uniform"
predicts, since pushing a variable through its own CDF gives a uniform. The
default stays at the paper's `1.6`; pass `alpha=IFSQ_ALPHA_KS_OPTIMAL` for the
refined value.

### Caveat: uniform *value* is not uniform *bin occupancy*

This one is worth understanding before you reach for iFSQ.

Step 2 of the algorithm scales the bounded value by `(L−1)/2` before rounding.
That makes the two **outermost bins half as wide** as the interior ones, so even
a perfectly uniform value on `[-1, 1]` yields the histogram
`[½, 1, 1, …, 1, ½] / (L−1)` — the end bins are under-filled by 2×.

Measuring the marginal level histogram for `z ~ N(0, 1)` (via
`level_histogram`), vanilla FSQ turns out to be *closer* to uniform than iFSQ at
every `L` from 5 to 65, because `tanh`'s bimodality happens to pile mass onto
exactly those half-width end bins. At `L = 5`:

| grid | occupancy | max deviation from ⅕ | realised entropy (max 2.322) |
| --- | --- | --- | --- |
| FSQ (`tanh`) | `.166 .234 .201 .234 .164` | 0.036 | 2.304 |
| iFSQ, paper scaling | `.113 .263 .250 .263 .111` | 0.089 | 2.221 |
| iFSQ, `edge_bins="equal"` | `.194 .207 .200 .207 .192` | **0.008** | **2.321** |

`α` fixes the pre-rounding distribution; it does not fix the grid. Passing
`edge_bins="equal"` scales by `L/2` instead, making every bin the same width and
delivering what the uniformity argument promises — an 11× reduction in
deviation at `L = 5`, and essentially the entropy ceiling. **This is our
observation, not the paper's**, so `"paper"` remains the default:

```python
IFSQ([5, 5, 5, 5], edge_bins="equal")
IFSQDAB(in_features=640, levels=[5, 5, 5, 5], edge_bins="equal")
```

It matters most with `hard=True`, where the level histogram *is* the code
distribution. In soft mode DAB's learned prior absorbs some of the
non-uniformity by itself.

### Index convention

The paper's pseudocode uses a big-endian basis `[L^(d−1), …, L^0]`, so `IFSQ`
defaults to `index_order="big"` while `FSQ` keeps the Google reference's
little-endian `[1, L₀, L₀L₁, …]`. Both are bijections onto `[0, ∏ⱼLⱼ)` — they
just number the codewords differently, so **tokens are not interchangeable
between the two**. Pass `index_order` explicitly if you need a specific
numbering. (`IFSQ([3,3,3,3])` reproduces the paper's worked example: digits
`(2,2,1,0)` → index 75.)

### Does it help for uncertainty?

Unknown, and this repository has not measured it. iFSQ's argument is about the
marginal distribution of a *Gaussian* latent, which is what a reconstruction
autoencoder tends to produce. DAB trains its latent under a rate–distortion
objective with a learned prior, so the premise does not automatically hold.
Treat `IFSQDAB` as an option worth A/B-ing against `FSQDAB`, not as a known
improvement. Setting `ifsq_alpha=2.0, index_order="little", fsq_eps=0.0`
reproduces `FSQDAB` numerically, which makes a controlled comparison easy.

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

### `FSQDAB`

```python
FSQDAB(
    in_features,            # size of the incoming feature vector
    levels=5,               # [L_1, ..., L_d], or an int with dab_dim
    dab_dim=None,           # only needed when levels is an int
    dab_tau=1.0,
    momentum=0.99,
    hard=False,             # False: soft Gibbs assignment; True: FSQ round + STE
    bound="fsq",            # the paper's bounding function, or None for a lattice
    code_scale_mode="ema",  # "ema" | "fixed" | "learned"
    code_scale_init=None,   # default: half the grid spacing of each coordinate
    fsq_eps=1e-3,
    activation="relu", use_bias=False,
    kernel_initializer="glorot_uniform",
    var_shift=5.0, var_floor=1e-5, generator=None,
)
```

`forward(inputs, training=None) -> DABOutput`. Extra attributes:
`codebook_size` (`∏ⱼ Lⱼ`), `num_codebook_parameters`, `levels`, and `fsq` (the
underlying `FSQ` quantizer).

### `FSQ`

```python
FSQ(levels, eps=1e-3)
```

| Member | Purpose |
| --- | --- |
| `bound(z)` | the paper's `f`: `tanh(z + shift) * half_l − offset` |
| `quantize(z)` | `round_ste(bound(z))`, renormalised to `[-1, 1]` |
| `forward(z)` | `(zhat, indices)` |
| `codes_to_indices` / `indices_to_codes` | bijection with `[0, ∏ⱼ Lⱼ)` |
| `per_channel_indices(zhat)` | per-coordinate level index |
| `codebook` | the full codebook, materialised on demand for inspection |
| `codebook_size`, `num_dimensions`, `channel_codes()` | grid properties |

### `build_bottleneck`

```python
build_bottleneck(kind, in_features, dab_dim=None, codebook_size=None,
                 levels=None, **kwargs) -> DABLayer
```

`kind` is `"full"`, `"diag"`, `"fsq"` or `"ifsq"`. Use it when the codebook
should be a config option rather than an import.

### `IFSQ` / `IFSQDAB`

```python
IFSQ(levels, alpha=1.6, eps=0.0, index_order="big", edge_bins="paper")
IFSQDAB(in_features, levels=5, dab_dim=None, ifsq_alpha=1.6,
        fsq_eps=0.0, index_order="big", edge_bins="paper", **FSQDAB_kwargs)
```

`IFSQ` subclasses `FSQ` and overrides only `bound`; `IFSQDAB` subclasses
`FSQDAB` and only swaps the quantizer. Everything else is inherited, so the
contracts above apply unchanged.

| Helper | Purpose |
| --- | --- |
| `ifsq_bound(z, alpha)` | `2σ(αz) − 1`; `α=2` is exactly `tanh` |
| `uniformity_error(alpha)` | KS and RMSE against a uniform — the paper's Fig. 2(b) |
| `level_histogram(quantizer)` | marginal bin occupancy for an `N(0,1)` latent |
| `IFSQ_ALPHA` (1.6), `IFSQ_ALPHA_KS_OPTIMAL` (1.702), `TANH_ALPHA` (2.0) | slopes |

### `DABOutput`

| Field | Shape | Meaning |
| --- | --- | --- |
| `latent` | `[B, d]` | features for the head; a sample when training, the mean at eval |
| `distance` (alias `uncertainty`) | `[B]` | distance from the codebook — **the uncertainty score** |
| `mean` | `[B, d]` | encoder mean `μ(x)` |
| `variance` | `[B, d]` | encoder variances (diagonal of `Σ(x)` for the full-cov layer) |
| `covariance` | `[B, d, d]` | encoder covariance (full-cov layer only) |
| `distances_from_centroids` | `[B, K]` | per-centroid `KL(encoder ‖ centroid)`; `[B, d, L]` for FSQ |
| `assignment` | `[B, K]` | E-step responsibilities, same shape (always detached) |
| `indices` | `[B]` | **FSQ only** — code index in `[0, ∏ⱼ Lⱼ)` |
| `per_coord_indices` | `[B, d]` | **FSQ only** — level index per coordinate |
| `pre_bound_mean` | `[B, d]` | **FSQ only** — the mean before the bounding function |
| `diagnostics` | dict | level occupancy, perplexity, `unused_level_frac`, `saturated_frac` |

`out.as_tensor()` returns `concat([latent, distance[:, None]], -1)` — the exact
tensor the reference Keras layer emits, if you are porting downstream code.

### Codebook state (buffers, saved in `state_dict`)

`centroid_means` (parameter), `centroid_probs`, `centroid_probs_mavg`,
`centroid_covariance`, `centroid_covariance_mavg`, `initialized`, and for the
full-covariance layer `centroid_precision`,
`centroid_covariance_log_abs_det`.

`FSQDAB` uses the same names, shaped `[d, L]` instead of `[K]` / `[K, d]`, with
`centroid_means` a **buffer** (the fixed grid) rather than a parameter, plus
`level_mask` and `grid_spacing`. Because the names match, `codebook_parameters`,
`rdfc_epoch` and the checkpointing story are identical for both codebooks.

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
| `build_bottleneck(kind, ...)` | build any codebook by name |
| `recommended_levels(size)` | the FSQ paper's Table 1 level sets |
| `ifsq_bound`, `uniformity_error`, `level_histogram` | iFSQ's bound and its diagnostics |
| `round_ste(z)` | rounding with straight-through gradients |

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

For the FSQ and iFSQ codebooks, `codebook_size` and `dab_dim` are both replaced
by `levels`: `d = len(levels)` and `|C| = ∏ⱼ Lⱼ`. Start from
`recommended_levels(K)` for whatever `K` you would have used, keep `Lⱼ ≥ 5`, and
tune `beta` and `dab_tau` exactly as above — `dab_tau` now sharpens the
assignment *within each coordinate*, so a value that worked for `K` explicit
codes is a reasonable starting point. iFSQ additionally requires every `Lⱼ` to
be odd.

One more knob worth knowing about, shared by all three codebooks:
`dirichlet_alpha` (default `5.0`, the reference value) smooths the
responsibilities in the covariance M-step. It is strong: with `alpha = 5` and a
batch of `N`, the weight denominator is dominated by `5N` while the numerator
only ranges over `[5, 6]`, so every code's fitted covariance is pulled to within
~20% of a *shared* batch covariance. That is what the reference DAB does and it
stays the default, but lower it (0.1–1.0) if you want codes whose covariances
genuinely specialise.

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

# ... or the same backbone with an FSQ codebook
model = wide_resnet_dab(depth=28, width_multiplier=10, num_classes=10,
                        bottleneck="fsq", levels=[8, 5, 5, 5], dab_tau=1.0)

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

# the same, with an FSQ codebook
python examples/train_cifar10.py --bottleneck fsq --levels 8 5 5 5
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

**FSQ: `saturated_frac` is high, or OOD AUROC is good for near-OOD and poor for
far-OOD.**
The `tanh` bound has saturated — see
[the saturation warning](#saturation-read-this-before-using-fsq-for-ood).
Try `bound=None`, more levels, or the reference codebook.

**iFSQ: `ValueError: iFSQ requires odd levels`.**
By design — see [Odd levels only](#odd-levels-only). Round up (`8 → 9`) or use
`bottleneck="fsq"`.

**iFSQ: bin occupancy still looks uneven.**
Expected with the paper's scaling; see
[the bin-occupancy caveat](#caveat-uniform-value-is-not-uniform-bin-occupancy).
Try `edge_bins="equal"`.

**FSQ: `build_optimizers` returned `codebook_optimizer=None`.**
Expected. The grid is fixed and the variances are fitted in closed form, so
there is nothing to descend on. `rdfc_epoch(model, None, ...)` still runs both
M-step sub-passes. Pass `code_scale_mode="learned"` if you want a trainable
codebook parameter.

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

220 tests.

*DAB core:* `fill_triangular` against the tfp docstring values, both KL
divergences against `torch.distributions`, the encoder parameterisation, the
E-step formula and its stop-gradient, uniform assignment before initialisation,
both closed-form covariance updates against explicit loops, the moving-average
formulas, the RDFC phase ordering (including that the priors are estimated with
the committed covariance), the parameter split, full-covariance sampling against
its empirical covariance, `state_dict` round-trips, and the LR schedule.

*FSQ:* a **numerical parity suite** — `tests/test_fsq_reference_parity.py`
carries a verbatim NumPy transcription of the reference JAX quantizer and
asserts that `bound`, `quantize`, the codebook and the saturating extremes agree
to float32 epsilon, with integer indices identical, across eight level sets
(odd, even, ragged, mixed). Plus: every channel taking exactly `L` values for
extreme inputs, the renormalised grid, the even-`L` asymmetry, `f(0)` landing
mid-grid, index round-trips, codebook enumeration against `itertools.product`,
STE gradients, Table 1, and codebook utilisation above 90%.

*FSQ-DAB:* the factorization identity against brute-force enumeration over the
full product codebook (several level sets × temperatures), assignment
factorisation, ragged-level masking, hard-mode one-hot assignment and codeword
latents, the closed-form variance M-step, RDFC without a codebook optimiser, and
the saturation diagnostic.

*iFSQ:* `2σ(2z)−1 == tanh(z)` exactly, the uniformity sweep and its minimum near
1.70, `α=2` reproducing plain FSQ, odd-level enforcement, the paper's worked
index example, the half-width end-bin effect at four values of `L`, and
`edge_bins="equal"` restoring a uniform histogram (including `L=7`, where
round-half-to-even would otherwise overflow the range).

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

If you use the FSQ codebook, please also cite:

```bibtex
@inproceedings{mentzer2024finite,
  title     = {Finite Scalar Quantization: {VQ-VAE} Made Simple},
  author    = {Mentzer, Fabian and Minnen, David and Agustsson, Eirikur and
               Tschannen, Michael},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}
```

And if you use the iFSQ codebook:

```bibtex
@article{lin2026ifsq,
  title   = {{iFSQ}: Improving {FSQ} for Image Generation with 1 Line of Code},
  author  = {Lin, Bin and Li, Zongjian and Niu, Yuwei and Gong, Kaixiong and
             Ge, Yunyang and Lin, Yunlong and Zheng, Mingzhe and Zhang, JianWei
             and Yang, Miles and Zhong, Zhao and Bo, Liefeng and Yuan, Li},
  journal = {arXiv preprint arXiv:2601.17124},
  year    = {2026}
}
```

Apache License 2.0, matching the original repository. The original TensorFlow
implementation is © 2024 Ifigeneia Apostolopoulou; this port keeps that notice
in every ported file. `dab/fsq.py` is a port of the JAX implementation in
[google-research/fsq](https://github.com/google-research/google-research/tree/master/fsq),
© 2023 The Google Research Authors. The WideResNet backbone follows
[Uncertainty Baselines](https://github.com/google/uncertainty-baselines/).

`FSQDAB` and `IFSQDAB` — DAB's objective over an FSQ or iFSQ grid — are
engineering extensions, not something any of these papers proposes; they are
documented as such in [`docs/FAITHFULNESS.md`](docs/FAITHFULNESS.md) §6–7 and
are never the default. So is `edge_bins="equal"`.
