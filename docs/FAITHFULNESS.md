# Faithfulness to the reference implementation

This document maps every part of the PyTorch port to the reference TensorFlow
code at [`ifiaposto/Distance_Aware_Bottleneck`](https://github.com/ifiaposto/Distance_Aware_Bottleneck),
so the two can be audited side by side. It also lists, explicitly, the places
where the port deviates and why.

## 1. File-level correspondence

| Reference (TensorFlow) | This repository |
| --- | --- |
| `layers/dab_normal.py` (`_NormalDAB`) | `dab/layers.py` (`_NormalDAB`) |
| `layers/dab_normal_diag_covariance.py` | `dab/layers.py` (`NormalDiagCovarianceDAB`) |
| `layers/dab_normal_full_covariance.py` | `dab/layers.py` (`NormalFullCovarianceDAB`) |
| `tfp.math.fill_triangular`, `tfp.distributions.*.kl_divergence` | `dab/functional.py` |
| `run_cifar.py` / `run_imagenet.py` → `rdfc_step` | `dab/trainer.py` (`rdfc_epoch`) |
| `run_cifar.py` / `run_imagenet.py` → `train_step` loss | `dab/objectives.py` (`dab_loss`, `codebook_distortion`) |
| `utils/schedules.py` | `dab/trainer.py` (`WarmUpPiecewiseConstantSchedule`) |
| `models/wide_resnet_dab.py` | `dab/models/wide_resnet_dab.py` |
| `models/pretrained_resnet50_dab.py` | `dab/models/resnet50_dab.py` |
| `run_cifar.py` (full script) | `examples/train_cifar10.py` |
| `synthetic_regression_demo.py` | `examples/synthetic_regression_demo.py` |

## 2. State-level correspondence

| Reference variable | Port | Trainable | Notes |
| --- | --- | --- | --- |
| `centroid_means` | `centroid_means` (`nn.Parameter`) | yes | `N(0, 0.1²)` init; the only codebook tensor touched by gradient descent |
| `centroid_probs` | buffer `centroid_probs` | no | committed prior used by this epoch's E-step |
| `centroid_probs_mavg` | buffer `centroid_probs_mavg` | no | moving average accumulated for the next epoch |
| `initialized` | buffer `initialized` (bool) | no | while `False`, assignments are uniform |
| `centroid_covariance` (diag) | buffer `centroid_covariance` | no | stores **variances**; init `1` |
| `centroid_covariance_mavg` | buffer `centroid_covariance_mavg` | no | init `0`, reset each RDFC phase |
| `centroid_precision` (full) | buffer `centroid_precision` | no | init `I` |
| `centroid_covariance_log_abs_det` (full) | buffer `centroid_covariance_log_abs_det` | no | init `0`; shape `[K]` instead of `[K, 1]` |

The name `centroid` is load-bearing: `dab.trainer.codebook_parameters` selects
parameters by substring exactly as the reference code does with
`if "centroid" in var.name`.

## 3. Equation-level correspondence

* **Encoder.** `params = Dense(activation(inputs))`, no bias by default.
  Diagonal: `mu, raw = split(params, [d, d])`, `scale = softplus(raw - 5) + 1e-5`.
  Full: `mu, raw = split(params, [d, d(d+1)/2])`, then
  `M = 0.5 * (fill_triangular(raw, upper=True) + fill_triangular(raw))`,
  `U, s, _ = svd(M)`, `s = softplus(s - 5) + 1e-5`, `Σ = U diag(s²) Uᵀ`.
  `dab.functional.fill_triangular` is a bit-exact port of `tfp.math.fill_triangular`,
  including its element ordering, and is tested against the values in the tfp
  docstring.
* **Distance.** `KL(encoder ‖ centroid)` for every centroid, in that order
  (the encoder is the first argument).
* **E-step.** `a = softmax(log π − τ · KL)`, always detached, with a uniform
  fallback while `initialized == False`.
* **Codebook distance.** `d(x) = Σ_k a_k · KL_k`, returned as the uncertainty.
* **M-step, priors.** `π ← relu(mean_i a_ik)` over the global batch, folded into
  the moving average as `mavg ← momentum · mavg + (1 − momentum) · π`.
* **M-step, covariances.** Davis & Dhillon (NeurIPS 2006) eq. (9) with a
  Dirichlet(5) smoothing of the responsibilities:
  `w_ik = (a_ik + 5) / (Σ_i a_ik + 5N)` and
  `Σ_k = Σ_i w_ik (Σ_i^enc + (μ_i − m_k)(μ_i − m_k)ᵀ)`, off-diagonals dropped for
  the diagonal variant. Folded into the moving average the same way.
* **Output.** Training: a sample from the encoder Gaussian. Evaluation: the mean.

Every one of these is covered by a unit test in `tests/`, checked either against
`torch.distributions` or against an explicit per-element loop.

## 4. Deliberate deviations

These are the only behavioural differences from the reference code. None of them
changes the DAB objective.

1. **Full-covariance distance uses the cached precision matrices.** The
   reference full-covariance layer precomputes `centroid_precision` and
   `centroid_covariance_log_abs_det` at the end of every RDFC phase and uses
   them in `compute_codebook_distance`; the port does the same
   (`dab.functional.kl_normal_full`). This is a batched `O(B K d²)` matmul
   rather than `B · K` factorisations, and matches the reference arithmetic to
   floating-point tolerance.
2. **Full-covariance sampling uses `U diag(s) Uᵀ` as the scale matrix**, where
   the reference builds a `MultivariateNormalFullCovariance` whose scale is the
   Cholesky factor of `Σ`. Both are valid square roots of the same covariance,
   so the sampling distribution is identical; the symmetric square root is
   already available from the encoder's SVD, so it avoids an extra
   factorisation. Verified empirically in
   `tests/test_layers.py::test_full_covariance_sampling_uses_the_full_covariance`.
3. **L2 regularisation is applied as optimiser `weight_decay`** rather than as
   Keras kernel/bias regularisers added to the loss. For SGD without adaptive
   scaling these coincide up to the usual factor-of-two convention on the
   penalty; `build_optimizers` passes `l2` straight through as `weight_decay`.
   If you need the exact `l2 · ‖w‖²` term in the loss, add it explicitly.
4. **`codebook_jitter` / `covariance_floor` hooks.** Both default to `0.0`,
   which reproduces the reference numerics exactly. They exist because the
   codebook covariance moving average starts at zero: with `momentum = m` it
   needs on the order of `1/(1 − m)` steps per RDFC phase before it is well
   conditioned. If you run very short epochs, either lower `momentum` or set a
   small jitter. `set_codebook_covariance` raises a descriptive error rather
   than silently producing `inf` if the covariance is singular.
5. **`log|Σ|` for the encoder is taken as `2 Σ_j log s_j`** from the SVD scale,
   instead of via the Cholesky factor's log-determinant. Algebraically identical
   and cheaper.
6. **Distributed reductions use `torch.distributed`** in place of TensorFlow's
   `replica_ctx.all_reduce` / `all_gather`. Semantics match, including the
   global-batch normalisation of the priors and the Dirichlet smoothing. Like
   the reference, this requires an equal per-rank batch size (use `drop_last`).
7. **Layer initialisation.** The `_NormalDAB` default is `glorot_uniform`
   (Keras's `Dense` default). The WideResNet model passes `he_normal`, matching
   the reference model. Backbone convolutions use He-normal init; exact
   per-tensor RNG streams naturally differ from TensorFlow's stateless seeds.

## 5. Known quirks reproduced on purpose

* `0.5 * (fill_triangular(raw, upper=True) + fill_triangular(raw))` is **not**
  symmetric, despite the comment in the reference source, because tfp's two
  `fill_triangular` orderings are not transposes of one another. It does not
  matter — the matrix is only consumed by an SVD, and `U diag(s²) Uᵀ` is SPD for
  any input — and the construction is reproduced exactly so that ported models
  behave identically. See `symmetrized_fill_triangular`.
* The covariance moving average is reset to **zero** (not to the previous value)
  at the start of every RDFC phase, so the committed covariance is
  `(1 − momentumᴺ)` times the true weighted average after `N` steps. This is the
  reference behaviour.
* A centroid whose prior collapses to exactly zero stays dead: `log 0 = −∞`
  drives its responsibility to zero, and `relu(mean(0)) = 0` keeps it there.
  The reference implementation has the same property.
* The moving averages are updated on **every** forward pass with
  `training=True`, including during the encoder/decoder phase. Those updates are
  discarded, because the next RDFC phase resets both averages before
  accumulating. Reproduced as-is.

## 6. The FSQ codebook is an extension, not part of the reference

`dab/fsq.py` and `dab/fsq_dab.py` are **not** ports of anything in the DAB
repository — the reference implementation has only the two explicit-centroid
layers documented above. They are provided as an optional alternative codebook
and are opt-in everywhere (`bottleneck="full"` remains the default).

* `dab/fsq.py` **is** a faithful port, but of a different reference: the JAX
  implementation in
  [`google-research/fsq`](https://github.com/google-research/google-research/tree/master/fsq),
  accompanying Mentzer et al., *Finite Scalar Quantization: VQ-VAE Made Simple*
  (ICLR 2024, [arXiv:2309.15505](https://arxiv.org/abs/2309.15505)). `bound`,
  `quantize`, `codes_to_indices`, `indices_to_codes` and `round_ste` follow it
  line for line, including the even-`L` offset/shift asymmetry and the
  `/(L//2)` renormalisation. `RECOMMENDED_LEVELS` is Table 1 of the paper, and
  the tests assert the per-channel value sets it implies.

* `dab/fsq_dab.py` combines the two: DAB's rate–distortion objective, E-step and
  closed-form M-step, over FSQ's product grid instead of explicit centroids.
  The combination is not published; treat it as an engineering option, not a
  reproduction. What *is* provable — and tested against brute-force enumeration
  in `tests/test_fsq_dab.py` — is that the coordinate-wise distance it computes
  equals the expected KL over all `prod_j L_j` product codes exactly, because
  the encoder covariance is diagonal and the prior factorises.

The DAB machinery is shared rather than duplicated: `FSQDAB` derives from the
same `DABLayer` base as `_NormalDAB`, so `rdfc_epoch`, `codebook_parameters`,
`find_dab_layers` and the phase protocol are literally the same code for both
codebooks.

## 7. iFSQ

`dab/ifsq.py` implements Lin et al., *iFSQ: Improving FSQ for Image Generation
with 1 Line of Code* ([arXiv:2601.17124](https://arxiv.org/abs/2601.17124)).
There is no reference implementation vendored here, so the port follows the
paper's Algorithm 1 and §3.1–3.2 directly:

* the bound is `2·sigmoid(alpha·z) − 1` with `alpha = 1.6` (§3.2 and the
  pseudocode's one-line diff);
* the grid scale is `(L−1)/2` and the renormalisation is `/(L−1)/2`
  (Algorithm 1, steps 2 and 4);
* rounding uses the same STE (step 3);
* the index basis is big-endian, `[L^(d−1), …, L^0]` (step 5). The paper's
  worked example — digits `(2,2,1,0)` with `L=3` giving index 75 — is asserted
  in `tests/test_ifsq.py`;
* levels are restricted to odd `L = 2K+1`, which §3.1 requires for an exact zero
  centre and which the bound's lack of an even-`L` offset makes mandatory.

`IFSQ` subclasses `FSQ` and overrides only `bound`; `IFSQDAB` subclasses
`FSQDAB` and only swaps the quantizer. With `alpha=2.0`, `eps=0.0` and
`index_order="little"` both reduce exactly to their FSQ counterparts, which is
tested.

### Two findings recorded during the port

Both are checkable in the test suite; neither changes the default behaviour.

1. **The optimal slope is ≈1.70, not 1.6.** The paper sweeps
   `alpha ∈ {1.0, 1.3, 1.6, 2.0, 2.4}` and reports 1.6 as the best. On a finer
   sweep the Kolmogorov-Smirnov minimum sits at `alpha ≈ 1.70` (KS 0.011 vs
   0.017 at 1.6 and 0.046 at `tanh`), which is the classic logistic
   approximation to the Gaussian CDF, `sigma(1.702x) ≈ Phi(x)`. This *supports*
   the paper's distribution-matching thesis and refines its constant.
   `IFSQ_ALPHA_KS_OPTIMAL` exposes it; the default remains the paper's 1.6.

2. **A uniform pre-rounding value does not give a uniform level histogram.**
   Scaling by `(L−1)/2` makes the two outermost bins half as wide as the
   interior ones, so a perfectly uniform value on `[-1, 1]` produces
   `[½, 1, …, 1, ½] / (L−1)`. Measured on an `N(0,1)` latent, vanilla FSQ is
   therefore *closer* to a uniform histogram than iFSQ at every `L` from 5 to
   65 — `tanh`'s bimodality happens to compensate for the narrow end bins.
   The opt-in `edge_bins="equal"` scales by `L/2` instead, equalising the bin
   widths; at `L=5` it cuts the maximum deviation from 0.089 to 0.008 and
   reaches 2.321 of the 2.322-bit ceiling. This is our observation, not the
   paper's, so `"paper"` is the default.

## 8. FSQ parity is machine-checked

`tests/test_fsq_reference_parity.py` carries a verbatim NumPy transcription of
the quantizer in `google-research/fsq/fsq.ipynb` and asserts, across eight level
sets covering odd, even, ragged and mixed grids:

| Quantity | Tolerance | Result |
| --- | --- | --- |
| `bound(z)` | 1e-5 | max Δ 3.3e-7 |
| `quantize(z)` | 1e-6 | max Δ 2.0e-8 |
| `codes_to_indices` | exact integers | identical |
| the full codebook | 1e-6 | max Δ 2.0e-8 |
| saturating inputs (±1e6) | 1e-6 | identical bins |

So the FSQ core is not merely "written from the paper" — it reproduces the
reference implementation's arithmetic. Anything that drifts in either direction
breaks these tests.
