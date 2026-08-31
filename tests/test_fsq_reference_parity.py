"""Numerical parity between our FSQ and the reference JAX implementation.

``RefFSQ`` below is a verbatim transcription of the quantizer in
``google-research/fsq/fsq.ipynb`` with ``jnp`` swapped for ``numpy``. Keeping it
here means the port is checked against the reference *arithmetic*, not against
our reading of it: if either drifts, these tests fail.
"""
import numpy as np
import pytest
import torch

from dab.fsq import FSQ

LEVEL_SETS = [[3], [4], [5], [8], [8, 6, 5], [8, 5, 5, 5], [7, 5, 5, 5, 5],
              [4, 3, 6, 2]]


class RefFSQ:
    """google-research/fsq quantizer, transcribed from JAX to NumPy."""

    def __init__(self, levels, eps=1e-3):
        self._levels = levels
        self._eps = eps
        self._levels_np = np.asarray(levels)
        self._basis = np.concatenate(
            ([1], np.cumprod(self._levels_np[:-1]))).astype(np.uint32)

    def bound(self, z):
        half_l = (self._levels_np - 1) * (1 - self._eps) / 2
        offset = np.where(self._levels_np % 2 == 1, 0.0, 0.5)
        shift = np.tan(offset / half_l)
        return np.tanh(z + shift) * half_l - offset

    def quantize(self, z):
        quantized = np.round(self.bound(z))
        half_width = self._levels_np // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels_np // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels_np // 2
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat):
        return (self._scale_and_shift(zhat)
                * self._basis).sum(axis=-1).astype(np.uint32)

    def indexes_to_codes(self, indices):
        indices = indices[..., np.newaxis]
        codes_non_centered = np.mod(
            np.floor_divide(indices, self._basis), self._levels_np)
        return self._scale_and_shift_inverse(codes_non_centered)


def _inputs(levels, n=4000, seed=0):
    z = np.random.default_rng(seed).normal(0, 3, size=(n, len(levels)))
    return z.astype(np.float32)


@pytest.mark.parametrize("levels", LEVEL_SETS)
def test_bound_matches_reference(levels):
    z = _inputs(levels)
    got = FSQ(levels).bound(torch.from_numpy(z)).numpy()
    np.testing.assert_allclose(got, RefFSQ(levels).bound(z), rtol=0, atol=1e-5)


@pytest.mark.parametrize("levels", LEVEL_SETS)
def test_quantize_matches_reference(levels):
    z = _inputs(levels)
    got = FSQ(levels).quantize(torch.from_numpy(z)).numpy()
    np.testing.assert_allclose(got, RefFSQ(levels).quantize(z),
                               rtol=0, atol=1e-6)


@pytest.mark.parametrize("levels", LEVEL_SETS)
def test_indices_match_reference_exactly(levels):
    """Integer tokens must be identical, not merely close."""
    z = _inputs(levels)
    ref = RefFSQ(levels)
    zhat = ref.quantize(z)
    got = FSQ(levels).codes_to_indices(torch.from_numpy(zhat)).numpy()
    np.testing.assert_array_equal(got, ref.codes_to_indexes(zhat).astype(np.int64))


@pytest.mark.parametrize("levels", LEVEL_SETS)
def test_codebook_matches_reference(levels):
    n = int(np.prod(levels))
    ref = RefFSQ(levels).indexes_to_codes(np.arange(n, dtype=np.uint32))
    got = FSQ(levels).codebook.numpy()
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6)


@pytest.mark.parametrize("levels", LEVEL_SETS)
def test_extremes_match_reference(levels):
    """Saturating inputs must land on the same end bins."""
    z = np.array([[-1e6] * len(levels), [1e6] * len(levels)], dtype=np.float32)
    np.testing.assert_allclose(FSQ(levels).quantize(torch.from_numpy(z)).numpy(),
                               RefFSQ(levels).quantize(z), rtol=0, atol=1e-6)
