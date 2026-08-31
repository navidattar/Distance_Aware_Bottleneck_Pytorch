"""Tests for the FSQ quantizer, checked against the paper and reference code."""
import itertools
import math

import pytest
import torch

from dab.fsq import (RECOMMENDED_LEVELS, FSQ, recommended_levels, round_ste)


def test_round_ste_value_and_gradient():
    z = torch.tensor([-1.4, -0.5, 0.2, 1.7], requires_grad=True)
    out = round_ste(z)
    torch.testing.assert_close(out, torch.round(z).detach())
    out.sum().backward()
    torch.testing.assert_close(z.grad, torch.ones(4))   # STE: gradient is 1


@pytest.mark.parametrize("levels", [[3], [4], [5], [8], [8, 6, 5], [8, 5, 5, 5]])
def test_each_channel_takes_exactly_L_values(levels):
    """f then round must produce exactly L distinct values per channel."""
    q = FSQ(levels)
    z = torch.linspace(-30, 30, 20001).reshape(-1, 1).repeat(1, len(levels))
    zhat, _ = q(z)
    for j, L in enumerate(levels):
        assert len(torch.unique(zhat[:, j])) == L


@pytest.mark.parametrize("levels", [[3], [5], [8, 6, 5]])
def test_quantized_values_are_the_renormalised_grid(levels):
    """Values are (i - L//2) / (L//2), matching the reference `quantize`."""
    q = FSQ(levels)
    z = torch.linspace(-30, 30, 20001).reshape(-1, 1).repeat(1, len(levels))
    zhat, _ = q(z)
    for j, L in enumerate(levels):
        expected = (torch.arange(L, dtype=torch.float32) - (L // 2)) / (L // 2)
        torch.testing.assert_close(torch.unique(zhat[:, j]), expected)


def test_odd_levels_are_symmetric_and_include_zero():
    torch.testing.assert_close(FSQ([5]).channel_codes()[0],
                               torch.tensor([-1., -0.5, 0., 0.5, 1.]))


def test_even_levels_are_asymmetric():
    """An even number of integers cannot be centred on zero; FSQ shifts instead."""
    torch.testing.assert_close(FSQ([4]).channel_codes()[0],
                               torch.tensor([-1., -0.5, 0., 0.5]))


def test_zero_maps_to_the_middle_of_the_grid():
    """The `shift = tan(offset / half_l)` term keeps f(0) centred."""
    for L in (3, 4, 5, 8):
        q = FSQ([L])
        zhat, _ = q(torch.zeros(1, 1))
        assert abs(float(zhat)) < 1e-6


def test_codebook_size_is_the_product_of_levels():
    assert FSQ([8, 6, 5]).codebook_size == 240
    assert FSQ([8, 5, 5, 5]).codebook_size == 1000
    assert FSQ([7, 5, 5, 5, 5]).codebook_size == 4375


@pytest.mark.parametrize("levels", [[3, 3], [4, 3], [5, 4, 3]])
def test_codebook_enumerates_the_full_cartesian_product(levels):
    q = FSQ(levels)
    cb = q.codebook
    assert cb.shape == (q.codebook_size, len(levels))
    expected = {tuple(round(v, 6) for v in combo)
                for combo in itertools.product(*[c.tolist()
                                                 for c in q.channel_codes()])}
    got = {tuple(round(v, 6) for v in row) for row in cb.tolist()}
    assert got == expected
    assert len(got) == q.codebook_size          # every codeword is distinct


@pytest.mark.parametrize("levels", [[3], [4, 3], [8, 6, 5]])
def test_index_round_trip(levels):
    q = FSQ(levels)
    idx = torch.arange(q.codebook_size)
    torch.testing.assert_close(q.codes_to_indices(q.indices_to_codes(idx)), idx)


def test_indices_are_in_range_and_match_quantized_codes():
    q = FSQ([8, 6, 5])
    z = torch.randn(500, 3) * 3
    zhat, idx = q(z)
    assert int(idx.min()) >= 0 and int(idx.max()) < q.codebook_size
    torch.testing.assert_close(q.indices_to_codes(idx), zhat)


def test_gradients_flow_through_quantize():
    q = FSQ([5, 5])
    z = torch.randn(6, 2, requires_grad=True)
    q.quantize(z).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert (z.grad.abs() > 0).any()


def test_bound_stays_inside_the_rounding_range_for_extreme_inputs():
    for L in (3, 4, 5, 8):
        q = FSQ([L])
        z = torch.tensor([[-1e6], [1e6]])
        levels = q._scale_and_shift(q.quantize(z)).round().long()
        assert int(levels.min()) == 0 and int(levels.max()) == L - 1


def test_recommended_levels_matches_table_1_of_the_paper():
    assert recommended_levels(2 ** 8) == [8, 6, 5]
    assert recommended_levels(2 ** 10) == [8, 5, 5, 5]
    assert recommended_levels(2 ** 12) == [7, 5, 5, 5, 5]
    assert recommended_levels(2 ** 14) == [8, 8, 8, 6, 5]
    assert recommended_levels(2 ** 16) == [8, 8, 8, 5, 5, 5]
    # every tabulated set honours the paper's L_i >= 5 heuristic
    for levels in RECOMMENDED_LEVELS.values():
        assert all(L >= 5 for L in levels)
    # untabulated targets fall back to the nearest tabulated set
    assert recommended_levels(1000) == [8, 5, 5, 5]


def test_rejects_bad_levels_and_wrong_input_width():
    with pytest.raises(ValueError):
        FSQ([1, 5])
    with pytest.raises(ValueError):
        FSQ([5, 5]).bound(torch.zeros(2, 3))


def test_codebook_utilisation_is_high_for_gaussian_inputs():
    """FSQ's headline property: no codebook collapse, by construction."""
    torch.manual_seed(0)
    q = FSQ([5, 5, 5])
    _, idx = q(torch.randn(20000, 3))
    used = len(torch.unique(idx))
    assert used / q.codebook_size > 0.9
