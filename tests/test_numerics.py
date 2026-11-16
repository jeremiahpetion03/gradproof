"""Numerical-stability tests: the failure modes that turn a loss into NaN."""

import numpy as np

from gradproof.nn import CausalSelfAttention, cross_entropy, cross_entropy_naive, softmax, softmax_naive


def test_naive_softmax_overflows_on_large_logits():
    x = np.array([[1000.0, 999.0, 998.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        assert np.isnan(softmax_naive(x)).any()
    p = softmax(x)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(), 1.0)
    np.testing.assert_allclose(p, softmax(x - 1000.0))  # shift invariance


def test_float32_overflow_starts_much_earlier():
    """exp overflows float32 above ~88.7, so even modest logits break the naive version."""
    x = np.array([[100.0, 0.0]], dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        assert not np.isfinite(softmax_naive(x)).all()
    assert np.isfinite(softmax(x)).all()


def test_stable_cross_entropy_matches_naive_where_naive_works():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((4, 5, 11))
    targets = rng.integers(0, 11, (4, 5))
    stable, _ = cross_entropy(logits, targets)
    np.testing.assert_allclose(stable, cross_entropy_naive(logits, targets), rtol=1e-12)


def test_stable_cross_entropy_survives_where_naive_fails():
    logits = np.zeros((1, 1, 3))
    logits[0, 0] = [2000.0, 0.0, -2000.0]
    targets = np.array([[1]])
    assert not np.isfinite(cross_entropy_naive(logits, targets))
    loss, grad = cross_entropy(logits, targets)
    np.testing.assert_allclose(loss, 2000.0)  # -log softmax = 2000 - 0 exactly
    assert np.isfinite(grad).all()


def test_cross_entropy_gradient_rows_sum_to_zero():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((2, 3, 7))
    targets = rng.integers(0, 7, (2, 3))
    _, grad = cross_entropy(logits, targets)
    np.testing.assert_allclose(grad.sum(axis=-1), 0.0, atol=1e-15)


def test_masked_positions_get_no_gradient():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((2, 4, 5))
    targets = rng.integers(0, 5, (2, 4))
    mask = np.array([[True, False, True, False], [False, False, True, True]])
    _, grad = cross_entropy(logits, targets, mask)
    assert (grad[~mask] == 0).all()


def test_attention_finite_with_huge_scores():
    rng = np.random.default_rng(3)
    attn = CausalSelfAttention(8, 2, rng, np.float32)
    attn.qkv.params["W"] *= 500.0  # scores in the 1e4+ range
    x = rng.standard_normal((2, 6, 8)).astype(np.float32)
    y = attn.forward(x)
    assert np.isfinite(y).all()
    dx = attn.backward(np.ones_like(y))
    assert np.isfinite(dx).all()
