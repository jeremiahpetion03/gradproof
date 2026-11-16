import numpy as np
import pytest

from conftest import SMALL, TASK
from gradproof.model import GPT, activation_bytes


def test_causal_mask_blocks_the_future():
    """Changing token t must not change any logit at positions before t."""
    model = GPT(SMALL, seed=0, dtype=np.float64)
    rng = np.random.default_rng(0)
    x = rng.integers(0, SMALL.vocab_size, (1, SMALL.block_size))
    base = model.forward(x).copy()
    for t in range(1, SMALL.block_size):
        x2 = x.copy()
        x2[0, t] = (x2[0, t] + 1) % SMALL.vocab_size
        out = model.forward(x2)
        np.testing.assert_array_equal(out[0, :t], base[0, :t])
        assert not np.array_equal(out[0, t], base[0, t])


def test_float32_stays_float32_end_to_end():
    """Guards against NumPy 2 promotion silently upcasting to float64,
    which would double memory and halve throughput without any error."""
    model = GPT(SMALL, seed=0, dtype=np.float32)
    x, y, mask = TASK.sample(np.random.default_rng(0), 4)
    from gradproof.nn import cross_entropy

    logits = model.forward(x)
    assert logits.dtype == np.float32
    _, d = cross_entropy(logits, y, mask)
    assert d.dtype == np.float32
    model.backward(d)
    for name, mod, k in model.named_parameters():
        assert mod.params[k].dtype == np.float32, name
        assert mod.grads[k].dtype == np.float32, name


def test_rejects_sequence_longer_than_block_size():
    model = GPT(SMALL, seed=0)
    with pytest.raises(ValueError):
        model.forward(np.zeros((1, SMALL.block_size + 1), dtype=int))


def test_state_dict_round_trip():
    a, b = GPT(SMALL, seed=0), GPT(SMALL, seed=1)
    x = np.random.default_rng(0).integers(0, SMALL.vocab_size, (2, 5))
    assert not np.array_equal(a.forward(x), b.forward(x))
    b.load_state_dict(a.state_dict())
    np.testing.assert_array_equal(a.forward(x), b.forward(x))


def test_generate_shape():
    model = GPT(SMALL, seed=0)
    prompt = np.zeros((3, 5), dtype=int)
    assert model.generate(prompt, 3).shape == (3, 3)


def test_float32_halves_activation_memory():
    x = np.random.default_rng(0).integers(0, SMALL.vocab_size, (8, SMALL.block_size))
    m32, m64 = GPT(SMALL, dtype=np.float32), GPT(SMALL, dtype=np.float64)
    m32.forward(x)
    m64.forward(x)
    # index arrays are int64 in both, so the ratio is close to (not exactly) 2
    assert 1.9 < activation_bytes(m64) / activation_bytes(m32) <= 2.0
