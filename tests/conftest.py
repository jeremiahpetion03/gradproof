import numpy as np
import pytest

from gradproof.data import VOCAB, Task
from gradproof.model import GPT, GPTConfig
from gradproof.nn import cross_entropy

TASK = Task("sort", 4)
SMALL = GPTConfig(vocab_size=VOCAB, block_size=TASK.block_size, n_layer=2, n_head=2, d_model=16)


def make_checkable_model(seed: int = 0, scale: float = 4.0) -> GPT:
    """float64 model with weights scaled up from the 0.02 init so attention is
    far from uniform and most gradients are well above rounding noise."""
    model = GPT(SMALL, seed=seed, dtype=np.float64)
    for _, mod, k in model.named_parameters():
        if k != "g":  # leave LayerNorm gains at 1
            mod.params[k][...] *= scale
    return model


def make_loss_fn(model: GPT, batch: int = 3, seed: int = 1):
    x, y, mask = TASK.sample(np.random.default_rng(seed), batch)

    def loss_fn(backward: bool) -> float:
        loss, dlogits = cross_entropy(model.forward(x), y, mask)
        if backward:
            model.backward(dlogits)
        return loss

    return loss_fn


@pytest.fixture
def checkable():
    model = make_checkable_model()
    return model, make_loss_fn(model)
