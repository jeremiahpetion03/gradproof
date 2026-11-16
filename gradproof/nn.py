"""Neural-network layers with hand-written forward and backward passes.

Every module caches what it needs during ``forward`` and accumulates parameter
gradients into ``self.grads`` during ``backward``. ``backward`` returns the
gradient with respect to the module's input.

Constants are plain Python floats on purpose: under NumPy 2 promotion rules a
``np.float64`` scalar would silently upcast float32 activations to float64.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np


class Module:
    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def children(self) -> Iterator[tuple[str, "Module"]]:
        for name, value in vars(self).items():
            if isinstance(value, Module):
                yield name, value
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        yield f"{name}.{i}", item

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, "Module", str]]:
        """Yield (full_name, owning_module, key) in a fixed, deterministic order."""
        for key in self.params:
            yield prefix + key, self, key
        for name, child in self.children():
            yield from child.named_parameters(prefix + name + ".")

    def zero_grad(self) -> None:
        for _, mod, key in self.named_parameters():
            mod.grads[key] = np.zeros_like(mod.params[key])

    def num_params(self) -> int:
        return sum(mod.params[k].size for _, mod, k in self.named_parameters())


# --------------------------------------------------------------------------- #
# Softmax and loss
# --------------------------------------------------------------------------- #


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax: subtract the row max before exponentiating."""
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def softmax_naive(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Textbook softmax WITHOUT max-subtraction. Kept only to demonstrate
    (and test for) the overflow bug it causes on large logits."""
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy(
    logits: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None
) -> tuple[float, np.ndarray]:
    """Mean negative log-likelihood over positions where ``mask`` is true.

    Uses log-sum-exp so it stays finite for arbitrarily large logits.
    Returns (loss, dloss/dlogits).
    """
    if mask is None:
        mask = np.ones(targets.shape, dtype=bool)
    w = mask.astype(logits.dtype)
    n = w.sum()
    if n == 0:
        raise ValueError("cross_entropy: mask selects no positions")

    z = logits - logits.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    t = targets[..., None]
    nll = -np.take_along_axis(logp, t, axis=-1)[..., 0]
    loss = float((nll * w).sum() / n)

    dlogits = np.exp(logp)
    np.put_along_axis(dlogits, t, np.take_along_axis(dlogits, t, axis=-1) - 1.0, axis=-1)
    dlogits *= (w / n)[..., None]
    return loss, dlogits


def cross_entropy_naive(logits: np.ndarray, targets: np.ndarray) -> float:
    """Cross-entropy via naive softmax then log. Overflows to NaN/inf on large
    logits. Exists only so tests can prove the stable version is needed."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        p = softmax_naive(logits)
        picked = np.take_along_axis(p, targets[..., None], axis=-1)[..., 0]
        return float(-np.log(picked).mean())


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #


class Linear(Module):
    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, dtype, std: float = 0.02):
        super().__init__()
        self.params["W"] = (rng.standard_normal((n_in, n_out)) * std).astype(dtype)
        self.params["b"] = np.zeros(n_out, dtype=dtype)
        self.zero_grad()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dy: np.ndarray) -> np.ndarray:
        n_in, n_out = self.params["W"].shape
        self.grads["W"] += self.x.reshape(-1, n_in).T @ dy.reshape(-1, n_out)
        self.grads["b"] += dy.reshape(-1, n_out).sum(axis=0)
        return dy @ self.params["W"].T


class GELU(Module):
    """tanh-approximation GELU (the variant used by GPT-2)."""

    C = math.sqrt(2.0 / math.pi)
    A = 0.044715

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        # x*x*x, not x**3: NumPy fast-paths x**2 but routes x**3 through the
        # generic pow loop, which profiled at ~100x slower and made this one
        # line 62% of the whole training step (see README, "Profiling").
        self.t = np.tanh(self.C * (x + self.A * (x * x * x)))
        return 0.5 * x * (1.0 + self.t)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        x, t = self.x, self.t
        dinner = self.C * (1.0 + 3.0 * self.A * x * x)
        return dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner)


class LayerNorm(Module):
    def __init__(self, d: int, dtype, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.params["g"] = np.ones(d, dtype=dtype)
        self.params["b"] = np.zeros(d, dtype=dtype)
        self.zero_grad()

    def forward(self, x: np.ndarray) -> np.ndarray:
        xc = x - x.mean(axis=-1, keepdims=True)
        var = (xc * xc).mean(axis=-1, keepdims=True)
        self.rstd = 1.0 / np.sqrt(var + self.eps)
        self.xhat = xc * self.rstd
        return self.xhat * self.params["g"] + self.params["b"]

    def backward(self, dy: np.ndarray) -> np.ndarray:
        d = dy.shape[-1]
        self.grads["g"] += (dy * self.xhat).reshape(-1, d).sum(axis=0)
        self.grads["b"] += dy.reshape(-1, d).sum(axis=0)
        dxhat = dy * self.params["g"]
        return self.rstd * (
            dxhat
            - dxhat.mean(axis=-1, keepdims=True)
            - self.xhat * (dxhat * self.xhat).mean(axis=-1, keepdims=True)
        )


class Embedding(Module):
    def __init__(self, vocab: int, d: int, rng: np.random.Generator, dtype, std: float = 0.02):
        super().__init__()
        self.params["W"] = (rng.standard_normal((vocab, d)) * std).astype(dtype)
        self.zero_grad()

    def forward(self, idx: np.ndarray) -> np.ndarray:
        self.idx = idx
        return self.params["W"][idx]

    def backward(self, dy: np.ndarray) -> None:
        d = dy.shape[-1]
        # add.at, not +=: repeated token ids must accumulate, not overwrite.
        np.add.at(self.grads["W"], self.idx.reshape(-1), dy.reshape(-1, d))


class PositionalEmbedding(Module):
    def __init__(self, block_size: int, d: int, rng: np.random.Generator, dtype, std: float = 0.02):
        super().__init__()
        self.params["P"] = (rng.standard_normal((block_size, d)) * std).astype(dtype)
        self.zero_grad()

    def forward(self, T: int) -> np.ndarray:
        self.T = T
        return self.params["P"][:T]

    def backward(self, dy: np.ndarray) -> None:
        self.grads["P"][: self.T] += dy.sum(axis=0)


class CausalSelfAttention(Module):
    def __init__(self, d: int, n_head: int, rng: np.random.Generator, dtype, proj_std: float = 0.02):
        super().__init__()
        if d % n_head:
            raise ValueError("d must be divisible by n_head")
        self.n_head = n_head
        self.qkv = Linear(d, 3 * d, rng, dtype)
        self.proj = Linear(d, d, rng, dtype, std=proj_std)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, C = x.shape
        h, hd = self.n_head, C // self.n_head
        q, k, v = np.split(self.qkv.forward(x), 3, axis=-1)
        q, k, v = (a.reshape(B, T, h, hd).transpose(0, 2, 1, 3) for a in (q, k, v))

        scale = 1.0 / math.sqrt(hd)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        future = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(future, -np.inf, scores)
        att = softmax(scores)  # masked entries become exactly 0

        self.cache = (q, k, v, att, scale)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj.forward(y)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        q, k, v, att, scale = self.cache
        dy = self.proj.backward(dout)
        B, T, C = dy.shape
        h, hd = self.n_head, C // self.n_head

        do = dy.reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        dv = att.transpose(0, 1, 3, 2) @ do
        datt = do @ v.transpose(0, 1, 3, 2)
        # softmax Jacobian-vector product, row-wise
        dscores = att * (datt - (datt * att).sum(axis=-1, keepdims=True))
        dscores *= scale
        dq = dscores @ k
        dk = dscores.transpose(0, 1, 3, 2) @ q

        def merge(a: np.ndarray) -> np.ndarray:
            return a.transpose(0, 2, 1, 3).reshape(B, T, C)

        return self.qkv.backward(np.concatenate([merge(dq), merge(dk), merge(dv)], axis=-1))


class MLP(Module):
    def __init__(self, d: int, rng: np.random.Generator, dtype, proj_std: float = 0.02):
        super().__init__()
        self.fc = Linear(d, 4 * d, rng, dtype)
        self.act = GELU()
        self.fc2 = Linear(4 * d, d, rng, dtype, std=proj_std)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.fc2.forward(self.act.forward(self.fc.forward(x)))

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return self.fc.backward(self.act.backward(self.fc2.backward(dy)))


class Block(Module):
    """Pre-norm transformer block: x + attn(ln1(x)), then + mlp(ln2(.))."""

    def __init__(self, d: int, n_head: int, rng: np.random.Generator, dtype, proj_std: float):
        super().__init__()
        self.ln1 = LayerNorm(d, dtype)
        self.attn = CausalSelfAttention(d, n_head, rng, dtype, proj_std)
        self.ln2 = LayerNorm(d, dtype)
        self.mlp = MLP(d, rng, dtype, proj_std)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x + self.attn.forward(self.ln1.forward(x))
        return x + self.mlp.forward(self.ln2.forward(x))

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dx1 = dy + self.ln2.backward(self.mlp.backward(dy))
        return dx1 + self.ln1.backward(self.attn.backward(dx1))
