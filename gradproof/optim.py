"""AdamW, gradient clipping, and a warmup + cosine learning-rate schedule."""

from __future__ import annotations

import math

import numpy as np

from .nn import Module


def clip_grad_norm(model: Module, max_norm: float) -> float:
    """Scale all gradients so their global L2 norm is at most max_norm.
    Returns the norm measured before clipping."""
    total = math.sqrt(
        sum(float((mod.grads[k].astype(np.float64) ** 2).sum()) for _, mod, k in model.named_parameters())
    )
    if not math.isfinite(total):
        raise FloatingPointError(f"non-finite gradient norm: {total}")
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for _, mod, k in model.named_parameters():
            mod.grads[k] *= scale
    return total


def lr_at(step: int, max_steps: int, lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


class AdamW:
    """Decoupled weight decay (Loshchilov & Hutter). Decay applies only to
    2-D weight matrices, not to biases, LayerNorm gains, or embeddings."""

    def __init__(self, model: Module, betas=(0.9, 0.99), eps: float = 1e-8, weight_decay: float = 0.01):
        self.slots = list(model.named_parameters())
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = [np.zeros_like(mod.params[k]) for _, mod, k in self.slots]
        self.v = [np.zeros_like(mod.params[k]) for _, mod, k in self.slots]

    def step(self, lr: float) -> None:
        self.t += 1
        c1 = 1.0 - self.b1**self.t
        c2 = 1.0 - self.b2**self.t
        for i, (name, mod, k) in enumerate(self.slots):
            p, g = mod.params[k], mod.grads[k]
            decays = p.ndim == 2 and not name.startswith(("tok.", "pos."))
            if self.wd and decays:
                p -= lr * self.wd * p
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * g * g
            p -= lr * (self.m[i] / c1) / (np.sqrt(self.v[i] / c2) + self.eps)
