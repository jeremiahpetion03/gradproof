"""A small GPT-style decoder built from the hand-written layers in ``nn``."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .nn import Block, Embedding, LayerNorm, Linear, Module, PositionalEmbedding


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 2
    n_head: int = 4
    d_model: int = 64

    def to_dict(self) -> dict:
        return asdict(self)


class GPT(Module):
    def __init__(self, cfg: GPTConfig, seed: int = 0, dtype=np.float32):
        super().__init__()
        self.cfg = cfg
        self.dtype = np.dtype(dtype)
        rng = np.random.default_rng(seed)
        d = cfg.d_model
        # GPT-2 style: scale residual projections by 1/sqrt(2 * n_layer)
        proj_std = 0.02 / math.sqrt(2 * cfg.n_layer)

        self.tok = Embedding(cfg.vocab_size, d, rng, dtype)
        self.pos = PositionalEmbedding(cfg.block_size, d, rng, dtype)
        self.blocks = [Block(d, cfg.n_head, rng, dtype, proj_std) for _ in range(cfg.n_layer)]
        self.ln_f = LayerNorm(d, dtype)
        self.head = Linear(d, cfg.vocab_size, rng, dtype)
        self.zero_grad()

    def forward(self, idx: np.ndarray) -> np.ndarray:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        x = self.tok.forward(idx) + self.pos.forward(T)[None]
        for block in self.blocks:
            x = block.forward(x)
        return self.head.forward(self.ln_f.forward(x))

    def backward(self, dlogits: np.ndarray) -> None:
        dx = self.ln_f.backward(self.head.backward(dlogits))
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        self.tok.backward(dx)
        self.pos.backward(dx)

    def generate(self, prompt: np.ndarray, n_new: int) -> np.ndarray:
        """Greedy decoding. Returns only the n_new generated tokens."""
        idx = prompt
        for _ in range(n_new):
            logits = self.forward(idx[:, -self.cfg.block_size :])
            nxt = logits[:, -1].argmax(axis=-1)[:, None]
            idx = np.concatenate([idx, nxt], axis=1)
        return idx[:, prompt.shape[1] :]

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: mod.params[k].copy() for name, mod, k in self.named_parameters()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for name, mod, k in self.named_parameters():
            mod.params[k][...] = state[name]


def activation_bytes(module: Module) -> int:
    """Bytes held in cached activations (everything an array attribute holds
    that is not a parameter or a gradient). Measured after a forward pass."""
    total = 0
    for value in vars(module).values():
        if isinstance(value, np.ndarray):
            total += value.nbytes
        elif isinstance(value, tuple):
            total += sum(v.nbytes for v in value if isinstance(v, np.ndarray))
    for _, child in module.children():
        total += activation_bytes(child)
    return total
