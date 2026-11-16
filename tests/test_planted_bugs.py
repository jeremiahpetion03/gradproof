"""Mutation tests: a gradient checker is only worth trusting if it catches
real mistakes. Each case plants a realistic backprop bug and asserts the
checker flags it by orders of magnitude."""

import math
import types

import numpy as np
import pytest

from gradproof.gradcheck import gradcheck
from gradproof.nn import GELU, Block, CausalSelfAttention, Embedding, LayerNorm


def layernorm_drops_mean_term(self, dy):
    d = dy.shape[-1]
    self.grads["g"] += (dy * self.xhat).reshape(-1, d).sum(axis=0)
    self.grads["b"] += dy.reshape(-1, d).sum(axis=0)
    dxhat = dy * self.params["g"]
    return self.rstd * (dxhat - self.xhat * (dxhat * self.xhat).mean(axis=-1, keepdims=True))


def gelu_ignores_tanh_derivative(self, dy):
    return dy * 0.5 * (1.0 + self.t)


def embedding_overwrites_repeated_tokens(self, dy):
    # Fancy-index += silently drops all but one update for repeated ids.
    self.grads["W"][self.idx.reshape(-1)] += dy.reshape(-1, dy.shape[-1])


def block_drops_residual_gradient(self, dy):
    dx1 = self.ln2.backward(self.mlp.backward(dy))  # missing "dy +"
    return dx1 + self.ln1.backward(self.attn.backward(dx1))


def _attention_backward(forget_scale=False, naive_softmax_grad=False):
    def backward(self, dout):
        q, k, v, att, scale = self.cache
        dy = self.proj.backward(dout)
        B, T, C = dy.shape
        h, hd = self.n_head, C // self.n_head
        do = dy.reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        dv = att.transpose(0, 1, 3, 2) @ do
        datt = do @ v.transpose(0, 1, 3, 2)
        if naive_softmax_grad:
            dscores = att * datt  # drops the "- sum(datt * att)" term
        else:
            dscores = att * (datt - (datt * att).sum(axis=-1, keepdims=True))
        if not forget_scale:
            dscores *= scale
        dq, dk = dscores @ k, dscores.transpose(0, 1, 3, 2) @ q

        def merge(a):
            return a.transpose(0, 2, 1, 3).reshape(B, T, C)

        return self.qkv.backward(np.concatenate([merge(dq), merge(dk), merge(dv)], axis=-1))

    return backward


MUTANTS = {
    "layernorm_drops_mean_term": (LayerNorm, layernorm_drops_mean_term),
    "gelu_ignores_tanh_derivative": (GELU, gelu_ignores_tanh_derivative),
    "embedding_overwrites_repeated_tokens": (Embedding, embedding_overwrites_repeated_tokens),
    "block_drops_residual_gradient": (Block, block_drops_residual_gradient),
    "attention_forgets_1_over_sqrt_d": (CausalSelfAttention, _attention_backward(forget_scale=True)),
    "softmax_backward_missing_term": (CausalSelfAttention, _attention_backward(naive_softmax_grad=True)),
}


def _instances(module, cls):
    if isinstance(module, cls):
        yield module
    for _, child in module.children():
        yield from _instances(child, cls)


@pytest.mark.parametrize("name", list(MUTANTS))
def test_planted_bug_is_caught(name, checkable):
    model, loss_fn = checkable
    cls, buggy = MUTANTS[name]
    targets = list(_instances(model, cls))
    assert targets, f"no {cls.__name__} in model"
    for mod in targets:
        mod.backward = types.MethodType(buggy, mod)

    report = gradcheck(model, loss_fn, max_entries=30)
    assert report.max_rel > 1e-2, f"{name} slipped through: max_rel={report.max_rel:.2e}"


def test_mutants_are_real_bugs_not_noise(checkable):
    """Control: with no mutation the same check passes cleanly, so the
    failures above come from the planted bugs, not from the checker."""
    model, loss_fn = checkable
    assert gradcheck(model, loss_fn, max_entries=30).max_rel < 1e-5
    assert math.isfinite(loss_fn(False))
