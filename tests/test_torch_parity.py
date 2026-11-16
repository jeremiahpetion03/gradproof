"""Independent reference: rebuild the same network from PyTorch primitives,
load identical weights, and require identical loss and gradients from
autograd. Skipped automatically when PyTorch is not installed (CI installs it)."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from conftest import SMALL, TASK, make_checkable_model  # noqa: E402
from gradproof.nn import cross_entropy  # noqa: E402


def torch_loss(P: dict, x: np.ndarray, y: np.ndarray, mask: np.ndarray, cfg) -> "torch.Tensor":
    B, T = x.shape
    d, h = cfg.d_model, cfg.n_head
    hd = d // h
    xi, yi = torch.as_tensor(x), torch.as_tensor(y)
    w = torch.as_tensor(mask, dtype=torch.float64)
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

    a = P["tok.W"][xi] + P["pos.P"][:T]
    for i in range(cfg.n_layer):
        p = f"blocks.{i}."
        z = F.layer_norm(a, (d,), P[p + "ln1.g"], P[p + "ln1.b"], eps=1e-5)
        q, k, v = (z @ P[p + "attn.qkv.W"] + P[p + "attn.qkv.b"]).split(d, dim=-1)
        q, k, v = (t.reshape(B, T, h, hd).transpose(1, 2) for t in (q, k, v))
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(hd)).masked_fill(causal, float("-inf")).softmax(dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, d)
        a = a + o @ P[p + "attn.proj.W"] + P[p + "attn.proj.b"]
        z = F.layer_norm(a, (d,), P[p + "ln2.g"], P[p + "ln2.b"], eps=1e-5)
        z = F.gelu(z @ P[p + "mlp.fc.W"] + P[p + "mlp.fc.b"], approximate="tanh")
        a = a + z @ P[p + "mlp.fc2.W"] + P[p + "mlp.fc2.b"]
    a = F.layer_norm(a, (d,), P["ln_f.g"], P["ln_f.b"], eps=1e-5)
    logits = a @ P["head.W"] + P["head.b"]
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yi.reshape(-1), reduction="none")
    return (nll * w.reshape(-1)).sum() / w.sum()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_loss_and_every_gradient_match_pytorch_autograd(seed):
    model = make_checkable_model(seed=seed)
    x, y, mask = TASK.sample(np.random.default_rng(100 + seed), 4)

    model.zero_grad()
    loss, dlogits = cross_entropy(model.forward(x), y, mask)
    model.backward(dlogits)

    P = {name: torch.tensor(mod.params[k], dtype=torch.float64, requires_grad=True)
         for name, mod, k in model.named_parameters()}
    ref = torch_loss(P, x, y, mask, SMALL)
    ref.backward()

    np.testing.assert_allclose(loss, ref.item(), rtol=1e-12)
    for name, mod, k in model.named_parameters():
        np.testing.assert_allclose(mod.grads[k], P[name].grad.numpy(), rtol=1e-8, atol=1e-12, err_msg=name)
