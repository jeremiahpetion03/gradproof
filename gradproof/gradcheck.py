"""Finite-difference gradient checking.

For each checked parameter entry p we compare the analytic gradient from
``backward`` against the central difference

    (L(p + eps) - L(p - eps)) / (2 * eps)

whose truncation error is O(eps^2). Run in float64: in float32 the rounding
error of the difference dominates and a correct gradient cannot be told
apart from a slightly wrong one.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .nn import Module

LossFn = Callable[[bool], float]  # loss_fn(backward) -> loss


#: Below this magnitude a gradient is dominated by finite-difference rounding
#: noise (~ machine_eps * |L| / eps ~ 5e-11 for L ~ 2, eps = 1e-5), so a
#: relative error there measures the noise, not the backward pass. Such
#: entries are judged by absolute error instead. Measured on this model:
#: eps = 1e-5 balances O(eps^2) truncation against rounding; at 1e-4
#: truncation dominates, at 1e-6 rounding does.
REL_FLOOR = 1e-4
DEFAULT_EPS = 1e-5


class GradReport(dict):
    """{parameter name: (max relative error, max absolute error)}."""

    @property
    def max_rel(self) -> float:
        return max(r for r, _ in self.values())

    @property
    def max_abs(self) -> float:
        return max(a for _, a in self.values())


def gradcheck(
    model: Module,
    loss_fn: LossFn,
    eps: float = DEFAULT_EPS,
    max_entries: int | None = None,
    seed: int = 0,
) -> GradReport:
    """Compare analytic and numeric gradients for every parameter.

    Relative error is taken over entries with |grad| >= REL_FLOOR; absolute
    error over all checked entries. ``loss_fn(True)`` must run forward +
    backward; ``loss_fn(False)`` forward only. ``max_entries=None`` checks
    every entry of every parameter.
    """
    for _, mod, k in model.named_parameters():
        if mod.params[k].dtype != np.float64:
            raise TypeError("gradcheck requires a float64 model")

    rng = np.random.default_rng(seed)
    model.zero_grad()
    loss_fn(True)
    analytic = {name: mod.grads[k].copy() for name, mod, k in model.named_parameters()}

    report = GradReport()
    for name, mod, k in model.named_parameters():
        p = mod.params[k]
        flat = np.arange(p.size)
        if max_entries is not None and p.size > max_entries:
            flat = rng.choice(p.size, size=max_entries, replace=False)
        worst_rel = worst_abs = 0.0
        for i in flat:
            ix = np.unravel_index(i, p.shape)
            old = p[ix]
            p[ix] = old + eps
            plus = loss_fn(False)
            p[ix] = old - eps
            minus = loss_fn(False)
            p[ix] = old
            numeric = (plus - minus) / (2 * eps)
            a = float(analytic[name][ix])
            diff = abs(a - numeric)
            worst_abs = max(worst_abs, diff)
            scale = max(abs(a), abs(numeric))
            if scale >= REL_FLOOR:
                worst_rel = max(worst_rel, diff / scale)
        report[name] = (worst_rel, worst_abs)
    return report
