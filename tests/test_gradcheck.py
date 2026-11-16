"""Every analytic gradient, checked against finite differences."""

import numpy as np

from gradproof.gradcheck import gradcheck

# Thresholds sit well above the measured noise floor of a correct backward
# pass (max relative error ~3e-7, see README) and far below the error of any
# planted bug (> 1e-2, see test_planted_bugs.py).
REL_TOL = 1e-5
ABS_TOL = 1e-8


def test_every_entry_of_every_parameter(checkable):
    model, loss_fn = checkable
    report = gradcheck(model, loss_fn)  # max_entries=None -> all entries
    assert len(report) == len(list(model.named_parameters()))
    bad = {name: err for name, err in report.items() if err[0] > REL_TOL or err[1] > ABS_TOL}
    assert not bad, f"gradient mismatch: {bad}"


def test_gradients_hold_across_seeds():
    from conftest import make_checkable_model, make_loss_fn

    for seed in (1, 2, 3):
        model = make_checkable_model(seed=seed)
        report = gradcheck(model, make_loss_fn(model, seed=seed + 10), max_entries=40, seed=seed)
        assert report.max_rel < REL_TOL and report.max_abs < ABS_TOL, (seed, report.max_rel, report.max_abs)


def test_key_bias_gradient_is_exactly_zero_in_theory(checkable):
    """Adding a constant vector to every key shifts each row of attention
    scores by the same amount, and softmax is invariant to that. So the
    gradient with respect to the key bias must vanish (up to rounding)."""
    model, loss_fn = checkable
    model.zero_grad()
    loss_fn(True)
    d = model.cfg.d_model
    for block in model.blocks:
        db = block.attn.qkv.grads["b"]
        key_part, query_part = db[d : 2 * d], db[:d]
        assert np.abs(key_part).max() < 1e-12
        assert np.abs(query_part).max() > 1e-4  # sanity: query bias does matter
