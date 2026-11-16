import numpy as np

from gradproof.train import TrainConfig, train

QUICK = dict(length=4, steps=40, batch=16, d_model=16, n_head=2, eval_n=20, log_every=1000)


def test_same_seed_is_bitwise_reproducible():
    a = train(TrainConfig(seed=7, **QUICK), verbose=False)
    b = train(TrainConfig(seed=7, **QUICK), verbose=False)
    assert a["losses"] == b["losses"]  # exact equality, not allclose
    sa, sb = a["model"].state_dict(), b["model"].state_dict()
    for name in sa:
        np.testing.assert_array_equal(sa[name], sb[name], err_msg=name)


def test_different_seed_gives_a_different_run():
    a = train(TrainConfig(seed=1, **QUICK), verbose=False)
    b = train(TrainConfig(seed=2, **QUICK), verbose=False)
    assert a["losses"] != b["losses"]


def test_reverse_task_learns(tmp_path):
    """End-to-end smoke test: a short run must actually learn the task,
    and must write every reproducibility artifact."""
    cfg = TrainConfig(task="reverse", length=4, steps=400, batch=32, d_model=32,
                      n_head=2, eval_n=200, log_every=100, seed=0)
    result = train(cfg, out=tmp_path, verbose=False)
    assert result["losses"][-1] < 0.1 * result["losses"][0]
    assert result["eval_exact_match"] > 0.9
    for name in ("config.json", "metrics.jsonl", "final.json", "model.npz"):
        assert (tmp_path / name).exists(), name
