"""Reproducible training run from the command line.

    python -m gradproof.train --task sort --steps 3000 --seed 0 --out runs/sort

Writes to --out:
    config.json    every setting needed to reproduce the run
    metrics.jsonl  one line per logged step (loss, grad norm, lr)
    final.json     held-out exact-match accuracy, timing, throughput
    model.npz      trained weights
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import VOCAB, Task
from .model import GPT, GPTConfig
from .nn import cross_entropy
from .optim import AdamW, clip_grad_norm, lr_at


@dataclass(frozen=True)
class TrainConfig:
    task: str = "sort"
    length: int = 8
    steps: int = 3000
    batch: int = 64
    lr: float = 3e-3
    warmup: int = 100
    weight_decay: float = 0.01
    clip: float = 1.0
    n_layer: int = 2
    n_head: int = 4
    d_model: int = 64
    dtype: str = "float32"
    seed: int = 0
    eval_n: int = 1000
    log_every: int = 100


def train(cfg: TrainConfig, out: Path | None = None, verbose: bool = True) -> dict:
    task = Task(cfg.task, cfg.length)
    model_cfg = GPTConfig(VOCAB, task.block_size, cfg.n_layer, cfg.n_head, cfg.d_model)
    model = GPT(model_cfg, seed=cfg.seed, dtype=np.dtype(cfg.dtype))
    opt = AdamW(model, weight_decay=cfg.weight_decay)
    data_rng = np.random.default_rng(cfg.seed + 1)

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        meta = {"train": asdict(cfg), "model": model_cfg.to_dict(), "num_params": model.num_params(),
                "numpy": np.__version__, "python": platform.python_version()}
        (out / "config.json").write_text(json.dumps(meta, indent=2))
        log = (out / "metrics.jsonl").open("w")

    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(cfg.steps):
        x, y, mask = task.sample(data_rng, cfg.batch)
        model.zero_grad()
        loss, dlogits = cross_entropy(model.forward(x), y, mask)
        model.backward(dlogits)
        gnorm = clip_grad_norm(model, cfg.clip)
        lr = lr_at(step, cfg.steps, cfg.lr, cfg.warmup)
        opt.step(lr)
        losses.append(loss)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            rec = {"step": step, "loss": round(loss, 6), "grad_norm": round(gnorm, 6), "lr": lr}
            if out is not None:
                log.write(json.dumps(rec) + "\n")
            if verbose:
                print(f"step {step:5d}  loss {loss:.4f}  grad_norm {gnorm:.3f}  lr {lr:.2e}")
    train_s = time.perf_counter() - t0
    if out is not None:
        log.close()

    acc = evaluate(model, task, np.random.default_rng(10_000 + cfg.seed), cfg.eval_n)
    result = {
        "final_loss": losses[-1],
        "eval_exact_match": acc["exact_match"],
        "eval_token_accuracy": acc["token_accuracy"],
        "eval_n": cfg.eval_n,
        "train_seconds": round(train_s, 2),
        "train_tokens_per_sec": round(cfg.steps * cfg.batch * (task.block_size) / train_s),
        "num_params": model.num_params(),
    }
    if out is not None:
        (out / "final.json").write_text(json.dumps(result, indent=2))
        np.savez(out / "model.npz", **model.state_dict())
    if verbose:
        print(json.dumps(result, indent=2))
    result["losses"] = losses
    result["model"] = model
    return result


def evaluate(model: GPT, task: Task, rng: np.random.Generator, n: int) -> dict:
    prompt, answer = task.eval_set(rng, n)
    pred = model.generate(prompt, task.length)
    return {
        "exact_match": float((pred == answer).all(axis=1).mean()),
        "token_accuracy": float((pred == answer).mean()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field, default in asdict(TrainConfig()).items():
        parser.add_argument(f"--{field.replace('_', '-')}", type=type(default), default=default)
    parser.add_argument("--out", type=Path, default=None)
    args = vars(parser.parse_args(argv))
    out = args.pop("out")
    train(TrainConfig(**args), out=out)


if __name__ == "__main__":
    main()
