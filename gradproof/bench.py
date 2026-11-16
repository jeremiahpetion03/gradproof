"""Measure training-step throughput and activation memory per precision.

    python -m gradproof.bench --batch 64 --repeats 50
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .data import VOCAB, Task
from .model import GPT, GPTConfig, activation_bytes
from .nn import cross_entropy


def bench(dtype: str, batch: int, repeats: int, length: int = 8, d_model: int = 64,
          n_layer: int = 2, n_head: int = 4) -> dict:
    task = Task("sort", length)
    model = GPT(GPTConfig(VOCAB, task.block_size, n_layer, n_head, d_model), seed=0, dtype=np.dtype(dtype))
    x, y, mask = task.sample(np.random.default_rng(0), batch)

    def step() -> None:
        model.zero_grad()
        _, d = cross_entropy(model.forward(x), y, mask)
        model.backward(d)

    for _ in range(5):  # warm up caches and BLAS threads
        step()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        step()
        times.append(time.perf_counter() - t0)
    median = float(np.median(times))
    return {
        "dtype": dtype,
        "batch": batch,
        "median_step_ms": round(median * 1e3, 3),
        "tokens_per_sec": round(batch * task.block_size / median),
        "activation_mb": round(activation_bytes(model) / 2**20, 3),
        "param_mb": round(sum(m.params[k].nbytes for _, m, k in model.named_parameters()) / 2**20, 3),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args(argv)
    rows = [bench(dt, args.batch, args.repeats) for dt in ("float64", "float32")]
    for r in rows:
        print(json.dumps(r))
    f64, f32 = rows
    print(f"float32 vs float64: {f64['median_step_ms'] / f32['median_step_ms']:.2f}x faster, "
          f"{f64['activation_mb'] / f32['activation_mb']:.2f}x less activation memory")


if __name__ == "__main__":
    main()
