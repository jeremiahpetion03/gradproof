"""Synthetic sequence tasks with exact, checkable answers.

Each example is ``digits SEP answer``. The model sees the whole sequence
(teacher forcing) but loss is computed only on the answer positions, and
evaluation is greedy generation of the answer from the prompt alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_DIGITS = 10
SEP = 10
VOCAB = 11

TASKS = {
    "sort": lambda d: np.sort(d, axis=1),
    "reverse": lambda d: d[:, ::-1],
}


@dataclass(frozen=True)
class Task:
    name: str
    length: int  # number of input digits

    @property
    def block_size(self) -> int:
        return 2 * self.length  # inputs are the full sequence minus its last token

    def sample(self, rng: np.random.Generator, batch: int):
        """Return (x, y, mask) for teacher-forced training."""
        digits = rng.integers(0, N_DIGITS, size=(batch, self.length))
        answer = TASKS[self.name](digits)
        seq = np.concatenate([digits, np.full((batch, 1), SEP), answer], axis=1)
        x, y = seq[:, :-1], seq[:, 1:]
        mask = np.zeros(x.shape, dtype=bool)
        mask[:, self.length :] = True  # positions whose target is an answer token
        return x, y, mask

    def eval_set(self, rng: np.random.Generator, n: int):
        """Return (prompt, answer) pairs for generation-based evaluation."""
        digits = rng.integers(0, N_DIGITS, size=(n, self.length))
        prompt = np.concatenate([digits, np.full((n, 1), SEP)], axis=1)
        return prompt, TASKS[self.name](digits)
