from __future__ import annotations
from typing import Sequence
import random
from .protocols import SelectionStrategy


class TournamentSelection(SelectionStrategy[float]):
    def __init__(self, k: int = 2, rng: random.Random | None = None) -> None:
        if k <= 0:
            raise ValueError("k must be > 0")
        self._k = k
        self._rng = rng or random.Random()

    def select(self, objectives: Sequence[float]) -> int:
        n = len(objectives)
        if n == 0:
            raise ValueError("cannot select from empty objectives")
        best_i: int | None = None
        best_val: float | None = None
        for _ in range(self._k):
            i = self._rng.randrange(n)
            val = float(objectives[i])
            if best_val is None or val < best_val:
                best_i = i
                best_val = val
        assert best_i is not None
        return best_i

__all__ = ["TournamentSelection"]
