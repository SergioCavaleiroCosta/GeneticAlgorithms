from __future__ import annotations
from typing import Tuple, Optional
import random
import numpy as np
from optimization.types import NDArrayFloat
from .protocols import CrossoverStrategy


class ArithmeticCrossover(CrossoverStrategy[NDArrayFloat]):
    def __init__(self, alpha: float = 0.5, prob: float = 1.0, rng: Optional[random.Random] = None) -> None:
        self._alpha = float(alpha)
        if not (0.0 <= prob <= 1.0):
            raise ValueError("prob must be in [0, 1]")
        self._prob = float(prob)
        self._rng = rng or random.Random()

    def crossover(self, a: NDArrayFloat, b: NDArrayFloat) -> Tuple[NDArrayFloat, NDArrayFloat]:
        if self._rng.random() > self._prob:
            return a.copy(), b.copy()
        w = self._alpha
        c1 = w * a + (1.0 - w) * b
        c2 = w * b + (1.0 - w) * a
        return c1.astype(np.float64, copy=False), c2.astype(np.float64, copy=False)

__all__ = ["ArithmeticCrossover"]
