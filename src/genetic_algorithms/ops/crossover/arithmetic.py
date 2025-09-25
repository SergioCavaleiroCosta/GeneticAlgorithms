from __future__ import annotations
from typing import Tuple, Optional
import random
import numpy as np
from optimization.types import NDArrayFloat
from .protocols import CrossoverStrategy


class ArithmeticCrossover(CrossoverStrategy[NDArrayFloat]):
    """Random-subset multi-allele arithmetic crossover.

    Behavior when applied (with probability ``prob``):
      1. Let ``n`` be genome length.
      2. Draw an integer subset size ``m`` uniformly from ``[1, n]``.
      3. Sample ``m`` distinct allele indices without replacement.
      4. For each selected index ``i`` sample an independent weight ``w_i ~ U(0,1)`` and blend:

            c1[i] = w_i * a[i] + (1 - w_i) * b[i]
            c2[i] = w_i * b[i] + (1 - w_i) * a[i]

         Non-selected indices are copied directly (c1[i] = a[i], c2[i] = b[i]).

    This generalizes classic arithmetic crossover by introducing adaptive locality: when
    ``m`` is small, it performs fine-grained perturbation; when ``m`` approaches ``n`` it
    approximates full-vector arithmetic blending with per-gene random weights.

    Parameters
    ----------
    prob : float, default 1.0
        Probability of performing the crossover; else parents are cloned.
    rng : random.Random | None
        Optional RNG instance for reproducibility.
    """
    def __init__(self, prob: float = 1.0, rng: Optional[random.Random] = None) -> None:
        # alpha=None -> sample per call; else use fixed provided value
        if not (0.0 <= prob <= 1.0):
            raise ValueError("prob must be in [0, 1]")
        # Probability of executing crossover vs passing parents through.
        self._prob = float(prob)
        # Dedicated RNG allows reproducibility and isolation from global state.
        self._rng = rng or random.Random()

    def crossover(self, a: NDArrayFloat, b: NDArrayFloat) -> Tuple[NDArrayFloat, NDArrayFloat]:
        # With probability (1 - prob), return copies (no crossover applied).
        if self._rng.random() > self._prob:
            return a.copy(), b.copy()
        n = a.shape[0]
        # Draw subset size m in [1, n]
        m = self._rng.randrange(1, n + 1)
        # Sample m distinct indices without replacement
        # For small n we can afford simple sampling loop
        indices = list(range(n))
        self._rng.shuffle(indices)
        sel = indices[:m]
        # Prepare offspring copies
        c1 = a.copy().astype(np.float64, copy=False)
        c2 = b.copy().astype(np.float64, copy=False)
        for idx in sel:
            w = self._rng.random()
            ai = a[idx]
            bi = b[idx]
            c1[idx] = w * ai + (1.0 - w) * bi
            c2[idx] = w * bi + (1.0 - w) * ai
        return c1, c2

__all__ = ["ArithmeticCrossover"]
