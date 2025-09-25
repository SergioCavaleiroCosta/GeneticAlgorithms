from __future__ import annotations
from typing import Tuple, Optional
import random
import numpy as np
from optimization.types import NDArrayFloat
from .protocols import CrossoverStrategy


class ArithmeticCrossover(CrossoverStrategy[NDArrayFloat]):
    """Single-locus arithmetic + tail-swap crossover.

    Behavior:
      1. With probability ``prob`` crossover is applied; otherwise parents are copied.
      2. Choose a random allele index ``k`` in ``[0, n-1]`` (n = genome length).
      3. Sample a mixing weight ``w ~ U(0,1)``.
      4. Blend only allele ``k``:

            c1[k] = w * a[k] + (1 - w) * b[k]
            c2[k] = w * b[k] + (1 - w) * a[k]

      5. For all positions strictly after ``k`` (k+1 .. end) swap the parent tails:

            c1[j] = b[j]; c2[j] = a[j]  for j > k

         Positions before k (0 .. k-1) are copied directly from their respective parents.

    This operator combines localized exploration at a single locus (arithmetic blend)
    with disruptive recombination of the remaining tail, encouraging both fine-grained
    exploitation and diversity injection.

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
        # Choose blend locus
        k = self._rng.randrange(n)
        w = self._rng.random()
        # Allocate offspring as copies (so we can modify in-place)
        c1 = a.copy().astype(np.float64, copy=False)
        c2 = b.copy().astype(np.float64, copy=False)
        # Blend only locus k
        ak = a[k]
        bk = b[k]
        c1[k] = w * ak + (1.0 - w) * bk
        c2[k] = w * bk + (1.0 - w) * ak
        # Swap tails after k
        if k + 1 < n:
            c1[k + 1 :], c2[k + 1 :] = b[k + 1 :], a[k + 1 :]
        return c1, c2

__all__ = ["ArithmeticCrossover"]
