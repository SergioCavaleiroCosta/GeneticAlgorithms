from __future__ import annotations
from __future__ import annotations
from typing import Optional
import random
import numpy as np
from optimization.types import NDArrayFloat
from .protocols import MutationStrategy


class UniformMutation(MutationStrategy[NDArrayFloat]):
    """Uniform mutation for normalized ([0,1]) or local noise mode.

    Args:
        scale: Local noise half-range when not normalized.
        prob: Probability of applying mutation to a candidate.
        normalized: If True, perform global resample in [0,1]^d.
        rng: Optional Python RNG; seeds an internal NumPy Generator.
    """

    def __init__(
        self,
        scale: float = 0.1,
        prob: float = 1.0,
        normalized: bool = False,
        rng: Optional[random.Random] = None,
    ) -> None:
        if scale < 0:
            raise ValueError("scale must be >= 0")
        if not (0.0 <= prob <= 1.0):
            raise ValueError("prob must be in [0, 1]")
        self._scale = float(scale)
        self._prob = float(prob)
        self._normalized = normalized
        self._rng = rng or random.Random()
        seed = self._rng.getrandbits(128)
        self._np_rng = np.random.default_rng(seed)

    def mutate(self, x: NDArrayFloat) -> NDArrayFloat:
        if self._rng.random() > self._prob:
            return x
        if self._normalized:
            return self._np_rng.uniform(0.0, 1.0, size=x.shape).astype(np.float64)
        if self._scale == 0.0:
            return x.copy()
        noise = self._np_rng.uniform(low=-self._scale, high=self._scale, size=x.shape).astype(np.float64)
        return (x + noise).astype(np.float64)


__all__ = ["UniformMutation"]
