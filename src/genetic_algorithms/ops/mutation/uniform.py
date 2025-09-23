from __future__ import annotations
from typing import Sequence, Optional
import random
import numpy as np
from optimization.types import NDArrayFloat
from .protocols import MutationStrategy


class UniformMutation(MutationStrategy[NDArrayFloat]):
    """Uniform mutation with global-in-bounds or local-noise modes.

    - If bounds are provided, a global uniform mutation is applied: each
      dimension is resampled uniformly within its [lo, hi] bound.
    - If bounds are not provided, a local mutation is applied by adding
      per-dimension uniform noise in [-scale, scale].
    - A global application probability controls whether mutation is applied.
    """

    def __init__(
        self,
        scale: float = 0.1,
        bounds: Sequence[tuple[float, float]] | None = None,
        prob: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        if scale < 0:
            raise ValueError("scale must be >= 0")
        if not (0.0 <= prob <= 1.0):
            raise ValueError("prob must be in [0, 1]")
        self._scale = float(scale)
        self._bounds = list(bounds) if bounds is not None else None
        self._prob = float(prob)
        # Single source of truth for randomness: a Python RNG plus a NumPy Generator seeded from it
        self._rng = rng or random.Random()
        # Seed NumPy Generator from Python RNG to ensure determinism across both probability and noise
        seed = self._rng.getrandbits(128)
        self._np_rng = np.random.default_rng(seed)

    def mutate(self, x: NDArrayFloat) -> NDArrayFloat:
        if self._rng.random() > self._prob:
            return x
        # Global uniform mutation within bounds if available
        if self._bounds is not None:
            y = np.empty_like(x, dtype=np.float64)
            for i, (lo, hi) in enumerate(self._bounds):
                lo = float(min(lo, hi))
                hi = float(max(lo, hi))
                y[i] = self._np_rng.uniform(lo, hi)
            return y
        # Fallback: local uniform noise if no bounds are provided
        if self._scale == 0.0:
            return x.copy()
        noise = self._np_rng.uniform(low=-self._scale, high=self._scale, size=x.shape).astype(np.float64)
        return (x + noise).astype(np.float64)


__all__ = ["UniformMutation"]
