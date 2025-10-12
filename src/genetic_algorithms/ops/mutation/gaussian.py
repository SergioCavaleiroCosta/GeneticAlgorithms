from __future__ import annotations
from typing import Sequence, Optional
import random
import numpy as np
from optimization.types import NDArrayFloat
from .protocols import MutationStrategy


class GaussianMutation(MutationStrategy[NDArrayFloat]):
    def __init__(self, sigma: float = 0.1, bounds: Sequence[tuple[float, float]] | None = None, prob: float = 1.0, rng: Optional[random.Random] = None) -> None:
        self._sigma = float(sigma)
        self._bounds = list(bounds) if bounds is not None else None
        if not (0.0 <= prob <= 1.0):
            raise ValueError("prob must be in [0, 1]")
        self._prob = float(prob)
        # Single source of truth for randomness: a Python RNG plus a NumPy Generator seeded from it
        self._rng = rng or random.Random()
        seed = self._rng.getrandbits(128)
        self._np_rng = np.random.default_rng(seed)

    def mutate(self, x: NDArrayFloat) -> NDArrayFloat:
        # Apply mutation probability per gene, not per individual
        y = x.copy().astype(np.float64)
        
        # Generate mutation mask: each gene has prob chance of being mutated
        mutation_mask = self._np_rng.random(size=x.shape) < self._prob
        
        if not mutation_mask.any():
            return y  # No genes selected for mutation
            
        # Generate noise only for selected genes
        noise = self._np_rng.normal(loc=0.0, scale=self._sigma, size=x.shape).astype(np.float64)
        
        # Apply noise only to selected genes
        y[mutation_mask] += noise[mutation_mask]
        
        # Apply bounds if provided
        # if self._bounds is not None:
        #     for i, (lo, hi) in enumerate(self._bounds):
        #         lo = float(min(lo, hi))
        #         hi = float(max(lo, hi))
        #         y[i] = np.clip(y[i], lo, hi)
        return y

__all__ = ["GaussianMutation"]
