from __future__ import annotations
from typing import Generic
from optimization.population import Population
from optimization.types import ST, OT

class TopKElitism(Generic[ST, OT]):
    def __init__(self, k: int = 1) -> None:
        if k < 0:
            raise ValueError("k must be >= 0")
        self._k = k

    def select_indices(self, population: Population[ST, OT]) -> list[int]:
        n = population.size
        if n == 0 or self._k == 0:
            return []
        indices = list(range(n))
        indices.sort(key=lambda i: population.objectives[i])
        return indices[: min(self._k, n)]

__all__ = ["TopKElitism"]
