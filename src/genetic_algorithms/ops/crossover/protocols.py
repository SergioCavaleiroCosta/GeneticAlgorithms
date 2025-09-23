from __future__ import annotations
from typing import Protocol, Tuple, TypeVar

S = TypeVar("S")

class CrossoverStrategy(Protocol[S]):
    """Combine two parents, producing two children of the same shape/type."""
    def crossover(self, a: S, b: S) -> Tuple[S, S]: ...

__all__ = ["CrossoverStrategy"]
