from __future__ import annotations
from typing import Protocol, Sequence
from optimization.types import OT_contra

class SelectionStrategy(Protocol[OT_contra]):
    """Select a parent index from a population given objective values (minimization)."""
    def select(self, objectives: Sequence[OT_contra]) -> int: ...

__all__ = ["SelectionStrategy"]
