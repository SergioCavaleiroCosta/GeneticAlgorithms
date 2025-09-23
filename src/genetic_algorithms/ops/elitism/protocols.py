from __future__ import annotations
from typing import Protocol
from optimization.population import Population
from optimization.types import ST, OT

class ElitismStrategy(Protocol[ST, OT]):
    """Select elite indices from the current population to carry over."""
    def select_indices(self, population: Population[ST, OT]) -> list[int]: ...

__all__ = ["ElitismStrategy"]
