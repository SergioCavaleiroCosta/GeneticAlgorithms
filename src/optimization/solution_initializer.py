"""Protocol for initializing a single solution."""
from typing import Protocol
from abc import abstractmethod
from .types import ST, OT_co
from .optimization_problem import OptimizationProblem


class SolutionInitializer(Protocol[ST]):
    """Create an initial feasible solution for a problem."""

    @abstractmethod
    def initialize(self, problem: OptimizationProblem[ST, OT_co]) -> ST:
        """Return an initial solution for the given problem."""
        ...


__all__ = ["SolutionInitializer"]
