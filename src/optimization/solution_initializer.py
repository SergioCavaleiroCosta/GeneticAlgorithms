"""Protocol for initializing a single solution and objective."""
from typing import Protocol
from abc import abstractmethod
from .types import ST, OT
from .optimization_problem import OptimizationProblem


class SolutionInitializer(Protocol[ST, OT]):
    """Create an initial feasible solution and its objective for a problem."""

    @abstractmethod
    def initialize(self, problem: OptimizationProblem[ST, OT]) -> tuple[ST, OT]:
        """Return an initial (solution, objective) for the given problem."""
        ...


__all__ = ["SolutionInitializer"]
