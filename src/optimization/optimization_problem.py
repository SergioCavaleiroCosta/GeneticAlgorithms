"""Protocol defining the interface for optimization problems."""
from typing import Protocol, Optional, List, Tuple
from .types import ST, OT_co
from abc import abstractmethod


class OptimizationProblem(Protocol[ST, OT_co]):
    """Protocol defining the interface for optimization problems."""

    @abstractmethod
    def evaluate(self, solution: ST) -> OT_co:
        """Evaluate the objective function for a given solution."""
        ...

    @abstractmethod
    def is_feasible(self, solution: ST) -> bool:
        """Check if a solution satisfies constraints."""
        ...

    @abstractmethod
    def generate_random_solution(self) -> ST:
        """Generate a random valid solution."""
        ...

    def get_evaluation_count(self) -> int:
        """Return the number of objective evaluations performed so far.

        Implementations should increment an internal counter each time
        `evaluate` is called. Default returns 0 for problems that do not
        track evaluations.
        """
        return 0

    def get_bounds(self) -> Optional[List[Tuple[float, float]]]:
        """Bounds for decision variables (min, max) per variable, if applicable."""
        return None

    def get_dimension(self) -> Optional[int]:
        """Dimension of the solution space, if applicable."""
        return None


__all__ = ["OptimizationProblem"]
