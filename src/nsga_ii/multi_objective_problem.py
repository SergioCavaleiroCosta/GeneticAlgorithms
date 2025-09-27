"""Protocol for multi-objective optimization problems."""
from __future__ import annotations
from typing import Protocol, Sequence
from optimization.types import ST_contra


class MultiObjectiveProblem(Protocol[ST_contra]):
    """Protocol defining the interface for multi-objective optimization problems.
    
    Unlike single-objective problems that return a scalar, multi-objective problems
    return a sequence of objective values.
    """

    def evaluate(self, solution: ST_contra) -> Sequence[float]:
        """Evaluate all objectives for a given solution.
        
        Args:
            solution: The candidate solution to evaluate
            
        Returns:
            A sequence of objective values (typically a list or numpy array)
        """
        ...

    def is_feasible(self, solution: ST_contra) -> bool:
        """Check if a solution satisfies constraints."""
        ...

    @property
    def num_objectives(self) -> int:
        """Number of objectives in this problem."""
        ...

    @property
    def evaluation_count(self) -> int:
        """Number of objective evaluations performed so far."""
        ...

    @evaluation_count.setter
    def evaluation_count(self, value: int) -> None:
        """Set the number of evaluations (e.g., reset to 0 at run start)."""
        ...

    def increment_evaluation_count(self) -> None:
        """Increment the evaluation counter by 1."""
        ...


__all__ = ["MultiObjectiveProblem"]