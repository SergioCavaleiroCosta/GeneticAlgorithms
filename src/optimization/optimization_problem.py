"""Protocol defining the interface for optimization problems and a base class.

Contains:
- OptimizationProblem: typing Protocol describing the required API
- BaseOptimizationProblem: reusable ABC that implements evaluation counting
"""
from typing import Protocol, Optional, List, Tuple, Generic
from .types import ST, ST_contra, OT_co
from abc import abstractmethod, ABC


class OptimizationProblem(Protocol[ST_contra, OT_co]):
    """Protocol defining the interface for optimization problems."""

    def evaluate(self, solution: ST_contra) -> OT_co:
        """Evaluate the objective function for a given solution."""
        ...

    def is_feasible(self, solution: ST_contra) -> bool:
        """Check if a solution satisfies constraints."""
        ...

    # Evaluation counting as a property (getter + setter)
    @property
    def evaluation_count(self) -> int:
        """Number of objective evaluations performed so far."""
        ...

    @evaluation_count.setter
    def evaluation_count(self, value: int) -> None:
        """Set the number of evaluations (e.g., reset to 0 at run start)."""
        ...

    def increment_evaluation_count(self) -> None:
        """Increment the evaluation counter by 1.

        Typical implementations call this inside `evaluate` each time the
        objective is computed.
        """
        ...
    
    @property
    def bounds(self) -> Optional[List[Tuple[float, float]]]:
        """Bounds for decision variables (min, max) per variable, if applicable."""
        return None

    @property
    def dimension(self) -> Optional[int]:
        """Dimension of the solution space, if applicable."""
        return None


class BaseOptimizationProblem(ABC, Generic[ST, OT_co]):
    """Abstract base class providing evaluation counting boilerplate.

        Subclasses must implement:
            - evaluate(solution: ST) -> OT_co
            - is_feasible(solution: ST) -> bool
    """

    def __init__(self) -> None:
        self._evaluation_count: int = 0

    # Evaluation counting as a property (getter + setter)
    @property
    def evaluation_count(self) -> int:
        """Number of objective evaluations performed so far."""
        return self._evaluation_count

    @evaluation_count.setter
    def evaluation_count(self, value: int) -> None:
        """Set the number of evaluations (e.g., reset to 0 at run start)."""
        if value < 0:
            raise ValueError("evaluation_count cannot be negative")
        self._evaluation_count = value

    def increment_evaluation_count(self) -> None:
        """Increment the evaluation counter by 1.

        Typical implementations call this inside `evaluate` each time the
        objective is computed.
        """
        self._evaluation_count += 1

    @property
    def bounds(self) -> Optional[List[Tuple[float, float]]]:
        return None

    @property
    def dimension(self) -> Optional[int]:
        return None

    # Required abstract API matching OptimizationProblem
    @abstractmethod
    def evaluate(self, solution: ST) -> OT_co:
        ...

    @abstractmethod
    def is_feasible(self, solution: ST) -> bool:
        ...


__all__ = ["OptimizationProblem", "BaseOptimizationProblem"]
