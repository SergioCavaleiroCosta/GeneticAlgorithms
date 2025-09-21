"""Protocol defining the interface for optimization algorithms."""
from typing import Protocol, Dict, Optional, Callable
from abc import abstractmethod
from .types import ST, OT
from .optimization_problem import OptimizationProblem
from .optimization_result import OptimizationResult


class OptimizationAlgorithm(Protocol[ST, OT]):
    """Protocol defining the interface for optimization algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the optimization algorithm."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, object]:
        """Get the current algorithm parameters."""
        ...

    @abstractmethod
    def set_parameters(self, **kwargs: object) -> None:
        """Set algorithm parameters."""
        ...

    @abstractmethod
    def optimize(
        self,
        problem: OptimizationProblem[ST, OT],
        max_iterations: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        target_objective: Optional[OT] = None,
        callback: Optional[Callable[[int, ST, OT], bool]] = None,
    ) -> OptimizationResult[ST, OT]:
        """Run the optimization algorithm and return the result container."""
        ...

    def reset(self) -> None:
        """Reset internal state for a fresh run."""
        pass

    def get_default_parameters(self) -> Dict[str, object]:
        """Default parameters for the algorithm."""
        return {}

    def validate_parameters(self) -> bool:
        """Validate current parameters."""
        return True


__all__ = ["OptimizationAlgorithm"]
