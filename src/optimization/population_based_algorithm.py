"""Protocol for population-based optimization algorithms."""
from typing import List, Protocol
from abc import abstractmethod
from .types import ST, OT
from .optimization_algorithm import OptimizationAlgorithm
from .optimization_problem import OptimizationProblem


class PopulationBasedAlgorithm(OptimizationAlgorithm[ST, OT], Protocol):
    """Extends base algorithms with population-specific methods."""

    @property
    @abstractmethod
    def population_size(self) -> int:
        """Get the current population size."""
        ...

    @abstractmethod
    def set_population_size(self, size: int) -> None:
        """Set the population size."""
        ...

    @abstractmethod
    def get_population(self) -> List[ST]:
        """Get the current population."""
        ...

    @abstractmethod
    def initialize_population(self, problem: OptimizationProblem[ST, OT]) -> None:
        """Initialize the population for the given problem."""
        ...


__all__ = ["PopulationBasedAlgorithm"]
