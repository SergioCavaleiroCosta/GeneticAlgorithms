"""Protocol for updating a solution (one optimization step)."""
from typing import Protocol
from abc import abstractmethod
from .types import ST, OT
from .optimization_problem import OptimizationProblem
from .convergence_checker import ConvergenceChecker
from .events import EventDispatcher


class UpdateRule(Protocol[ST, OT]):
    """Perform a single update step given the current state."""

    @abstractmethod
    def set_dispatcher(self, dispatcher: EventDispatcher[ST, OT]) -> None:
        """Provide the shared event dispatcher for emitting algorithm-specific events."""
        ...

    @property
    @abstractmethod
    def problem(self) -> OptimizationProblem[ST, OT]:
        """The optimization problem associated with this update rule."""
        ...

    @abstractmethod
    def seed(self, initial_solution: ST, initial_objective: OT) -> None:
        """Seed the updater with an initial state (solution and its objective)."""
        ...

    @abstractmethod
    def step(
        self,
        convergence: ConvergenceChecker[ST, OT],
    ) -> tuple[ST, OT]:
        """Return the new (solution, objective) after one step.

        The update rule can introspect convergence.iteration and other
        convergence-specific state, and use self.problem as needed.
        """
        ...


__all__ = ["UpdateRule"]
