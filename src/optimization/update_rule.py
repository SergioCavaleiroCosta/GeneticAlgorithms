"""Protocol for updating a solution (one optimization step)."""
from __future__ import annotations
from typing import Protocol, TYPE_CHECKING
from abc import abstractmethod
from .types import ST, OT
from .optimization_problem import OptimizationProblem
from .events import EventDispatcher
if TYPE_CHECKING:
    from .optimization_engine import OptimizationEngine


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
    def step(self, engine: "OptimizationEngine[ST, OT]") -> None:
        """Perform the algorithm-specific update for a single iteration.

        Implementations should mutate ``engine.population`` (e.g.,
        ``update_single`` or ``replace_all``). The engine's template method
        takes care of recording iteration metadata and emitting Stage.ITERATION
        after this method returns.
        """
        ...


__all__ = ["UpdateRule"]
