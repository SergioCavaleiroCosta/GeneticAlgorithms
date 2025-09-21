"""Protocol for checking convergence in an optimization process."""
from typing import Protocol
from abc import abstractmethod
from .types import ST, OT
from .population import Population
from .events import EventDispatcher


class ConvergenceChecker(Protocol[ST, OT]):
    """Decide whether the optimization process should stop and track iteration state."""

    @abstractmethod
    def set_dispatcher(self, dispatcher: EventDispatcher[ST, OT]) -> None:
        """Provide the shared event dispatcher for emitting iteration lifecycle events."""
        ...

    @property
    @abstractmethod
    def iteration(self) -> int:
        """Current iteration count maintained internally by the convergence strategy."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (e.g., iteration counters) before a new run."""
        ...

    @abstractmethod
    def advance_iteration(self) -> None:
        """Advance the internal iteration counter after completing an iteration."""
        ...

    @abstractmethod
    def should_continue(self, population: Population[ST, OT]) -> bool:
        """Return True if the optimization should continue; otherwise False."""
        ...

    @abstractmethod
    def on_run_completed(
        self,
        *,
        best_solution: ST,
        best_objective: OT,
        elapsed: float,
        evaluations: int,
        success: bool,
        termination_reason: str,
    ) -> None:
        """Hook invoked by the engine on termination to allow emitting a completion event."""
        ...


__all__ = ["ConvergenceChecker"]
