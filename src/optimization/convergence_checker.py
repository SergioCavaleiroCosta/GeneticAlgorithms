"""Convergence checking: concrete checker with pluggable stop strategies."""
from __future__ import annotations
from typing import Protocol, Iterable, List, Generic, cast, TYPE_CHECKING
from abc import abstractmethod
from .types import ST, OT
from .events import EventDispatcher

if TYPE_CHECKING:  # avoid circular imports at runtime
    from .optimization_engine import OptimizationEngine


class _IterationAware(Protocol):
    iteration: int


class ConvergenceChecker(Generic[ST, OT]):
    """Concrete convergence checker that uses a list of stop strategies.

    - Maintains a 1-based iteration counter.
    - Continues while no strategy requests stop (logical OR for stopping).
    """

    def __init__(
        self,
        strategies: Iterable["StopStrategy[ST, OT]"] = [],
        iteration_strategy: "MaxIterationsStop[ST, OT] | None" = None,
    ) -> None:
        # Ensure an iteration strategy always exists (default is a very large cap)
        if iteration_strategy is None:
            iteration_strategy = MaxIterationsStop()

        # Build the evaluation order: iteration strategy first
        ordered: List[StopStrategy[ST, OT]] = [iteration_strategy]
        for s in strategies:
            if s is not iteration_strategy:
                ordered.append(s)
        
        self._strategies: List[StopStrategy[ST, OT]] = ordered
        self._dispatcher: EventDispatcher[ST, OT] | None = None

    def set_dispatcher(self, dispatcher: EventDispatcher[ST, OT]) -> None:
        self._dispatcher = dispatcher

    @property
    def iteration(self) -> int:
        # First strategy is guaranteed to be the iteration-aware one
        first = self._strategies[0]
        return cast(_IterationAware, first).iteration

    def reset(self) -> None:
        for s in self._strategies:
            s.reset()

    def should_continue(self, engine: "OptimizationEngine[ST, OT]") -> bool:
        # Stop if any strategy requests termination. The iteration strategy, if present,
        # is evaluated first and can update its own internal counter.
        for s in self._strategies:
            if s.should_stop(engine):
                return False
        return True

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
        # Default no-op; users can implement their own subclass if they need hooks.
        _ = (best_solution, best_objective, elapsed, evaluations, success, termination_reason)

class StopStrategy(Protocol[ST, OT]):
    """Single continuation criterion invoked each iteration.

    Return True from ``should_stop`` to request termination; False to continue.
    """

    @abstractmethod
    def should_stop(self, engine: "OptimizationEngine[ST, OT]") -> bool:
        ...

    def reset(self) -> None:  # default no-op allowed by Protocol at runtime
        pass


class MaxIterationsStop(StopStrategy[ST, OT]):
    """Continue while iteration < max_iterations (strict).

    Uses 1-based iteration counter as maintained by the composite.
    """

    def __init__(self, max_iterations: int = 1_000_000_000_000) -> None:
        self.iteration = 0
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        self._max = max_iterations

    def should_stop(self, engine: "OptimizationEngine[ST, OT]") -> bool:  # noqa: ARG002
        self.iteration += 1
        return self.iteration >= self._max

    def reset(self) -> None:
        # Stateless; nothing to reset
        self.iteration = 0


class MaxEvaluationsStop(StopStrategy[ST, OT]):
    """Stop when the number of objective evaluations reaches a maximum."""

    def __init__(self, max_evaluations: int) -> None:
        if max_evaluations <= 0:
            raise ValueError("max_evaluations must be > 0")
        self._max = max_evaluations

    def should_stop(self, engine: "OptimizationEngine[ST, OT]") -> bool:
        # Access evaluation count via the engine's public problem property
        return engine.problem.evaluation_count >= self._max

    def reset(self) -> None:
        # Stateless
        pass



__all__ = [
    "ConvergenceChecker",
    "StopStrategy",
    "MaxIterationsStop",
    "MaxEvaluationsStop",
]
