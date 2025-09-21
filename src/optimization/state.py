"""Dataclass for the live optimization state used across initialize, run, and result.

Note: This module is considered internal infrastructure for the optimization engine.
It is not part of the public API surface; symbols are intentionally not exported
via ``__all__``. Downstream users should not import from this module directly.
"""
from __future__ import annotations
from typing import Generic, List, cast
from dataclasses import dataclass, field
from .types import ST, OT
from .population import Population
from .optimization_result import OptimizationResult

# Internal module: do not export symbols by default
__all__: list[str] = []

@dataclass(slots=True)
class OptimizationState(Generic[ST, OT]):
    """Concrete, minimal state implementation used by the engine by default.

    Internal: not exported via __all__. Holds the live population and collects
    minimal metadata necessary to produce an OptimizationResult. History stores
    best objective per iteration; initial best is recorded at begin_run via
    external recorders if desired, or can be included here if needed.
    """

    # Live population (starts as empty; initialize() will replace it)
    _population: Population[ST, OT] = field(
        default_factory=lambda: cast(Population[ST, OT], Population([], []))
    )

    # Bookkeeping
    _started: bool = False
    _initial_solution: ST | None = None
    _initial_objective: OT | None = None
    _evaluations_at_start: int = 0

    # Simple convergence history of best objective values
    _history: List[OT] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Initialize history with a precisely typed empty list
        self._history = cast(List[OT], [])

    @property
    def population(self) -> Population[ST, OT]:
        return self._population

    @population.setter
    def population(self, population: Population[ST, OT]) -> None:
        self._population = population

    # Note: State is passive and does not own event emission.

    # Projections
    @property
    def current_solution(self) -> ST:
        return self.population.current_solution

    @property
    def current_objective(self) -> OT:
        return self.population.current_objective

    @property
    def best_solution(self) -> ST:
        return self.population.best_solution

    @property
    def best_objective(self) -> OT:
        return self.population.best_objective

    # Lifecycle hooks
    def begin_run(self, initial_solution: ST, initial_objective: OT, evaluations: int) -> None:
        self._started = True
        self._initial_solution = initial_solution
        self._initial_objective = initial_objective
        self._evaluations_at_start = evaluations

    def record_iteration(self, iteration: int, elapsed: float, evaluations: int) -> None:
        # Record current best objective after each engine step
        self._history.append(self.best_objective)

    # Finalization
    def build_result(
        self,
        execution_time: float,
        iterations: int,
        success: bool,
        termination_reason: str,
    ) -> OptimizationResult[ST, OT]:
        if not self._started:
            raise RuntimeError("Run not started; call begin_run() before build_result()")
        return OptimizationResult(
            best_solution=self.best_solution,
            best_objective=self.best_objective,
            convergence_history=list(self._history),
            execution_time=execution_time,
            iterations=iterations,
            success=success,
            termination_reason=termination_reason,
            additional_info={
                "initial_solution": self._initial_solution,
                "initial_objective": self._initial_objective,
                "evaluations_at_start": self._evaluations_at_start,
            },
        )
