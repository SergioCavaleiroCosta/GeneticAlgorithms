"""Generic optimization engine orchestrating initialization, stepping, and convergence."""
from __future__ import annotations
from typing import Generic
from time import perf_counter
from .types import ST, OT
from .optimization_result import OptimizationResult
from .solution_initializer import SolutionInitializer
from .update_rule import UpdateRule
from .convergence_checker import ConvergenceChecker
from .population import Population
from .events import EventDispatcher, Stage
from .state import OptimizationState


class OptimizationEngine(Generic[ST, OT]):
    """
    Reusable loop that runs an optimization process using pluggable components.

    This class is deliberately not a Protocol and can be reused by all algorithms.
    Algorithms focus on the update rule (one step), while the engine handles
    initialization, loop control, convergence checks, and result packaging.
    """

    def __init__(
        self,
        initializer: SolutionInitializer[ST, OT],
        updater: UpdateRule[ST, OT],
        convergence: ConvergenceChecker[ST, OT],
        state: OptimizationState[ST, OT],
        dispatcher: EventDispatcher[ST, OT] | None = None,
    ) -> None:
        self._initializer = initializer
        self._updater = updater
        self._convergence = convergence
        self._dispatcher: EventDispatcher[ST, OT] = dispatcher or EventDispatcher()
        # Live state is the single source of truth
        self._state: OptimizationState[ST, OT] = state
        # Wire dispatcher into strategies if supported
        self._convergence.set_dispatcher(self._dispatcher)
        self._updater.set_dispatcher(self._dispatcher)
        
    @property
    def population(self) -> Population[ST, OT]:
        """Current population (always present; may be empty before initialize())."""
        return self._state.population

    def initialize(self) -> Population[ST, OT]:
        """Reset and prepare the initial state for a run.

        Returns the initial population.
        """
        self._convergence.reset()
        problem = self._updater.problem
        
        # Delegate population construction and seeding to initializer
        population = self._initializer.initialize(problem, self._updater)
        self._state.population = population
        
        return population

    def run(self) -> OptimizationResult[ST, OT]:
        start = perf_counter()
        self.initialize()
        current_solution = self._state.current_solution
        current_objective = self._state.current_objective
        problem = self._updater.problem

        # Inform state to begin the run (after initialization, before events)
        self._state.begin_run(
            initial_solution=current_solution,
            initial_objective=current_objective,
            evaluations=problem.get_evaluation_count(),
        )
        # Emit run-start via dispatcher strategies
        self._dispatcher.emit(engine=self, stage=Stage.RUN_START)

        # Delegate continuation decision to the convergence checker
        while self._convergence.should_continue(self.population):
            # Step
            new_solution, new_objective = self._updater.step(self._convergence)
            current_solution, current_objective = new_solution, new_objective

            # Update population with new current
            self.population.update_single(current_solution, current_objective)
            # Let state record iteration metadata
            self._state.record_iteration(
                iteration=self._convergence.iteration,
                elapsed=perf_counter() - start,
                evaluations=problem.get_evaluation_count(),
            )
            # Emit iteration via dispatcher strategies
            self._dispatcher.emit(engine=self, stage=Stage.ITERATION)

        elapsed = perf_counter() - start
        result = self._state.build_result(
            execution_time=elapsed,
            iterations=self._convergence.iteration,
            success=True,
            termination_reason="stopped by criteria",
        )
        # Emit run-end via dispatcher strategies then notify convergence
        self._dispatcher.emit(engine=self, stage=Stage.RUN_END)
        self._convergence.on_run_completed(
            best_solution=result.best_solution,
            best_objective=result.best_objective,
            elapsed=elapsed,
            evaluations=problem.get_evaluation_count(),
            success=result.success,
            termination_reason=result.termination_reason,
        )
        return result


__all__ = ["OptimizationEngine"]
