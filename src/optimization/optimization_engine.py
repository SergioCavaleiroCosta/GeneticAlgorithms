"""Generic optimization engine orchestrating initialization, stepping, and convergence."""
from __future__ import annotations
from typing import Generic, List
from time import perf_counter
from .types import ST, OT
from .optimization_result import OptimizationResult
from .solution_initializer import SolutionInitializer
from .update_rule import UpdateRule
from .convergence_checker import ConvergenceChecker
from .population import Population
from .events import (
    EventDispatcher,
    RunStarted,
)


class OptimizationEngine(Generic[ST, OT]):
    """
    Reusable loop that runs an optimization process using pluggable components.

    This class is deliberately not a Protocol and can be reused by all algorithms.
    Algorithms focus on the update rule (one step), while the engine handles
    initialization, loop control, convergence checks, and result packaging.
    """

    def __init__(
        self,
        initializer: SolutionInitializer[ST],
        updater: UpdateRule[ST, OT],
        convergence: ConvergenceChecker[ST, OT],
        dispatcher: EventDispatcher[ST, OT] | None = None,
    ) -> None:
        self._initializer = initializer
        self._updater = updater
        self._convergence = convergence
        self._dispatcher: EventDispatcher[ST, OT] = dispatcher or EventDispatcher()
        # Wire dispatcher into strategies if supported
        try:
            self._convergence.set_dispatcher(self._dispatcher)
        except AttributeError:
            pass
        try:
            self._updater.set_dispatcher(self._dispatcher) 
        except AttributeError:
            pass

    def initialize(self) -> tuple[Population[ST, OT], ST, OT]:
        """Reset and prepare the initial state for a run.

        Returns the initial Population, solution, and objective.
        """
        self._convergence.reset()
        problem = self._updater.problem
        initial_solution = self._initializer.initialize(problem)
        initial_objective = problem.evaluate(initial_solution)
        population: Population[ST, OT] = Population([initial_solution], [initial_objective])
        self._updater.seed(initial_solution, initial_objective)
        # Emit run started
        self._dispatcher.emit(
            RunStarted(
                elapsed=0.0,
                evaluations=problem.get_evaluation_count(),
                initial_solution=initial_solution,
                initial_objective=initial_objective,
            )
        )
        return population, initial_solution, initial_objective

    def run(self) -> OptimizationResult[ST, OT]:
        start = perf_counter()
        population, current_solution, current_objective = self.initialize()
        problem = self._updater.problem
        history: List[OT] = [current_objective]

        # Delegate continuation decision to the convergence checker
        while self._convergence.should_continue(population):
            
            # Step
            new_solution, new_objective = self._updater.step(self._convergence)
            current_solution, current_objective = new_solution, new_objective
            
            # Update population with new current
            population.update_single(current_solution, current_objective)

            # Track best
            history.append(population.best_objective)

        elapsed = perf_counter() - start
        result = OptimizationResult(
            best_solution=population.best_solution,
            best_objective=population.best_objective,
            convergence_history=history,
            execution_time=elapsed,
            iterations=self._convergence.iteration,
            success=True,
            termination_reason="stopped by criteria",
            additional_info={"evaluations": problem.get_evaluation_count()},
        )
        # Notify convergence that the run completed so it can emit completion if desired
        try:
            self._convergence.on_run_completed(
                best_solution=result.best_solution,
                best_objective=result.best_objective,
                elapsed=elapsed,
                evaluations=problem.get_evaluation_count(),
                success=result.success,
                termination_reason=result.termination_reason,
            )  # type: ignore[attr-defined]
        except AttributeError:
            pass
        return result


__all__ = ["OptimizationEngine"]
