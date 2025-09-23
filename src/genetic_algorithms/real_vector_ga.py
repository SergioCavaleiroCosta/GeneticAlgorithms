"""Real-valued Genetic Algorithm UpdateRule (NumPy-based)."""
from __future__ import annotations

from typing import Optional, Sequence
import random

from optimization.update_rule import UpdateRule
from optimization.events import EventDispatcher
from optimization.optimization_problem import OptimizationProblem
from optimization.population import Population
from optimization.types import NDArrayFloat
from genetic_algorithms.ops.selection import SelectionStrategy
from genetic_algorithms.ops.selection.tournament import TournamentSelection
from genetic_algorithms.ops.crossover import CrossoverStrategy
from genetic_algorithms.ops.crossover.arithmetic import ArithmeticCrossover
from genetic_algorithms.ops.mutation import MutationStrategy
from genetic_algorithms.ops.elitism import ElitismStrategy
from genetic_algorithms.ops.elitism.topk import TopKElitism
from genetic_algorithms.ops.mutation.gaussian import GaussianMutation


class RealVectorGA(UpdateRule[NDArrayFloat, float]):
    """Genetic Algorithm for real-valued vectors.

    Features:
      - Tournament selection
      - BLX-Alpha style arithmetic crossover (alpha in [0, 1])
      - Gaussian mutation with per-dimension sigma
      - Elitism (copy top-k)
      - Optional bounds clamping using problem.bounds
    """

    def __init__(
        self,
        problem: OptimizationProblem[NDArrayFloat, float],
        *,
        rng: Optional[random.Random] = None,
        # New: pluggable strategies (defaults mirror previous behavior)
        selection: Optional[SelectionStrategy[float]] = None,
        crossover: Optional[CrossoverStrategy[NDArrayFloat]] = None,
        mutation: Optional[MutationStrategy[NDArrayFloat]] = None,
        elitism: Optional[ElitismStrategy[NDArrayFloat, float]] = None,
    ) -> None:
        self._problem = problem
        self._rng = rng or random.Random()
        self._dispatcher: Optional[EventDispatcher[NDArrayFloat, float]] = None

        # Cache bounds if provided for faster clamp
        self._bounds = problem.bounds

        # Strategies (defaults)
        self._selection: SelectionStrategy[float] = selection or TournamentSelection(rng=self._rng)
        self._crossover: CrossoverStrategy[NDArrayFloat] = crossover or ArithmeticCrossover()
        self._mutation: MutationStrategy[NDArrayFloat] = mutation or GaussianMutation(bounds=self._bounds)
        self._elitism: ElitismStrategy[NDArrayFloat, float] = elitism or TopKElitism(k=2)

    # UpdateRule interface
    def set_dispatcher(self, dispatcher: EventDispatcher[NDArrayFloat, float]) -> None:
        self._dispatcher = dispatcher

    @property
    def problem(self) -> OptimizationProblem[NDArrayFloat, float]:
        return self._problem

    def seed(self, initial_solution: NDArrayFloat, initial_objective: float) -> None:  # noqa: ARG002
        # GA does not require special seeding beyond having an initial population
        pass

    # Core GA operators via strategies
    def _select_parent(self, objectives: Sequence[float]) -> int:
        return self._selection.select(objectives)

    def _do_crossover(self, a: NDArrayFloat, b: NDArrayFloat) -> tuple[NDArrayFloat, NDArrayFloat]:
        return self._crossover.crossover(a, b)

    def _mutate(self, x: NDArrayFloat) -> NDArrayFloat:
        return self._mutation.mutate(x)

    def step(self, engine) -> None:  # type: ignore[override]
        population: Population[NDArrayFloat, float] = engine.population
        n = population.size
        if n == 0:
            raise RuntimeError("Population is empty; cannot step GA")

        # Elitism via strategy: indices of elites to carry over
        elites_idx = self._elitism.select_indices(population)
        new_candidates: list[NDArrayFloat] = [population.candidates[i].copy() for i in elites_idx]
        new_objectives: list[float] = [population.objectives[i] for i in elites_idx]

        # Fill the rest with offspring
        target_size = n  # keep population size stable by default
        while len(new_candidates) < target_size:
            i1 = self._select_parent(population.objectives)
            i2 = self._select_parent(population.objectives)
            p1 = population.candidates[i1]
            p2 = population.candidates[i2]
            c1, c2 = self._do_crossover(p1, p2)
            c1 = self._mutate(c1)
            c2 = self._mutate(c2)

            # Evaluate children and append (respect population size)
            for child in (c1, c2):
                if len(new_candidates) >= target_size:
                    break
                obj = self._problem.evaluate(child)
                new_candidates.append(child)
                new_objectives.append(float(obj))

        engine.population.replace_all(new_candidates, new_objectives)


__all__ = ["RealVectorGA"]