"""Real-valued Genetic Algorithm UpdateRule (NumPy-based)."""
from __future__ import annotations

from typing import Optional, Sequence
import random
import numpy as np

from optimization.update_rule import UpdateRule
from optimization.events import EventDispatcher
from optimization.optimization_problem import OptimizationProblem
from optimization.population import Population
from optimization.types import NDArrayFloat


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
        population_size: int,
        tournament_size: int = 2,
        crossover_prob: float = 0.9,
        alpha: float = 0.5,  # crossover mixing parameter
        mutation_prob: float = 0.1,
        mutation_sigma: float = 0.1,
        elitism: int = 1,
        rng: Optional[random.Random] = None,
    ) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be > 0")
        if tournament_size <= 0:
            raise ValueError("tournament_size must be > 0")
        if not (0.0 <= crossover_prob <= 1.0):
            raise ValueError("crossover_prob must be in [0, 1]")
        if not (0.0 <= mutation_prob <= 1.0):
            raise ValueError("mutation_prob must be in [0, 1]")
        if elitism < 0:
            raise ValueError("elitism must be >= 0")

        self._problem = problem
        self._population_size = population_size
        self._k = tournament_size
        self._pc = crossover_prob
        self._alpha = alpha
        self._pm = mutation_prob
        self._sigma = mutation_sigma
        self._elitism = elitism
        self._rng = rng or random.Random()
        self._dispatcher: Optional[EventDispatcher[NDArrayFloat, float]] = None

        # Cache bounds if provided for faster clamp
        self._bounds = problem.bounds

    # UpdateRule interface
    def set_dispatcher(self, dispatcher: EventDispatcher[NDArrayFloat, float]) -> None:
        self._dispatcher = dispatcher

    @property
    def problem(self) -> OptimizationProblem[NDArrayFloat, float]:
        return self._problem

    def seed(self, initial_solution: NDArrayFloat, initial_objective: float) -> None:  # noqa: ARG002
        # GA does not require special seeding beyond having an initial population
        pass

    # Core GA operators
    def _tournament(self, objectives: Sequence[float]) -> int:
        # Return index of the best individual from k sampled indices
        n = len(objectives)
        best_i = None
        best_val = None
        for _ in range(self._k):
            i = self._rng.randrange(n)
            val = objectives[i]
            if best_val is None or val < best_val:
                best_i = i
                best_val = val
        assert best_i is not None
        return best_i

    def _crossover(self, a: NDArrayFloat, b: NDArrayFloat) -> tuple[NDArrayFloat, NDArrayFloat]:
        if self._rng.random() > self._pc:
            return a.copy(), b.copy()
        # BLX-alpha: sample within extended range; simpler arithmetic mix here
        w = self._alpha
        c1 = w * a + (1.0 - w) * b
        c2 = w * b + (1.0 - w) * a
        return c1.astype(np.float64, copy=False), c2.astype(np.float64, copy=False)

    def _mutate(self, x: NDArrayFloat) -> NDArrayFloat:
        if self._rng.random() <= self._pm:
            noise = np.random.normal(loc=0.0, scale=self._sigma, size=x.shape).astype(np.float64)
            x = (x + noise).astype(np.float64)
            # Clamp if bounds are provided
            if self._bounds is not None:
                for i, (lo, hi) in enumerate(self._bounds):
                    lo = float(min(lo, hi))
                    hi = float(max(lo, hi))
                    x[i] = np.clip(x[i], lo, hi)
        return x

    def step(self, engine) -> None:  # type: ignore[override]
        population: Population[NDArrayFloat, float] = engine.population
        n = population.size
        if n == 0:
            raise RuntimeError("Population is empty; cannot step GA")

        # Elitism: keep top-k from current population
        k = min(self._elitism, n) if self._elitism > 0 else 0
        indices = list(range(n))
        indices.sort(key=lambda i: population.objectives[i])
        elites_idx = indices[:k]
        new_candidates: list[NDArrayFloat] = [population.candidates[i].copy() for i in elites_idx]
        new_objectives: list[float] = [population.objectives[i] for i in elites_idx]

        # Fill the rest with offspring
        while len(new_candidates) < self._population_size:
            i1 = self._tournament(population.objectives)
            i2 = self._tournament(population.objectives)
            p1 = population.candidates[i1]
            p2 = population.candidates[i2]
            c1, c2 = self._crossover(p1, p2)
            c1 = self._mutate(c1)
            c2 = self._mutate(c2)

            # Evaluate children and append (respect population size)
            for child in (c1, c2):
                if len(new_candidates) >= self._population_size:
                    break
                obj = self._problem.evaluate(child)
                self._problem.increment_evaluation_count()
                new_candidates.append(child)
                new_objectives.append(float(obj))

        engine.population.replace_all(new_candidates, new_objectives)


__all__ = ["RealVectorGA"]