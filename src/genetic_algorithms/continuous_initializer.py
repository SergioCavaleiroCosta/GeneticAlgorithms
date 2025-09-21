"""Initializer for real-valued vector problems using bounds/dimension (NumPy)."""
from __future__ import annotations

from typing import List
import numpy as np

from optimization.solution_initializer import SolutionInitializer
from optimization.population import Population
from optimization.update_rule import UpdateRule
from optimization.optimization_problem import OptimizationProblem
from optimization.types import NDArrayFloat


class RealVectorInitializer(SolutionInitializer[NDArrayFloat, float]):
    """Samples the initial population for real vectors using problem.bounds/dimension.

    Fallbacks:
      - If bounds is None but dimension is provided, samples U(-1, 1)
      - If both are None, raises a ValueError
    """

    def __init__(self, population_size: int) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be > 0")
        self._population_size = population_size

    def initialize(
        self,
        problem: OptimizationProblem[NDArrayFloat, float],
        updater: UpdateRule[NDArrayFloat, float],
    ) -> Population[NDArrayFloat, float]:
        bounds = problem.bounds
        dim = problem.dimension if bounds is None else len(bounds)
        if dim is None:
            raise ValueError("Problem must provide bounds or dimension for initialization")

        def sample_one() -> NDArrayFloat:
            if bounds is None:
                return np.random.uniform(-1.0, 1.0, size=(dim,)).astype(np.float64)
            lo = np.array([min(a, b) for a, b in bounds], dtype=np.float64)
            hi = np.array([max(a, b) for a, b in bounds], dtype=np.float64)
            r = np.random.uniform(0.0, 1.0, size=(dim,)).astype(np.float64)
            return (lo + r * (hi - lo)).astype(np.float64)

        candidates: List[NDArrayFloat] = []
        objectives: List[float] = []
        for _ in range(self._population_size):
            s = sample_one()
            obj = problem.evaluate(s)
            problem.increment_evaluation_count()
            candidates.append(s)
            objectives.append(obj)

        population = Population[NDArrayFloat, float](candidates, objectives)

        # Seed updater with last element as current (Population convention)
        updater.seed(population.current_solution, population.current_objective)
        return population


__all__ = ["RealVectorInitializer"]