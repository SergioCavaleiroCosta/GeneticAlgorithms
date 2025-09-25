"""Initializer for real-valued vector problems.

If the problem exposes `parameters`, candidates are sampled uniformly in the
normalized [0,1]^d space and only denormalized for objective evaluation. The
stored population solutions remain normalized vectors.

"""
from __future__ import annotations

from typing import List
import numpy as np

from optimization.solution_initializer import SolutionInitializer
from optimization.population import Population
from optimization.update_rule import UpdateRule
from optimization.optimization_problem import OptimizationProblem
from optimization.types import NDArrayFloat


class RealVectorInitializer(SolutionInitializer[NDArrayFloat, float]):
    """Samples the initial population.

    Prefers normalized parameter space when available.
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
        params = getattr(problem, "parameters", None)
        if params is not None:
            dim = len(params)
            def sample_one_norm() -> tuple[NDArrayFloat, float]:
                norm = np.random.uniform(0.0, 1.0, size=(dim,)).astype(np.float64)
                real = np.array([p.normalizer.to_real(norm[i]) for i, p in enumerate(params)], dtype=np.float64)
                obj = problem.evaluate(real)
                return norm, float(obj)

            candidates: List[NDArrayFloat] = []
            objectives: List[float] = []
            for _ in range(self._population_size):
                nvec, obj = sample_one_norm()
                candidates.append(nvec)
                objectives.append(obj)
            population = Population[NDArrayFloat, float](candidates, objectives)
            updater.seed(population.current_solution, population.current_objective)
            return population

        dim = problem.dimension
        if dim is None:
            raise ValueError("Problem must define dimension (or parameters) for initialization")

        candidates: List[NDArrayFloat] = []
        objectives: List[float] = []
        for _ in range(self._population_size):
            s = np.random.random(size=(dim,)).astype(np.float64)
            obj = problem.evaluate(s)
            candidates.append(s)
            objectives.append(obj)

        population = Population[NDArrayFloat, float](candidates, objectives)

        # Seed updater with last element as current (Population convention)
        updater.seed(population.current_solution, population.current_objective)
        return population


__all__ = ["RealVectorInitializer"]