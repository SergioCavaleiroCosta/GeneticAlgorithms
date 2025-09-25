"""Initializer for real-valued vector problems.

If the problem exposes `parameters`, candidates are sampled uniformly in the
normalized [0,1]^d space and only denormalized for objective evaluation. The
stored population solutions remain normalized vectors.

"""
from __future__ import annotations

from typing import List, Sequence
from optimization.parameters.protocols import ParameterSpec
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

    def __init__(self, population_size: int, parameters: Sequence[ParameterSpec]) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be > 0")
        self._population_size = population_size
        self._parameters: list[ParameterSpec] = list(parameters)

    def initialize(
        self,
        problem: OptimizationProblem[NDArrayFloat, float],
        updater: UpdateRule[NDArrayFloat, float],
    ) -> Population[NDArrayFloat, float]:
        params = self._parameters
        if not params:
            raise ValueError("parameters required for RealVectorInitializer")
        dim = len(params)

        def sample_one_norm() -> tuple[NDArrayFloat, float]:
            norm = np.random.random((dim,)).astype(np.float64)
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

    # Candidates stored normalized; objectives computed on real-domain values.


__all__ = ["RealVectorInitializer"]