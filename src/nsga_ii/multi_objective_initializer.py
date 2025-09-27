"""Multi-objective initializer for NSGA-II."""
from __future__ import annotations

from typing import Optional, Sequence, List
import random
import numpy as np

from optimization.solution_initializer import SolutionInitializer
from optimization.population import Population
from optimization.types import NDArrayFloat
from optimization.parameters import ContinuousParameter
from nsga_ii.multi_objective_problem import MultiObjectiveProblem


class MultiObjectiveInitializer(SolutionInitializer[NDArrayFloat, Sequence[float]]):
    """Initializer for multi-objective optimization problems.
    
    Creates a random population in the normalized [0,1]^d space.
    """

    def __init__(
        self,
        population_size: int,
        parameters: List[ContinuousParameter],
        *,
        rng: Optional[random.Random] = None,
    ) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if not parameters:
            raise ValueError("parameters cannot be empty")

        self._population_size = population_size
        self._parameters = parameters
        self._rng = rng or random.Random()

    def initialize(
        self, 
        problem: MultiObjectiveProblem[NDArrayFloat], 
        updater  # type: ignore
    ) -> Population[NDArrayFloat, Sequence[float]]:
        """Initialize a random population for multi-objective optimization."""
        dimension = len(self._parameters)
        
        def sample_one_norm() -> tuple[NDArrayFloat, Sequence[float]]:
            # Sample uniformly in [0, 1]^d
            norm = np.random.rand(dimension).astype(np.float64)
            
            # Convert to real domain and evaluate
            real = np.array([
                param.normalizer.to_real(norm[i]) 
                for i, param in enumerate(self._parameters)
            ], dtype=np.float64)
            
            obj = problem.evaluate(real)
            return norm, obj

        candidates = []
        objectives = []
        
        for _ in range(self._population_size):
            candidate, objective = sample_one_norm()
            candidates.append(candidate)
            objectives.append(objective)

        return Population(candidates, objectives)


__all__ = ["MultiObjectiveInitializer"]