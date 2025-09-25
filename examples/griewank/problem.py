from __future__ import annotations

from typing import Optional
import math
import numpy as np

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def griewank_vector(x: NDArrayFloat) -> float:
    """Griewank function for a vector x.

    f(x) = 1 + (1/4000) * sum_{i=1}^d x_i^2 - prod_{i=1}^d cos(x_i / sqrt(i))
    Domain: x_i in [-600, 600]
    Global minimum: f(0,...,0)=0
    """
    d = x.shape[0]
    sum_sq = np.sum(np.square(x)) / 4000.0
    prod_cos = 1.0
    for i, val in enumerate(x, start=1):
        prod_cos *= math.cos(val / math.sqrt(i))
    return 1.0 + sum_sq - prod_cos


class GriewankProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Griewank test function (d-D)."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self._dimension = dimension

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return self._dimension

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        value = float(griewank_vector(solution))
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == self._dimension


__all__ = ["GriewankProblem", "griewank_vector"]
