from __future__ import annotations

from typing import Optional
import numpy as np

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


class SphereProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Sphere (De Jong) test function in d dimensions.

    f(x) = sum_{i=1}^d x_i^2
    Global minimum at x = (0,...,0) with f(x)=0.

    Domain commonly used: x_i in [-5.12, 5.12].
    Dimension fixed externally (we supply parameters via engine state), but
    we expose a dimension property for clarity.
    """

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self._dimension = dimension

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return self._dimension

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        # Sphere objective: sum of squares
        value = float(np.sum(np.square(solution)))
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == self._dimension


__all__ = ["SphereProblem"]
