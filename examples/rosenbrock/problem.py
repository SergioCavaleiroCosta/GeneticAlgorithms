from __future__ import annotations

from typing import Optional
import numpy as np

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def rosenbrock(values: NDArrayFloat, a: float = 1.0, b: float = 100.0) -> float:
    """Compute the Rosenbrock function for a vector.

    Standard form (generalized):
        f(x) = sum_{i=1}^{n-1} [ b*(x_{i+1} - x_i^2)^2 + (a - x_i)^2 ]
    Global minimum: f(1,...,1)=0 within typical domain [-5,10] or [-2.048,2.048].
    """
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        return 0.0
    xi = x[:-1]
    xnext = x[1:]
    return float(np.sum(b * (xnext - xi * xi) ** 2 + (a - xi) ** 2))


class RosenbrockProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Rosenbrock test function (banana function) in d dimensions."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self._dimension = dimension

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return self._dimension

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        value = rosenbrock(solution)
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == self._dimension


__all__ = ["RosenbrockProblem", "rosenbrock"]
