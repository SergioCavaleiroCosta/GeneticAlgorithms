from __future__ import annotations

from typing import Optional
import numpy as np

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def rastrigin(values: np.ndarray) -> float:
    """Compute the Rastrigin function for a 1D numpy array.

    Standard definition (A=10):
        f(x) = A * n + sum( x_i^2 - A * cos(2*pi*x_i) )
    Global minimum: f(0,...,0) = 0
    Typical domain: x_i in [-5.12, 5.12]
    """
    A = 10.0
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    return float(A * n + np.sum(x * x - A * np.cos(2.0 * np.pi * x)))


class RastriginProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Rastrigin test function in d dimensions (separable, highly multimodal)."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self._dimension = dimension

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return self._dimension

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        value = rastrigin(solution)
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == self._dimension


__all__ = ["RastriginProblem", "rastrigin"]
