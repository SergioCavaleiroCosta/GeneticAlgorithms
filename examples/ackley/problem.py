from __future__ import annotations

from typing import Optional
import numpy as np

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def ackley(values: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2.0 * np.pi) -> float:
    """Compute Ackley function for a 1D numpy array.

    Standard form:
        f(x) = -a * exp(-b * sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(c*x_i))) + a + e
    Global minimum at x=0 with f(0)=0.
    Typical domain: x_i in [-32.768, 32.768]
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n == 0:
        return 0.0
    sq_term = np.sum(x * x)
    cos_term = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sq_term / n))
    term2 = -np.exp(cos_term / n)
    return float(term1 + term2 + a + np.e)


class AckleyProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Ackley test function in d dimensions."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self._dimension = dimension

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return self._dimension

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        value = ackley(solution)
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == self._dimension


__all__ = ["AckleyProblem", "ackley"]
