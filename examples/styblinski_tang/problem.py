from __future__ import annotations

from typing import Optional
import numpy as np

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def styblinski_tang(values: NDArrayFloat) -> float:
    """Styblinski–Tang function in n dimensions.

    Definition:
        f(x) = 1/2 * sum_{i=1}^n (x_i^4 - 16 x_i^2 + 5 x_i)
    Global minimizer (separable): x_i ≈ -2.903534 for all i
    Per-dimension minimized value (already including 1/2 factor) ≈ -39.166165703771 / 2 ≈ -19.5830828518855
    Thus for n dimensions: f(x*) ≈ -19.5830828518855 * n
    Typical domain: x_i in [-5, 5]
    """
    x = np.asarray(values, dtype=np.float64)
    return float(0.5 * np.sum(x**4 - 16.0 * x**2 + 5.0 * x))


class StyblinskiTangProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Styblinski–Tang test function (moderately multimodal, separable)."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self._dimension = dimension

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return self._dimension

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        value = styblinski_tang(solution)
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == self._dimension


__all__ = ["StyblinskiTangProblem", "styblinski_tang"]
