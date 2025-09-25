from __future__ import annotations

from typing import Optional
import math

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def bukin_n6(x: float, y: float) -> float:
    """Bukin function N.6.

    f(x,y) = 100 * sqrt(|y - 0.01 x^2|) + 0.01 * |x + 10|
    Domain: x in [-15,-5], y in [-3,3]
    Global minimum at (-10, 1) with f=0.
    """
    return 100.0 * math.sqrt(abs(y - 0.01 * x * x)) + 0.01 * abs(x + 10.0)


class Bukin6Problem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Bukin N.6 test function (2D)."""

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return 2

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        x, y = solution
        value = float(bukin_n6(float(x), float(y)))
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == 2


__all__ = ["Bukin6Problem", "bukin_n6"]
