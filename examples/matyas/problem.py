from __future__ import annotations

from typing import Optional

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def matyas(x: float, y: float) -> float:
    """Matyas function.

    f(x,y) = 0.26 (x^2 + y^2) - 0.48 x y
    Domain: x,y in [-10,10]
    Global minimum at (0,0) with f=0.
    """
    return 0.26 * (x * x + y * y) - 0.48 * x * y


class MatyasProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Matyas test function (2D)."""

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return 2

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        x, y = solution
        value = float(matyas(float(x), float(y)))
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == 2


__all__ = ["MatyasProblem", "matyas"]
