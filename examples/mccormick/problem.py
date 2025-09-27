from __future__ import annotations

from typing import Optional
import math

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def mccormick(x: float, y: float) -> float:
    """McCormick function.

    f(x,y) = sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1
    Domain: x in [-1.5, 4], y in [-3, 4]
    Global minimum ≈ -1.913222 at (x,y) ≈ (-0.54719, -1.54719)
    """
    return math.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1.0


class McCormickProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """McCormick test function (2D)."""

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return 2

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        x, y = solution
        value = float(mccormick(float(x), float(y)))
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == 2


__all__ = ["McCormickProblem", "mccormick"]
