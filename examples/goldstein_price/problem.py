from __future__ import annotations

from typing import Optional
import numpy as np

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def goldstein_price(x: float, y: float) -> float:
    """Goldstein-Price function.

    f(x,y) = [1 + (x + y + 1)^2 * (19 - 14x + 3x^2 - 14y + 6xy + 3y^2)] *
             [30 + (2x - 3y)^2 * (18 - 32x + 12x^2 + 48y - 36xy + 27y^2)]
    Domain: x,y in [-2, 2]
    Global minimum f(0,-1) = 3
    """
    a = x + y + 1.0
    b = 2.0 * x - 3.0 * y
    term1 = 1.0 + a * a * (19.0 - 14.0 * x + 3.0 * x * x - 14.0 * y + 6.0 * x * y + 3.0 * y * y)
    term2 = 30.0 + b * b * (18.0 - 32.0 * x + 12.0 * x * x + 48.0 * y - 36.0 * x * y + 27.0 * y * y)
    return term1 * term2


class GoldsteinPriceProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Goldstein-Price test function (2D)."""

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return 2

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        x, y = solution
        value = float(goldstein_price(float(x), float(y)))
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == 2


__all__ = ["GoldsteinPriceProblem", "goldstein_price"]
