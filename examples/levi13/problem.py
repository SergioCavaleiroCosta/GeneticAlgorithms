from __future__ import annotations

from typing import Optional
import math

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def levi_n13(x: float, y: float) -> float:
    """Levi function N.13

    f(x,y) = sin^2(3πx) + (x-1)^2 * (1 + sin^2(3πy)) + (y-1)^2 * (1 + sin^2(2πy))
    Domain: x,y in [-10, 10]
    Global minimum: f(1,1)=0
    """
    return math.sin(3 * math.pi * x) ** 2 + (x - 1) ** 2 * (1 + math.sin(3 * math.pi * y) ** 2) + (y - 1) ** 2 * (
        1 + math.sin(2 * math.pi * y) ** 2
    )


class Levi13Problem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Levi N.13 test function (2D)."""

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return 2

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        x, y = solution
        value = float(levi_n13(float(x), float(y)))
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == 2


__all__ = ["Levi13Problem", "levi_n13"]
