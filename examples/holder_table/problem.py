from __future__ import annotations

from typing import Optional
import math

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def holder_table(x: float, y: float) -> float:
    """Holder table function.

    f(x,y) = -| sin(x) * cos(y) * exp(|1 - sqrt(x^2 + y^2)/π|) |
    Domain: x,y in [-10,10]
    Global minima ≈ -19.2085 at (±8.05502, ±9.66459) (4 symmetric points)
    """
    r = math.sqrt(x * x + y * y)
    inner = abs(1.0 - r / math.pi)
    return -abs(math.sin(x) * math.cos(y) * math.exp(inner))


class HolderTableProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Holder Table test function (2D)."""

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return 2

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        x, y = solution
        value = float(holder_table(float(x), float(y)))
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == 2


__all__ = ["HolderTableProblem", "holder_table"]
