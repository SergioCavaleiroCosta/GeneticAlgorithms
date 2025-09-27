from __future__ import annotations

from typing import Optional
import math

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def easom(x: float, y: float) -> float:
    """Easom function.

    f(x,y) = -cos(x) * cos(y) * exp(-((x-π)^2 + (y-π)^2))
    Standard domain often quoted: x,y in [-100,100]; we use [-10,10] for visualization.
    Global minimum at (π, π) with f = -1.
    """
    return -math.cos(x) * math.cos(y) * math.exp(-((x - math.pi) ** 2 + (y - math.pi) ** 2))


class EasomProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Easom test function (2D)."""

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return 2

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        x, y = solution
        value = float(easom(float(x), float(y)))
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == 2


__all__ = ["EasomProblem", "easom"]
