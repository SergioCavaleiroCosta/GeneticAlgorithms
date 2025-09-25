from __future__ import annotations

from typing import Optional

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def booth(x: float, y: float) -> float:
    """Booth function.

    f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
    Global minimum at (1,3) with f=0. Domain often [-10,10].
    """
    return (x + 2.0 * y - 7.0) ** 2 + (2.0 * x + y - 5.0) ** 2


class BoothProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Booth test function (2D)."""

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return 2

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        x, y = solution
        value = float(booth(float(x), float(y)))
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == 2


__all__ = ["BoothProblem", "booth"]
