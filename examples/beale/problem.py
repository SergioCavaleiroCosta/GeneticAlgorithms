from __future__ import annotations

from typing import Optional
import numpy as np

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def beale(x: float, y: float) -> float:
    """Beale function (2D).

    f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
    Domain usually taken as x,y in [-4.5, 4.5]
    Global minimum at (3, 0.5) with f=0.
    """
    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y * y) ** 2
        + (2.625 - x + x * y * y * y) ** 2
    )


class BealeProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Beale test function (2D)."""

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return 2

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        x, y = solution
        value = float(beale(float(x), float(y)))
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == 2


__all__ = ["BealeProblem", "beale"]
