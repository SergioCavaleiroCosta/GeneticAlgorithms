from __future__ import annotations

from typing import Optional
import math

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def schaffer_n2(x: float, y: float) -> float:
    r"""Schaffer function N.2.

    f(x,y) = 0.5 + ( sin^2(x^2 - y^2) - 0.5 ) / (1 + 0.001 (x^2 + y^2))^2
    Domain: x,y in [-100,100] (using narrower [-50,50] for practicality)
    Global minimum at (0,0) with f=0.
    """
    num = math.sin(x * x - y * y) ** 2 - 0.5
    den = (1 + 0.001 * (x * x + y * y)) ** 2
    return 0.5 + num / den


class SchafferN2Problem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Schaffer N.2 test function (2D)."""

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return 2

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        x, y = solution
        value = float(schaffer_n2(float(x), float(y)))
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == 2


__all__ = ["SchafferN2Problem", "schaffer_n2"]
