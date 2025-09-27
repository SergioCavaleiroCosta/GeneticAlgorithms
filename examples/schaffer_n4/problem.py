from __future__ import annotations

from typing import Optional
import math

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def schaffer_n4(x: float, y: float) -> float:
    """Schaffer function N.4.

    One common form:
      f(x,y) = 0.5 + (cos^2(sin(|x^2 - y^2|)) - 0.5) / (1 + 0.001 (x^2 + y^2))^2
    Domain: x,y in [-100,100] (using narrower [-50,50] operationally)
    Global minimum at (0,0) with f=0.
    """
    num = math.cos(math.sin(abs(x * x - y * y))) ** 2 - 0.5
    den = (1 + 0.001 * (x * x + y * y)) ** 2
    return 0.5 + num / den


class SchafferN4Problem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Schaffer N.4 test function (2D)."""

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return 2

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        x, y = solution
        value = float(schaffer_n4(float(x), float(y)))
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == 2


__all__ = ["SchafferN4Problem", "schaffer_n4"]
