from __future__ import annotations

from typing import Optional

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def three_hump_camel(x: float, y: float) -> float:
    """Three-hump camel function.

    f(x,y) = 2x^2 - 1.05x^4 + (x^6)/6 + x y + y^2
    Domain: x,y in [-5,5]
    Global minimum at (0,0) with f=0
    """
    return 2 * x * x - 1.05 * (x ** 4) + (x ** 6) / 6.0 + x * y + y * y


class ThreeHumpCamelProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Three-hump camel test function (2D)."""

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return 2

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        x, y = solution
        value = float(three_hump_camel(float(x), float(y)))
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == 2


__all__ = ["ThreeHumpCamelProblem", "three_hump_camel"]
