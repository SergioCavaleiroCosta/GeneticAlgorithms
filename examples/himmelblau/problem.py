from __future__ import annotations

from typing import Optional

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def himmelblau(x: float, y: float) -> float:
    """Himmelblau's function.

    f(x,y) = (x^2 + y - 11)^2 + (x + y^2 -7)^2
    Domain: x,y in [-5,5]
    Global minima (all f=0) approximately at:
      (3.0, 2.0)
      (-2.805118, 3.131312)
      (-3.779310, -3.283186)
      (3.584428, -1.848126)
    """
    return (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2


class HimmelblauProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Himmelblau test function (2D)."""

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return 2

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        x, y = solution
        value = float(himmelblau(float(x), float(y)))
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == 2


__all__ = ["HimmelblauProblem", "himmelblau"]
