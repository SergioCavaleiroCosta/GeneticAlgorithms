from __future__ import annotations

from typing import Optional
import math

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


def cross_in_tray(x: float, y: float) -> float:
    """Cross-in-tray function.

    f(x,y) = -1e-4 * ( | sin(x) * sin(y) * exp(|100 - sqrt(x^2 + y^2)/π|) | + 1 )^{0.1}
    Domain: x,y in [-10,10]
    Global minima (≈ -2.06261218) at:
      (±1.349406685, ±1.349406685) (four combinations)
    Note: Highly rugged with steep basins.
    """
    r = math.sqrt(x * x + y * y)
    inner = abs(math.sin(x) * math.sin(y) * math.exp(abs(100.0 - r / math.pi))) + 1.0
    return -1e-4 * (inner ** 0.1)


class CrossInTrayProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Cross-in-tray test function (2D)."""

    @property
    def dimension(self) -> Optional[int]:  # type: ignore[override]
        return 2

    def evaluate(self, solution: NDArrayFloat) -> float:  # type: ignore[override]
        x, y = solution
        value = float(cross_in_tray(float(x), float(y)))
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        return len(solution) == 2


__all__ = ["CrossInTrayProblem", "cross_in_tray"]
