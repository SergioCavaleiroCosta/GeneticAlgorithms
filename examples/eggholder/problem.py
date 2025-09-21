from __future__ import annotations

from typing import Optional, List, Tuple
import numpy as np

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat


class EggholderProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Eggholder function in 2D.

    Domain: x, y in [-512, 512]
    Global minimum at (512, 404.2319) with value ~ -959.6407 (depending on convention)
    We’ll use the standard definition from literature.
    """

    def __init__(self) -> None:
        super().__init__()
        self._bounds: List[Tuple[float, float]] = [(-512.0, 512.0), (-512.0, 512.0)]

    @property
    def bounds(self) -> Optional[List[Tuple[float, float]]]:
        return self._bounds

    @property
    def dimension(self) -> Optional[int]:
        return 2

    def evaluate(self, solution: NDArrayFloat) -> float:
        # Ensure shape (2,)
        x = float(solution[0])
        y = float(solution[1])
        # Standard eggholder formula
        term1 = -(y + 47.0) * np.sin(np.sqrt(abs(x/2.0 + (y + 47.0))))
        term2 = -x * np.sin(np.sqrt(abs(x - (y + 47.0))))
        value = term1 + term2
        self.increment_evaluation_count()
        return float(value)

    def is_feasible(self, solution: NDArrayFloat) -> bool:
        # Feasible if within bounds
        (x_min, x_max), (y_min, y_max) = self._bounds
        x = float(solution[0])
        y = float(solution[1])
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)
