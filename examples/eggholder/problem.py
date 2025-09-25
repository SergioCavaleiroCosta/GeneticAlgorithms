from __future__ import annotations

from typing import Optional, Any
import numpy as np

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat
# Parameter specifications are now supplied externally via engine state.


def eggholder_formula(x: Any, y: Any) -> Any:
    """Vectorized Eggholder formula.

    Accepts scalars or NumPy arrays for x and y; returns matching shape.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    term1 = -(y_arr + 47.0) * np.sin(np.sqrt(np.abs(x_arr / 2.0 + (y_arr + 47.0))))
    term2 = -x_arr * np.sin(np.sqrt(np.abs(x_arr - (y_arr + 47.0))))
    return term1 + term2


class EggholderProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Eggholder function in 2D.

    Domain: x, y in [-512, 512]
    Global minimum at (512, 404.2319) with value ~ -959.6407 (depending on convention)
    We’ll use the standard definition from literature.
    """

    def __init__(self) -> None:
        super().__init__()
        # Parameters are now expected to be managed externally (engine.state.parameters)
        # Problem keeps only intrinsic dimension knowledge.
        self._parameters: tuple[()] = tuple()

    @property
    def dimension(self) -> Optional[int]:
        return 2

    # parameters property intentionally removed; parameters live in engine.state

    def evaluate(self, solution: NDArrayFloat) -> float:
        # Ensure shape (2,) then evaluate via shared vectorized formula
        x = float(solution[0])
        y = float(solution[1])
        value = float(eggholder_formula(x, y))
        self.increment_evaluation_count()
        return value

    def is_feasible(self, solution: NDArrayFloat) -> bool:
        """Check feasibility using parameter normalizers' real-domain intervals.

        Falls back to True for any parameter whose normalizer does not expose
        ``lo``/``hi`` attributes (keeps behavior permissive for custom strategies).
        """
        # Without internal parameters, assume feasibility by dimension only.
        return len(solution) == 2


__all__ = ["EggholderProblem", "eggholder_formula"]
