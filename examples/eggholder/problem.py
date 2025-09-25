from __future__ import annotations

from typing import Optional, Sequence, Any
import numpy as np

from optimization.optimization_problem import BaseOptimizationProblem
from optimization.types import NDArrayFloat
from optimization.parameters import ContinuousParameter, LinearNormalization, ParameterSpec


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
        # Parameter specifications (real domain mapped to normalized [0,1])
        self._parameters: Sequence[ParameterSpec] = [
            ContinuousParameter(
                _name="x",
                _normalizer=LinearNormalization(-512.0, 512.0),
                _description="Eggholder x dimension",
                _unit="units",
            ),
            ContinuousParameter(
                _name="y",
                _normalizer=LinearNormalization(-512.0, 512.0),
                _description="Eggholder y dimension",
                _unit="units",
            ),
        ]

    @property
    def dimension(self) -> Optional[int]:
        return 2

    @property
    def parameters(self) -> Sequence[ParameterSpec]:  # type: ignore[override]
        """Return parameter specifications (real bounds + normalization).

        The optimization loop may keep candidate vectors in normalized [0,1]^d
        space; these specs allow conversion to real domain for evaluation.
        """
        return self._parameters

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
        params = self._parameters
        if len(solution) != len(params):  # dimension mismatch -> infeasible
            return False
        for i, p in enumerate(params):
            norm = p.normalizer
            lo = getattr(norm, "lo", None)
            hi = getattr(norm, "hi", None)
            if lo is None or hi is None:
                # Cannot derive bounds; assume feasible for this dimension
                continue
            v = float(solution[i])
            lo_f = float(lo)
            hi_f = float(hi)
            if lo_f > hi_f:
                lo_f, hi_f = hi_f, lo_f
            if not (lo_f <= v <= hi_f):
                return False
        return True


__all__ = ["EggholderProblem", "eggholder_formula"]
