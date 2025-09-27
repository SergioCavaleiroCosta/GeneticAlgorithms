"""Multi-objective test problems for NSGA-II demonstration."""
from __future__ import annotations

from typing import Optional, Sequence
import numpy as np

from optimization.types import NDArrayFloat
from nsga_ii.multi_objective_problem import MultiObjectiveProblem


class ZDT1Problem(MultiObjectiveProblem[NDArrayFloat]):
    """ZDT1 multi-objective test problem.
    
    ZDT1 is a bi-objective minimization problem with:
    - f1(x) = x1
    - f2(x) = g(x) * (1 - sqrt(x1/g(x)))
    - g(x) = 1 + 9 * sum(x2...xm) / (m-1)
    
    where x_i in [0, 1] for i = 1, ..., m
    
    The Pareto front is f2 = 1 - sqrt(f1) for f1 in [0, 1]
    """

    def __init__(self, dimension: int = 30) -> None:
        if dimension < 2:
            raise ValueError("ZDT1 requires at least 2 dimensions")
        self._dimension = dimension
        self._evaluation_count = 0

    @property 
    def num_objectives(self) -> int:
        return 2

    @property
    def dimension(self) -> Optional[int]:
        return self._dimension

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    @evaluation_count.setter
    def evaluation_count(self, value: int) -> None:
        if value < 0:
            raise ValueError("evaluation_count cannot be negative")
        self._evaluation_count = value

    def increment_evaluation_count(self) -> None:
        self._evaluation_count += 1

    def evaluate(self, solution: NDArrayFloat) -> Sequence[float]:
        x = np.asarray(solution, dtype=np.float64)
        if len(x) != self._dimension:
            raise ValueError(f"Solution must have {self._dimension} dimensions")
        
        self.increment_evaluation_count()
        
        # f1(x) = x1
        f1 = float(x[0])
        
        # g(x) = 1 + 9 * sum(x2...xm) / (m-1)
        if self._dimension > 1:
            g = 1.0 + 9.0 * np.sum(x[1:]) / (self._dimension - 1)
        else:
            g = 1.0
        
        # f2(x) = g(x) * (1 - sqrt(x1/g(x)))
        if g > 0 and f1 >= 0:
            f2 = g * (1.0 - np.sqrt(f1 / g))
        else:
            f2 = g  # fallback
        
        return [f1, float(f2)]

    def is_feasible(self, solution: NDArrayFloat) -> bool:
        x = np.asarray(solution, dtype=np.float64)
        return (len(x) == self._dimension and 
                np.all(x >= 0.0) and 
                np.all(x <= 1.0))


class ZDT2Problem(MultiObjectiveProblem[NDArrayFloat]):
    """ZDT2 multi-objective test problem.
    
    ZDT2 is a bi-objective minimization problem with:
    - f1(x) = x1
    - f2(x) = g(x) * (1 - (x1/g(x))^2)
    - g(x) = 1 + 9 * sum(x2...xm) / (m-1)
    
    The Pareto front is f2 = 1 - f1^2 for f1 in [0, 1]
    """

    def __init__(self, dimension: int = 30) -> None:
        if dimension < 2:
            raise ValueError("ZDT2 requires at least 2 dimensions")
        self._dimension = dimension
        self._evaluation_count = 0

    @property 
    def num_objectives(self) -> int:
        return 2

    @property
    def dimension(self) -> Optional[int]:
        return self._dimension

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    @evaluation_count.setter
    def evaluation_count(self, value: int) -> None:
        if value < 0:
            raise ValueError("evaluation_count cannot be negative")
        self._evaluation_count = value

    def increment_evaluation_count(self) -> None:
        self._evaluation_count += 1

    def evaluate(self, solution: NDArrayFloat) -> Sequence[float]:
        x = np.asarray(solution, dtype=np.float64)
        if len(x) != self._dimension:
            raise ValueError(f"Solution must have {self._dimension} dimensions")
        
        self.increment_evaluation_count()
        
        # f1(x) = x1
        f1 = float(x[0])
        
        # g(x) = 1 + 9 * sum(x2...xm) / (m-1)
        if self._dimension > 1:
            g = 1.0 + 9.0 * np.sum(x[1:]) / (self._dimension - 1)
        else:
            g = 1.0
        
        # f2(x) = g(x) * (1 - (x1/g(x))^2)
        if g > 0:
            ratio = f1 / g
            f2 = g * (1.0 - ratio * ratio)
        else:
            f2 = g  # fallback
        
        return [f1, float(f2)]

    def is_feasible(self, solution: NDArrayFloat) -> bool:
        x = np.asarray(solution, dtype=np.float64)
        return (len(x) == self._dimension and 
                np.all(x >= 0.0) and 
                np.all(x <= 1.0))


def true_pareto_front_zdt1(n_points: int = 100) -> np.ndarray:
    """Generate true Pareto front for ZDT1 problem."""
    f1 = np.linspace(0, 1, n_points)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def true_pareto_front_zdt2(n_points: int = 100) -> np.ndarray:
    """Generate true Pareto front for ZDT2 problem."""
    f1 = np.linspace(0, 1, n_points)
    f2 = 1 - f1**2
    return np.column_stack([f1, f2])


__all__ = ["ZDT1Problem", "ZDT2Problem", "true_pareto_front_zdt1", "true_pareto_front_zdt2"]