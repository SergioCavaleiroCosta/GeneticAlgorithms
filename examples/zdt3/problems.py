"""ZDT3 Multi-Objective Benchmark Problem."""
from __future__ import annotations

import numpy as np
from typing import Sequence

from optimization.types import NDArrayFloat


class ZDT3Problem:
    """ZDT3 multi-objective benchmark problem.
    
    This is a bi-objective optimization problem with a disconnected Pareto front.
    
    Objectives:
    - f1(x) = x1
    - f2(x) = g(x) * (1 - sqrt(x1/g(x)) - (x1/g(x)) * sin(10*π*x1))
    - g(x) = 1 + 9 * sum(x2...xm) / (m-1)
    
    Domain: x_i ∈ [0, 1] for all i = 1, ..., m
    
    Properties:
    - Disconnected Pareto front with several gaps
    - Tests algorithm's ability to maintain diversity across separate regions
    - 30 variables typically used
    """
    
    def __init__(self, dimension: int = 30):
        """Initialize ZDT3 problem.
        
        Args:
            dimension: Problem dimension (number of variables)
        """
        self.name = "ZDT3"
        self.dimension = dimension
        self._num_objectives = 2
        self._evaluation_count = 0
    
    @property
    def num_objectives(self) -> int:
        """Number of objectives in this problem."""
        return self._num_objectives
    
    @property
    def evaluation_count(self) -> int:
        """Number of objective evaluations performed so far."""
        return self._evaluation_count
    
    @evaluation_count.setter
    def evaluation_count(self, value: int) -> None:
        """Set the number of evaluations (e.g., reset to 0 at run start)."""
        self._evaluation_count = value
    
    def increment_evaluation_count(self) -> None:
        """Increment the evaluation counter by 1."""
        self._evaluation_count += 1
    
    def is_feasible(self, solution: NDArrayFloat) -> bool:
        """Check if a solution satisfies constraints."""
        # ZDT3 has box constraints: x_i ∈ [0, 1]
        x = np.asarray(solution)
        return bool(np.all(x >= 0.0) and np.all(x <= 1.0))
    
    def evaluate(self, solution: NDArrayFloat) -> Sequence[float]:
        """Evaluate ZDT3 objectives.
        
        Args:
            solution: Decision variables in [0, 1]^dimension
            
        Returns:
            Tuple of (f1, f2) objective values
        """
        self._evaluation_count += 1
        
        # Ensure solution is a numpy array
        x = np.asarray(solution)
        
        # f1(x) = x1
        f1 = float(x[0])
        
        # g(x) = 1 + 9 * sum(x2...xm) / (m-1)
        if self.dimension > 1:
            g = 1.0 + 9.0 * float(np.sum(x[1:])) / (self.dimension - 1)
        else:
            g = 1.0
        
        # f2(x) = g(x) * (1 - sqrt(x1/g(x)) - (x1/g(x)) * sin(10*π*x1))
        ratio = f1 / g
        f2 = g * (1.0 - np.sqrt(ratio) - ratio * np.sin(10.0 * np.pi * f1))
        
        return [f1, float(f2)]
    
    def get_true_pareto_front(self, num_points: int = 1000) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Generate the true Pareto front for ZDT3.
        
        Args:
            num_points: Number of points to generate
            
        Returns:
            Tuple of (f1_values, f2_values) for the true Pareto front
        """
        # ZDT3 has disconnected regions with both positive and negative f2 values
        f1 = np.linspace(0, 1, num_points)
        f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
        
        # For ZDT3, the true Pareto front includes negative f2 values
        # The disconnection comes from the sine term, not from filtering f2 >= 0
        # So we return all calculated points
        return f1, f2