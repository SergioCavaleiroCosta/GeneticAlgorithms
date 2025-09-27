"""Schaffer N.1 Multi-Objective Benchmark Problem."""
from __future__ import annotations

import numpy as np
from typing import Sequence

from optimization.types import NDArrayFloat


class SchafferN1Problem:
    """Schaffer N.1 multi-objective benchmark problem.
    
    This is the simplest multi-objective optimization problem with a single variable.
    
    Objectives:
    - f1(x) = x²
    - f2(x) = (x - 2)²
    
    Domain: x ∈ [-A, A] where A is typically 10³
    
    Properties:
    - Single decision variable
    - Convex Pareto front
    - Easy to visualize and understand
    - Global Pareto front: x ∈ [0, 2]
    """
    
    def __init__(self, domain_bound: float = 1000.0):
        """Initialize Schaffer N.1 problem.
        
        Args:
            domain_bound: Domain bound A, so x ∈ [-A, A]
        """
        self.name = "Schaffer_N1"
        self.dimension = 1
        self._num_objectives = 2
        self._evaluation_count = 0
        self.domain_bound = domain_bound
    
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
        # Schaffer N.1 has box constraints: x ∈ [-A, A]
        x = np.asarray(solution)
        return bool(np.all(x >= -self.domain_bound) and np.all(x <= self.domain_bound))
    
    def evaluate(self, solution: NDArrayFloat) -> Sequence[float]:
        """Evaluate Schaffer N.1 objectives.
        
        Args:
            solution: Decision variable x (single value)
            
        Returns:
            Tuple of (f1, f2) objective values
        """
        self._evaluation_count += 1
        
        # Ensure solution is a numpy array and extract single value
        x = float(np.asarray(solution).flatten()[0])
        
        # f1(x) = x²
        f1 = x * x
        
        # f2(x) = (x - 2)²
        f2 = (x - 2.0) * (x - 2.0)
        
        return [f1, f2]
    
    def get_true_pareto_front(self, num_points: int = 100) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Generate the true Pareto front for Schaffer N.1.
        
        The true Pareto optimal solutions are x ∈ [0, 2].
        
        Args:
            num_points: Number of points to generate
            
        Returns:
            Tuple of (f1_values, f2_values) for the true Pareto front
        """
        x_vals = np.linspace(0, 2, num_points)
        f1_vals = x_vals ** 2
        f2_vals = (x_vals - 2) ** 2
        return f1_vals, f2_vals
    
    def get_pareto_optimal_x(self, num_points: int = 100) -> NDArrayFloat:
        """Get the Pareto optimal x values.
        
        Args:
            num_points: Number of points to generate
            
        Returns:
            Array of Pareto optimal x values in [0, 2]
        """
        return np.linspace(0, 2, num_points)