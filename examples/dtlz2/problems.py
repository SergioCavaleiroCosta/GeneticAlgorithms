"""DTLZ2 Multi-Objective Benchmark Problem."""
from __future__ import annotations

import numpy as np
from typing import Sequence

from optimization.types import NDArrayFloat


class DTLZ2Problem:
    """DTLZ2 multi-objective benchmark problem.
    
    This is a scalable multi-objective optimization problem with a spherical Pareto front.
    
    Objectives (for M objectives):
    - f1(x) = (1 + g(x_M)) * cos(x1*π/2) * cos(x2*π/2) * ... * cos(x_{M-1}*π/2)  
    - f2(x) = (1 + g(x_M)) * cos(x1*π/2) * cos(x2*π/2) * ... * sin(x_{M-1}*π/2)
    - ...
    - fM(x) = (1 + g(x_M)) * sin(x1*π/2)
    - g(x_M) = sum((xi - 0.5)^2) for i = M to n
    
    Domain: x_i ∈ [0, 1] for all i = 1, ..., n
    
    Properties:
    - Spherical Pareto front
    - Scalable to any number of objectives M
    - Well-suited for testing algorithm performance on many objectives
    - Recommended: n = M + k - 1 variables (k=10 typically)
    """
    
    def __init__(self, num_objectives: int = 2, dimension: int = 12):
        """Initialize DTLZ2 problem.
        
        Args:
            num_objectives: Number of objectives (M)
            dimension: Problem dimension (number of variables n)
                      Recommended: n = M + 10 
        """
        self.name = "DTLZ2"
        self.dimension = dimension
        self._num_objectives = num_objectives
        self._evaluation_count = 0
        
        # Validate dimensions
        if dimension < num_objectives:
            raise ValueError(f"Dimension ({dimension}) must be >= num_objectives ({num_objectives})")
    
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
        # DTLZ2 has box constraints: x_i ∈ [0, 1]
        x = np.asarray(solution)
        return bool(np.all(x >= 0.0) and np.all(x <= 1.0))
    
    def evaluate(self, solution: NDArrayFloat) -> Sequence[float]:
        """Evaluate DTLZ2 objectives.
        
        Args:
            solution: Decision variables in [0, 1]^dimension
            
        Returns:
            List of M objective values
        """
        self._evaluation_count += 1
        
        # Ensure solution is a numpy array
        x = np.asarray(solution)
        M = self._num_objectives
        
        # g(x_M) = sum((xi - 0.5)^2) for i = M to n
        if self.dimension > M:
            g = np.sum((x[M-1:] - 0.5) ** 2)
        else:
            g = 0.0
        
        # Calculate objectives
        objectives = []
        
        for i in range(M):
            # Common factor: (1 + g(x_M))
            f_i = 1.0 + g
            
            # Product of cosines for dimensions before current objective
            for j in range(M - 1 - i):
                f_i *= np.cos(x[j] * np.pi / 2.0)
            
            # Sine term for current dimension (if not the last objective)
            if i < M - 1:
                f_i *= np.sin(x[M - 1 - i] * np.pi / 2.0)
            
            objectives.append(float(f_i))
        
        return objectives
    
    def get_true_pareto_front(self, num_points: int = 100) -> NDArrayFloat:
        """Generate points on the true Pareto front for DTLZ2.
        
        For 2 objectives, this is a quarter circle.
        For 3+ objectives, this becomes a hyper-sphere.
        
        Args:
            num_points: Number of points to generate
            
        Returns:
            Array of shape (num_points, num_objectives) with Pareto front points
        """
        if self._num_objectives == 2:
            # For 2D: quarter circle from (1,0) to (0,1)
            theta = np.linspace(0, np.pi/2, num_points)
            f1 = np.cos(theta)
            f2 = np.sin(theta)
            return np.column_stack([f1, f2])
        
        elif self._num_objectives == 3:
            # For 3D: eighth of a sphere
            # Use spherical coordinates
            n_theta = int(np.sqrt(num_points))
            n_phi = int(np.sqrt(num_points))
            
            theta = np.linspace(0, np.pi/2, n_theta)  # 0 to π/2
            phi = np.linspace(0, np.pi/2, n_phi)      # 0 to π/2
            
            points = []
            for t in theta:
                for p in phi:
                    f1 = np.cos(p) * np.cos(t)  # x
                    f2 = np.cos(p) * np.sin(t)  # y  
                    f3 = np.sin(p)              # z
                    points.append([f1, f2, f3])
            
            return np.array(points)
        
        else:
            # For higher dimensions, generate random points on unit hyper-sphere
            # in the positive orthant
            points = np.random.randn(num_points, self._num_objectives)
            points = np.abs(points)  # Ensure positive orthant
            
            # Normalize to unit hyper-sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
            return points