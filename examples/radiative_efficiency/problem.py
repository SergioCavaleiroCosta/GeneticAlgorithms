"""Radiative Efficiency Optimization Problem.

This problem optimizes the parameters of a radiative efficiency model
trained using PySR (Symbolic Regression). The model predicts radiative
efficiency based on 5 input parameters from fluidized bed combustion.

The objective is to MAXIMIZE radiative efficiency by finding optimal
parameter combinations, but since the GA minimizes, we negate the output.

IMPORTANT: Radiative efficiency is only computed when the flame classifier
(alpha) is above 0.5. If alpha ≤ 0.5, no flame exists and efficiency is 0.

Model Parameters (5 inputs):
- phi: Equivalence ratio (razão de equivalência)
- u_avg: Average inlet velocity [m/s] (velocidade média de entrada)
- ar: Aspect ratio (razão de aspecto)
- lt_1: Second layer thickness [m] (espessura da segunda camada)
- a_0: First layer extinction coefficient [1/m] (coeficiente de extinção da primeira camada)

Two PySR models are used:

1. Flame Classifier (Model ID: 20251010_155720_BXNBv9) - computes alpha:
   abs(abs(abs(1.8626698 - abs(abs(3.0440187 - abs((phi + ((sqrt(0.43977848 / u_avg) + 
   -1.2142289) / ar)) / -0.18428192)) + -0.4066447)) + -0.37922555) - 0.18653812) + 
   -0.093743525

2. Radiative Efficiency (Model ID: 20251010_162508_k1g9rP) - if alpha > 0.5:
   (exp(((-0.21451901 - sqrt(u_avg / ar)) / (phi * 0.7542987)) - 
   ((phi * ((phi + -0.6582989) / u_avg)) * 1.4569359)) * 1.2530051) + 
   (lt_1 * (a_0 * 0.0002242238))
"""
from __future__ import annotations

import numpy as np
import sys
import os
from typing import Dict
from optimization.types import NDArrayFloat
from optimization.optimization_problem import BaseOptimizationProblem

# Add exported_models to path to import the equations
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'exported_models', 'equations_only'))

try:
    from equations import predict_alpha_BXNBv9, predict_radiative_efficiency_k1g9rP
except ImportError:
    # Fallback: define the functions directly if import fails
    def predict_alpha_BXNBv9(variables: Dict[str, float]) -> float:
        """Flame classifier equation from PySR model 20251010_155720_BXNBv9"""
        phi, u_avg, ar = variables['phi'], variables['u_avg'], variables['ar'] 
        eq = "abs(abs(abs(1.8626698 - abs(abs(3.0440187 - abs((phi + ((sqrt(0.43977848 / u_avg) + -1.2142289) / ar)) / -0.18428192)) + -0.4066447)) + -0.37922555) - 0.18653812) + -0.093743525"
        eq = eq.replace("sqrt", "np.sqrt").replace("abs", "np.abs")
        namespace = {"np": np, "phi": phi, "u_avg": u_avg, "ar": ar}
        return eval(eq, {"__builtins__": {}}, namespace)
    
    def predict_radiative_efficiency_k1g9rP(variables: Dict[str, float]) -> float:
        """Radiative efficiency equation from PySR model 20251010_162508_k1g9rP"""
        phi, u_avg, ar, lt_1, a_0 = variables['phi'], variables['u_avg'], variables['ar'], variables['lt_1'], variables['a_0']
        eq = "(exp(((-0.21451901 - sqrt(u_avg / ar)) / (phi * 0.7542987)) - ((phi * ((phi + -0.6582989) / u_avg)) * 1.4569359)) * 1.2530051) + (lt_1 * (a_0 * 0.0002242238))"
        eq = eq.replace("sqrt", "np.sqrt").replace("exp", "np.exp")
        namespace = {"np": np, "phi": phi, "u_avg": u_avg, "ar": ar, "lt_1": lt_1, "a_0": a_0}
        return eval(eq, {"__builtins__": {}}, namespace)


class RadiativeEfficiencyProblem(BaseOptimizationProblem[NDArrayFloat, float]):
    """Radiative efficiency optimization using trained PySR model."""
    """Radiative efficiency optimization using trained PySR model.
    
    The model is a symbolic regression equation that predicts radiative
    efficiency from 7 process parameters. We minimize the negative of
    the predicted efficiency to maximize actual efficiency.
    
    Args:
        bounds: Dictionary mapping parameter names to (min, max) tuples
        penalty_weight: Weight for constraint violations (default: 1e6)
    """
    
    def __init__(
        self,
        bounds: dict[str, tuple[float, float]] | None = None,
        penalty_weight: float = 1e6
    ):
        """Initialize the radiative efficiency problem.
        
        Args:
            bounds: Parameter bounds as {param_name: (min, max)}
                   If None, uses default bounds from training data ranges
            penalty_weight: Penalty for constraint violations
        """
        super().__init__()
        self.penalty_weight = penalty_weight
        
        # Default bounds based on training data ranges
        # These correspond to the actual ranges used during PySR model training
        self.default_bounds = {
            'phi': (0.3, 0.9),           # Equivalence ratio
            'u_avg': (0.5, 2.4),         # Average inlet velocity [m/s]
            'ar': (1.1, 5.0),            # Aspect ratio
            'lt_1': (0.005, 0.015),      # Second layer thickness [m]
            'a_0': (1000.0, 1300.0),     # First layer extinction coefficient [1/m]
        }
        
        self.bounds = bounds if bounds is not None else self.default_bounds
        self.param_names = ['phi', 'u_avg', 'ar', 'lt_1', 'a_0']
        
    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        """Check if solution is feasible.
        
        Args:
            solution: Candidate solution
            
        Returns:
            True if solution has correct dimensionality
        """
        return len(solution.ravel()) == 5
        
    def evaluate(self, solution: NDArrayFloat) -> float:
        """Evaluate the radiative efficiency model.
        
        This method:
        1. First computes alpha (flame classifier)
        2. If alpha > 0.5, computes radiative efficiency
        3. If alpha ≤ 0.5, returns 0 (no flame condition)
        
        Args:
            solution: Array of 5 parameters [phi, u_avg, ar, lt_1, a_0]
        
        Returns:
            Negative radiative efficiency (to minimize for maximization)
            Plus penalty for constraint violations
        """
        self.increment_evaluation_count()
        
        # Extract parameters
        phi, u_avg, ar, lt_1, a_0 = solution.ravel()[:5]
        
        # Check bounds and apply penalties
        penalty = 0.0
        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.bounds[param_name]
            val = solution.ravel()[i]
            if val < min_val:
                penalty += self.penalty_weight * (min_val - val) ** 2
            elif val > max_val:
                penalty += self.penalty_weight * (val - max_val) ** 2
        
        try:
            # Create variables dictionary for PySR models
            variables = {
                'phi': phi,
                'u_avg': u_avg,
                'ar': ar,
                'lt_1': lt_1,
                'a_0': a_0
            }
            
            # ==========================================
            # STEP 1: Compute Alpha (Flame Classifier)
            # Model: 20251010_155720_BXNBv9
            # ==========================================
            alpha = predict_alpha_BXNBv9(variables)
            
            # Check if alpha is valid
            if not np.isfinite(alpha):
                return self.penalty_weight * 10.0 + penalty
            
            # ==========================================
            # STEP 2: Check flame condition (alpha > 0.5)
            # ==========================================
            if alpha <= 0.5:
                # No flame condition - radiative efficiency is 0
                # Return large positive value (since we're minimizing negative efficiency)
                # This effectively makes this solution very poor for maximization
                return 0.0 + penalty
            
            # ==========================================
            # STEP 3: Compute Radiative Efficiency
            # Model: 20251010_162508_k1g9rP
            # Only executed if alpha > 0.5
            # ==========================================
            radiative_efficiency = predict_radiative_efficiency_k1g9rP(variables)
            
            # Check for NaN or Inf
            if not np.isfinite(radiative_efficiency):
                return self.penalty_weight * 10.0 + penalty
            
            # Return negative (since we minimize, but want to maximize efficiency)
            return -float(radiative_efficiency) + penalty
            
        except (ZeroDivisionError, FloatingPointError, OverflowError):
            # Numerical error - return large penalty
            return self.penalty_weight * 10.0 + penalty
    
    def get_bounds(self) -> list[tuple[float, float]]:
        """Get parameter bounds in order.
        
        Returns:
            List of (min, max) tuples for each parameter
        """
        return [self.bounds[name] for name in self.param_names]
    
    def get_optimal_value(self) -> float | None:
        """Get known optimal value if available.
        
        Returns:
            None (unknown optimal for this real-world problem)
        """
        return None
    
    def get_dimension(self) -> int:
        """Get problem dimensionality.
        
        Returns:
            5 (number of parameters)
        """
        return 5
    
    def describe(self) -> str:
        """Get problem description.
        
        Returns:
            Human-readable problem description
        """
        bounds_str = "\n".join(
            f"  {name}: [{self.bounds[name][0]:.4f}, {self.bounds[name][1]:.4f}]"
            for name in self.param_names
        )
        return (
            f"Radiative Efficiency Optimization (PySR Model 20251010_162508_k1g9rP)\n"
            f"Dimensions: {self.get_dimension()}\n"
            f"Objective: Maximize radiative efficiency\n"
            f"Parameters:\n{bounds_str}\n"
            f"Model: Symbolic regression equation with complexity 29"
        )
