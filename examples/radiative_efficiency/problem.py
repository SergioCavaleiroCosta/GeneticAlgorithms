"""Radiative Efficiency Optimization Problem.

This problem optimizes the parameters of a radiative efficiency model
trained using PySR (Symbolic Regression). The model predicts radiative
efficiency based on 7 input parameters from fluidized bed combustion.

The objective is to MAXIMIZE radiative efficiency by finding optimal
parameter combinations, but since the GA minimizes, we negate the output.

IMPORTANT: Radiative efficiency is only computed when the flame classifier
(alpha) is above 0.5. If alpha ≤ 0.5, no flame exists and efficiency is 0.

Model Parameters (7 inputs):
- phi: Equivalence ratio (razão de equivalência)
- u_avg: Average inlet velocity [m/s] (velocidade média de entrada)
- ar: Aspect ratio (razão de aspecto)
- lt_0: First layer thickness [m] (espessura da primeira camada)
- lt_1: Second layer thickness [m] (espessura da segunda camada)
- eps_1: Emissivity of second layer (emissividade da segunda camada)
- a_1: Second layer extinction coefficient [1/m] (coef. extinção 2ª camada)

Two PySR models are used and selected by their target labels in the
exported metadata: one for ``alpha`` (flame classifier) and one for
``radiative_efficiency``.
"""
from __future__ import annotations

import numpy as np
import sys
import os
from typing import Any, Dict, cast
from optimization.types import NDArrayFloat
from optimization.optimization_problem import BaseOptimizationProblem

# Add exported_models to path to import the equations metadata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'exported_models', 'equations_only'))

# We select equations by target labels and evaluate them locally with a safe namespace
from equations import MODEL_INFO  # type: ignore

def _safe_eval(eq: str, variables: Dict[str, float]) -> float:
    # Minimal sanitizer for PySR expressions
    eq2 = eq.replace('^', '**')
    ns: dict[str, Any] = {
        'np': np,
        'sqrt': np.sqrt,
        'exp': np.exp,
        'log': np.log,
        'abs': np.abs,
        'tanh': np.tanh,
        **variables,
    }
    # Allow __import__ to avoid KeyError in some multiprocessing contexts
    return float(eval(eq2, {'__builtins__': {'__import__': __import__}}, ns))  # type: ignore[arg-type]

_alpha_eq: str | None = None
_eta_eq: str | None = None
_mi: dict[str, dict[str, Any]] = cast(dict[str, dict[str, Any]], MODEL_INFO)
for _key, info in _mi.items():
    target = str(info.get('target_variable', '')).lower()
    if target == 'alpha' and _alpha_eq is None:
        _alpha_eq = str(info.get('equation', ''))
    if target == 'radiative_efficiency' and _eta_eq is None:
        _eta_eq = str(info.get('equation', ''))

if not _alpha_eq or not _eta_eq:
    raise RuntimeError('Alpha and radiative_efficiency equations not found in MODEL_INFO')

def predict_alpha_BXNBv9(variables: Dict[str, float]) -> float:
    return _safe_eval(_alpha_eq, variables)  # type: ignore[arg-type]

def predict_radiative_efficiency_k1g9rP(variables: Dict[str, float]) -> float:
    return _safe_eval(_eta_eq, variables)  # type: ignore[arg-type]


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
        
        # Updated bounds for current models (union of required variables)
        self.default_bounds = {
            'phi': (0.3, 0.9),            # Equivalence ratio
            'u_avg': (0.5, 2.4),          # Average inlet velocity [m/s]
            'ar': (1.1, 5.0),             # Aspect ratio
            'lt_0': (0.005, 0.015),       # First layer thickness [m]
            'lt_1': (0.03, 0.08),         # Second layer thickness [m]
            'eps_1': (0.4, 0.95),         # Emissivity of second layer
            'a_1': (400.0, 800.0),        # Second layer extinction coefficient [1/m]
        }

        self.bounds = bounds if bounds is not None else self.default_bounds
        self.param_names = ['phi', 'u_avg', 'ar', 'lt_0', 'lt_1', 'eps_1', 'a_1']
        
    def is_feasible(self, solution: NDArrayFloat) -> bool:  # type: ignore[override]
        """Check if solution is feasible.
        
        Args:
            solution: Candidate solution
            
        Returns:
            True if solution has correct dimensionality
        """
        flat = solution.ravel()
        if len(flat) != len(self.param_names):
            return False
        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.bounds[param_name]
            val = flat[i]
            if val < min_val or val > max_val:
                return False
        return True
        
    def evaluate(self, solution: NDArrayFloat) -> float:
        """Evaluate the radiative efficiency model.
        
        This method:
        1. First computes alpha (flame classifier)
        2. If alpha > 0.5, computes radiative efficiency
        3. If alpha ≤ 0.5, returns 0 (no flame condition)
        
        Args:
            solution: Array of 7 parameters [phi, u_avg, ar, lt_0, lt_1, eps_1, a_1]
        
        Returns:
            Negative radiative efficiency (to minimize for maximization)
            Plus penalty for constraint violations
        """
        self.increment_evaluation_count()
        
        # Extract parameters
        phi, u_avg, ar, lt_0, lt_1, eps_1, a_1 = solution.ravel()[:7]
        
        # Check feasibility (dimensionality and bounds)
        if not self.is_feasible(solution):
            return 0.0
        try:
            # Create variables dictionary for PySR models
            variables = {
                'phi': phi,
                'u_avg': u_avg,
                'ar': ar,
                'lt_0': lt_0,
                'lt_1': lt_1,
                'eps_1': eps_1,
                'a_1': a_1,
            }
            # ==========================================
            # STEP 1: Compute Alpha (Flame Classifier)
            # Model: 20251010_155720_BXNBv9
            # ==========================================
            alpha = predict_alpha_BXNBv9(variables)
            # Check if alpha is valid
            if not np.isfinite(alpha):
                return self.penalty_weight * 10.0
            # ==========================================
            # STEP 2: Check flame condition (alpha > 0.5)
            # ==========================================
            if alpha <= 0.5:
                # No flame condition - radiative efficiency is 0
                # Return large positive value (since we're minimizing negative efficiency)
                # This effectively makes this solution very poor for maximization
                return 0.0
            # ==========================================
            # STEP 3: Compute Radiative Efficiency
            # Model: 20251010_162508_k1g9rP
            # Only executed if alpha > 0.5
            # ==========================================
            radiative_efficiency = predict_radiative_efficiency_k1g9rP(variables)
            # Check for NaN or Inf
            if not np.isfinite(radiative_efficiency):
                return self.penalty_weight * 10.0
            # Return negative (since we minimize, but want to maximize efficiency)
            return -float(radiative_efficiency)
        except (ZeroDivisionError, FloatingPointError, OverflowError):
            # Numerical error - return large penalty
            return self.penalty_weight * 10.0
    
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
            7 (number of parameters)
        """
        return len(self.param_names)
    
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
            "Radiative Efficiency Optimization (PySR Models)\n"
            f"Dimensions: {self.get_dimension()}\n"
            "Objective: Maximize radiative efficiency (alpha > 0.5 gate)\n"
            f"Parameters:\n{bounds_str}"
        )
