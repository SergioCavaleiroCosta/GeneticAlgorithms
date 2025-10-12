"""Multi-objective optimization problem: Maximize radiation, minimize pressure drop.

This problem uses the symbolic regression models to optimize two competing objectives:
1. Maximize radiative efficiency (η)
2. Minimize pressure drop (ΔP)

Both objectives are evaluated under flame conditions (α > 0.5).
"""
from __future__ import annotations
import sys
from pathlib import Path

# Add exported_models to path
exported_models_path = Path(__file__).parents[2] / "exported_models" / "equations_only"
sys.path.insert(0, str(exported_models_path))

from equations import (
    predict_alpha_v2sJPa,
    predict_radiative_efficiency_z5mtkr,
    predict_pressure_drop_5J9e4I,
)

import numpy as np
from numpy.typing import NDArray

RealVector = NDArray[np.floating]


class RadiationPressureMOOP:
    """Multi-objective optimization: maximize radiation, minimize pressure drop.
    
    Decision Variables (7D):
    - phi: equivalence ratio [0.3, 0.9]
    - u_avg: average velocity [0.5, 2.4] m/s
    - ar: aspect ratio [1.1, 5.0]
    - lt_0: lower tube parameter 0 [0.005, 0.015] m
    - lt_1: lower tube parameter 1 [0.03, 0.08] m
    - eps_1: emissivity parameter 1 [0.4, 0.95]
    - a_1: absorptivity parameter 1 [400.0, 800.0] 1/m
    
    Objectives:
    - f1: Maximize radiative efficiency η (converted to minimization: -η)
          Note: η = 0 when α ≤ 0.5 (no flame)
    - f2: Minimize pressure drop ΔP
    
    Constraints:
    - Box constraints on all variables (out of bounds → penalized)
    
    Note: The flame condition α > 0.5 is NOT a constraint.
          When α ≤ 0.5, the solution is still feasible but has η = 0.
    """
    
    def __init__(self):
        self._evaluation_count = 0
        self._num_objectives = 2
        
        # Parameter bounds (same as radiative_efficiency single-objective)
        self.bounds = {
            'phi': (0.3, 0.9),            # Equivalence ratio
            'u_avg': (0.5, 2.4),          # Average velocity [m/s]
            'ar': (1.1, 5.0),             # Aspect ratio
            'lt_0': (0.005, 0.015),       # Lower tube parameter 0 [m]
            'lt_1': (0.03, 0.08),         # Lower tube parameter 1 [m]
            'eps_1': (0.4, 0.95),         # Emissivity parameter 1
            'a_1': (400.0, 800.0),        # Absorptivity parameter 1 [1/m]
        }
        
        # Fixed parameters (not optimized)
        self.fixed_params = {
            'eps_0': 0.8,  # Fixed emissivity parameter 0
            'k_0': 0.1,    # Fixed absorption coefficient
        }
        
    @property
    def num_objectives(self) -> int:
        return self._num_objectives
    
    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count
    
    @evaluation_count.setter
    def evaluation_count(self, value: int) -> None:
        self._evaluation_count = value
        
    def increment_evaluation_count(self) -> None:
        self._evaluation_count += 1
    
    def _create_variable_dict(self, x: RealVector) -> dict[str, float]:
        """Convert solution vector to variable dictionary."""
        param_names = list(self.bounds.keys())
        variables = {name: float(x[i]) for i, name in enumerate(param_names)}
        variables.update(self.fixed_params)
        return variables
    
    def evaluate(self, x: RealVector) -> list[float]:
        """Evaluate both objectives: [-η, ΔP].
        
        Args:
            x: Solution vector [phi, u_avg, ar, lt_0, lt_1, eps_1, a_1]
            
        Returns:
            List of two objectives: [-radiative_efficiency, pressure_drop]
            
        Note:
            - If parameters are out of bounds (infeasible): return [0.0, 1e10]
            - If α ≤ 0.5 (no flame): return [0.0, pressure_drop_calculated]
            - If α > 0.5 (flame exists): return [-η, pressure_drop_calculated]
        """
        self.increment_evaluation_count()
        
        # Check box constraints (out of bounds = infeasible)
        if not self.is_feasible(x):
            # Penalize solutions outside bounds
            return [0.0, 1e10]  # Zero radiation, huge pressure drop penalty
        
        variables = self._create_variable_dict(x)
        
        try:
            # Check flame condition
            alpha = predict_alpha_v2sJPa(variables)
            
            # Calculate pressure drop (always calculated)
            pressure_drop = predict_pressure_drop_5J9e4I(variables)
            f2 = float(pressure_drop)
            
            # Radiative efficiency depends on flame condition
            if alpha > 0.5:
                # Flame exists: calculate radiative efficiency
                eta = predict_radiative_efficiency_z5mtkr(variables)
                f1 = -float(eta)  # Negate because we want to maximize
            else:
                # No flame: zero radiative efficiency
                f1 = 0.0
            
            return [f1, f2]
            
        except Exception as e:
            print(f"Error evaluating solution: {e}")
            return [0.0, 1e10]
    
    def is_feasible(self, x: RealVector) -> bool:
        """Check if solution satisfies box constraints.
        
        Only checks if variables are within their bounds.
        The flame condition (α > 0.5) is NOT a feasibility constraint,
        it only affects the radiative efficiency (zero if no flame).
        """
        param_names = list(self.bounds.keys())
        
        # Check box constraints
        for i, name in enumerate(param_names):
            lb, ub = self.bounds[name]
            if x[i] < lb or x[i] > ub:
                return False
        
        return True
    
    def get_bounds(self) -> tuple[list[float], list[float]]:
        """Get lower and upper bounds for all variables."""
        lower = [bounds[0] for bounds in self.bounds.values()]
        upper = [bounds[1] for bounds in self.bounds.values()]
        return lower, upper


def create_problem() -> RadiationPressureMOOP:
    """Factory function to create the problem instance."""
    return RadiationPressureMOOP()


if __name__ == "__main__":
    # Test the problem
    problem = create_problem()
    
    print("Multi-Objective Optimization Problem:")
    print("=" * 60)
    print(f"Number of objectives: {problem.num_objectives}")
    print(f"Number of variables: {len(problem.bounds)}")
    print("\nVariable bounds:")
    for name, (lb, ub) in problem.bounds.items():
        print(f"  {name:8s}: [{lb:6.3f}, {ub:6.3f}]")
    
    print("\nFixed parameters:")
    for name, value in problem.fixed_params.items():
        print(f"  {name:8s}: {value:6.3f}")
    
    # Test with a random solution
    print("\n" + "=" * 60)
    print("Testing with random solution...")
    np.random.seed(42)
    lower, upper = problem.get_bounds()
    x_test = np.random.uniform(lower, upper)
    
    print(f"\nTest solution:")
    param_names = list(problem.bounds.keys())
    for i, name in enumerate(param_names):
        print(f"  {name:8s} = {x_test[i]:.4f}")
    
    is_feas = problem.is_feasible(x_test)
    print(f"\nFeasible: {is_feas}")
    
    if is_feas:
        objectives = problem.evaluate(x_test)
        print(f"\nObjectives:")
        print(f"  f1 (-η)  = {objectives[0]:10.6f}  (η = {-objectives[0]:.6f})")
        print(f"  f2 (ΔP)  = {objectives[1]:10.2f}")
    
    print(f"\nTotal evaluations: {problem.evaluation_count}")
