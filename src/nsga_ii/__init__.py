"""NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation.

This package provides a multi-objective optimization algorithm based on
the NSGA-II approach by Deb et al. (2002).

Main components:
- NSGAII: The main algorithm class implementing UpdateRule
- MultiObjectiveProblem: Protocol for multi-objective optimization problems
- pareto_utils: Utilities for Pareto dominance and non-dominated sorting
- crowding_distance: Crowding distance calculation for diversity preservation
"""

from .nsga_ii import NSGAII
from .multi_objective_problem import MultiObjectiveProblem
from .multi_objective_initializer import MultiObjectiveInitializer
from .pareto_utils import is_dominated, non_dominated_sort, crowding_distance_assignment

__all__ = [
    "NSGAII",
    "MultiObjectiveProblem",
    "MultiObjectiveInitializer",
    "is_dominated",
    "non_dominated_sort",
    "crowding_distance_assignment"
]