"""Aggregation module re-exporting optimization protocols and types.

This file preserves backward compatibility for existing imports while the
codebase follows a one-class-per-file structure.
"""

from .types import ST, OT, OT_co, NDArrayFloat
from .optimization_result import OptimizationResult
from .optimization_problem import OptimizationProblem
from .optimization_algorithm import OptimizationAlgorithm
from .population_based_algorithm import PopulationBasedAlgorithm
from .aliases import (
    ContinuousOptimizer,
    DiscreteOptimizer,
    BinaryOptimizer,
    PermutationOptimizer,
)

__all__ = [
    "ST",
    "OT",
    "OT_co",
    "NDArrayFloat",
    "OptimizationResult",
    "OptimizationProblem",
    "OptimizationAlgorithm",
    "PopulationBasedAlgorithm",
    "ContinuousOptimizer",
    "DiscreteOptimizer",
    "BinaryOptimizer",
    "PermutationOptimizer",
]
