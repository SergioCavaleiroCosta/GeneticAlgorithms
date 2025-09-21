"""Common type aliases for optimizers."""
from typing import List
from .types import NDArrayFloat
from .optimization_algorithm import OptimizationAlgorithm

ContinuousOptimizer = OptimizationAlgorithm[NDArrayFloat, float]
DiscreteOptimizer = OptimizationAlgorithm[List[int], float]
BinaryOptimizer = OptimizationAlgorithm[List[bool], float]
PermutationOptimizer = OptimizationAlgorithm[List[int], float]

__all__ = [
    "ContinuousOptimizer",
    "DiscreteOptimizer",
    "BinaryOptimizer",
    "PermutationOptimizer",
]
