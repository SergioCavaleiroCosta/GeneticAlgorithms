"""Container class for optimization results."""
from typing import Generic, List, Optional, Dict
from .types import ST, OT


class OptimizationResult(Generic[ST, OT]):
    """
    Container for optimization results.
    
    Attributes:
        best_solution: The best solution found during optimization
        best_objective: The objective value of the best solution
        convergence_history: History of best objective values over iterations
        execution_time: Total execution time in seconds
        iterations: Number of iterations performed
        success: Whether the optimization was successful
        termination_reason: Reason for termination
        additional_info: Any additional algorithm-specific information
    """
    
    def __init__(
        self,
        best_solution: ST,
        best_objective: OT,
        convergence_history: Optional[List[OT]] = None,
        execution_time: Optional[float] = None,
        iterations: Optional[int] = None,
        success: bool = True,
        termination_reason: str = "Max iterations reached",
        additional_info: Optional[Dict[str, object]] = None
    ):
        self.best_solution = best_solution
        self.best_objective = best_objective
        self.convergence_history = convergence_history or []
        self.execution_time = execution_time
        self.iterations = iterations
        self.success = success
        self.termination_reason = termination_reason
        self.additional_info = additional_info or {}


__all__ = ["OptimizationResult"]
